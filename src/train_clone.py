import os
import re
from pathlib import Path
import numpy as np
from f5_tts.infer.utils_infer import (
    infer_process,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiTPrepend, CFM
from f5_tts.train.utils import draw_plot
from f5_tts.train.validation import validate
import torch
from f5_tts.model.utils import (
    get_tokenizer,
    convert_char_to_pinyin,
    list_str_to_idx,
    lens_to_mask,
    mask_from_frac_lengths
)
from transformers import T5EncoderModel, AutoTokenizer
from torch.cuda.amp import autocast
from accelerate import Accelerator, DistributedDataParallelKwargs
from f5_tts.model.cfm import T5Conditioner
from tqdm import tqdm
import random

from f5_tts.custom.dataset import CustomDataset, collate_fn, DynamicBatchSampler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# -----------------------------------------

mel_spec_type = "vocos"
vocoder_name = mel_spec_type

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None
preset = "/home/khj6051/tts_sfx"

# -----------------------------------------

# t5_model_name = "t5-base"
# text_conditioner = T5Conditioner(t5_model_name="t5-base", max_length=32).to(device)
# text_conditioner.eval()

accelerator = Accelerator(
    mixed_precision='no'
)

model_cls = DiTPrepend
model_cfg = dict(
    dim=1024, 
    depth=22, 
    heads=16, 
    ff_mult=2, 
    text_dim=512, 
    conv_layers=4
)

vocab_file = "./f5_tts/infer/examples/vocab.txt"
tokenizer = "custom"
ode_method = "euler"

vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
if vocoder_name == "bigvgan":
    vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=False)
else:
    vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=True, local_path=f"{preset}/src/f5_tts/vocoder")

transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels)

mel_spec_kwargs=dict(
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    n_mel_channels=n_mel_channels,
    target_sample_rate=target_sample_rate,
    mel_spec_type=mel_spec_type,
)

odeint_kwargs=dict(
    method=ode_method,
)

model = CFM(
    transformer=transformer,
    mel_spec_kwargs=mel_spec_kwargs,
    odeint_kwargs=odeint_kwargs,
    vocab_char_map=vocab_char_map,
    frac_lengths_mask=(0.7, 1.0),
    audio_drop_prob=0.3,
    cond_drop_prob=0.2,
    caption_drop_prob=1.0
).to(device)

# model.transformer.caption_mlp만 학습할까? 나머지는.. 조금만

# ckpt_path = f"{preset}/src/weights_1203_prepend_vocos_cross_all/model_0.pt"
# checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
# model.load_state_dict(checkpoint, strict=False)

if vocoder_name == "bigvgan":
    ckpt_path = f"{preset}/ckpts/model_1250000.pt"
else:
    ckpt_path = f"{preset}/ckpts/model_1200000.pt"
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

checkpoint["model_state_dict"] = {
    k.replace("ema_model.", ""): v
    for k, v in checkpoint["ema_model_state_dict"].items()
    if k not in ["initted", "step"]
}

# patch for backward compatibility, 305e3ea
for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
    if key in checkpoint["model_state_dict"]:
        del checkpoint["model_state_dict"][key]

model.load_state_dict(checkpoint["model_state_dict"], strict=False)

del checkpoint
torch.cuda.empty_cache()

batch_size   = 16
lr           = 0.0000012
weight_decay = 0.001
betas        = (0.9, 0.999)
sample_rate  = 24000
train_duration  = 20.0

num_workers = 8
num_epochs = 100
output_dir   = 'weights_1206_vocos_clone_qwen_filter'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

train_dataset = CustomDataset(
    # f"{preset}/datas/train_all_expresso_libritrain_voicesfx_verbals_20.csv",
    f"{preset}/datas/train_qwen_voicesfx_expresso_20_filterrobotvoice.csv",
    target_sample_rate = target_sample_rate,
    mode           = "train",
    hop_length     = hop_length,
    n_mel_channels = n_mel_channels,
    win_length     = win_length,
    n_fft          = n_fft,
    mel_spec_type  = mel_spec_type,
    preprocessed_mel = False,
    mel_spec_module = None
)
valid_dataset = CustomDataset(
    f"{preset}/datas/valid_expresso.csv",
    target_sample_rate = target_sample_rate,
    mode           = "validation",
    hop_length     = hop_length,
    n_mel_channels = n_mel_channels,
    win_length     = win_length,
    n_fft          = n_fft,
    mel_spec_type  = mel_spec_type,
    preprocessed_mel = False,
    mel_spec_module = None
)

# Define Train Sampler
steps_per_epoch = 5999
num_samples_per_epoch = steps_per_epoch * batch_size  # 총 샘플 수 = 스텝 수 * 배치 크기 # train 96,000 samples per epoch
train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=num_samples_per_epoch)  # Train Sampler with replacement

# sampler = SequentialSampler(train_dataset)
# batch_sampler = DynamicBatchSampler(
#     sampler, 32000, max_samples=32000, random_seed=None, drop_last=False
# )

train_loader = DataLoader(
    train_dataset,
    collate_fn=collate_fn,
    batch_size=batch_size,
    sampler=train_sampler,
    # sampler=batch_sampler,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    collate_fn=collate_fn,
    shuffle=True,  # Shuffle validation data without sampler
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=2,
)

libri_valid_dataset = CustomDataset(
    f"{preset}/datas/libri_test_clean.csv",
    target_sample_rate = target_sample_rate,
    mode           = "validation",
    hop_length     = hop_length,
    n_mel_channels = n_mel_channels,
    win_length     = win_length,
    n_fft          = n_fft,
    mel_spec_type  = mel_spec_type,
    preprocessed_mel = False,
    mel_spec_module  = None
)
libri_valid_loader = DataLoader(libri_valid_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, prefetch_factor=2)

trainer = {
    'train_losses': [],
    'valid_losses': [],
    'libri_valid_losses': [],
    'wer': [],
    'cer': [],
    'lrs': [],
}

# for param in model.parameters():
#     param.requires_grad = False

# # caption_mlp의 파라미터만 requires_grad를 True로 설정
# for param in model.transformer.caption_mlp.parameters():
#     param.requires_grad = True

# # Optimizer를 caption_mlp의 파라미터로만 초기화
# optimizer = torch.optim.AdamW(
#     model.transformer.caption_mlp.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
# )
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=50, # 10 epoch마다 1/10이 된다.
    cycle_mult=1.0,
    max_lr=lr,
    min_lr=lr/10,
    warmup_steps=0,
    gamma=0.2 # 한 사이클 돌 때마다 max lr이 10%가 된다.
)

noise_scheduler = None # 지금 세팅 안되어있음
max_grad_norm = 1.0

model, train_loader, valid_loader, libri_valid_loader = accelerator.prepare(model, train_loader, valid_loader, libri_valid_loader)

print("start training")

gradient_accumulation_steps = 4
for epoch in range(num_epochs):
    model.train()
    
    epoch_loss = 0
    tqdm_bar = tqdm(total=len(train_loader), desc="F5-TTS Training")
    
    for idx, batch in enumerate(train_loader):
        with accelerator.accumulate(model):
            mel = batch["mel"]
            mel_lengths = batch["mel_lengths"]
            scripts = batch["script"]
            caption = batch["caption"]
            mel_spec = mel.permute(0, 2, 1).to(device)
            mel_lengths = mel_lengths.to(device)

            # with autocast():
            loss = model(
                mel_spec, text=scripts, lens=mel_lengths, noise_scheduler=noise_scheduler
            )
            # loss.backward()
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if (idx + 1) % gradient_accumulation_steps == 0 or (idx + 1) == len(train_loader):
                if max_grad_norm > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.cpu().detach().item()
            tqdm_bar.update()
            # print(loss)

            if idx%2000 == 1999:
                scheduler.step()
                trainer['train_losses'].append(epoch_loss/idx)
                trainer['lrs'].append(optimizer.param_groups[0]['lr'])
                
                draw_plot('train_losses', trainer, output_dir=output_dir)
                draw_plot('lrs', trainer, output_dir=output_dir)
                # 텍스트 파일에 쓰기
                with open(f'./{output_dir}/middle_logs.txt', 'a') as file:
                    file.write(f"\nEpoch - {epoch} : {epoch_loss/idx}\n")
                
                torch.cuda.empty_cache()
    
    print(epoch_loss/idx)
    with open(f'./{output_dir}/middle_logs.txt', 'a') as file:
        file.write(f"\nEpoch - {epoch} : {epoch_loss/idx}\n")
    scheduler.step()
    trainer['train_losses'].append(epoch_loss/idx)
    trainer['lrs'].append(optimizer.param_groups[0]['lr'])
    
    draw_plot('train_losses', trainer, output_dir=output_dir)
    draw_plot('lrs', trainer, output_dir=output_dir)

    model.eval()
    # unwrapped_model = model
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.state_dict(), f'./{output_dir}/model_{epoch}.pt')

    valid_expresso_loss, wer, cer = validate(
        unwrapped_model,
        valid_loader,
        vocoder,
        mel_spec_type=mel_spec_type,
        output_dir=output_dir,
        epoch=epoch,
        noise_scheduler=None,
        is_caption=False,
        is_make_samples=True,
        is_whisper=True,
        device='cuda'
    )
    valid_libri_loss, _, _ = validate(
        unwrapped_model,
        libri_valid_loader,
        vocoder,
        mel_spec_type=mel_spec_type,
        output_dir=output_dir,
        epoch=epoch,
        noise_scheduler=None,
        is_caption=False,
        is_make_samples=False,
        is_whisper=False,
        device='cuda'
    )
    
    trainer['valid_losses'].append(valid_expresso_loss)
    trainer['libri_valid_losses'].append(valid_libri_loss)
    
    draw_plot('valid_losses', trainer, output_dir=output_dir)
    draw_plot('libri_valid_losses', trainer, output_dir=output_dir)
    
    trainer['wer'].append(wer)
    trainer['cer'].append(cer)
    
    draw_plot('wer', trainer, output_dir=output_dir)
    draw_plot('cer', trainer, output_dir=output_dir)
    
    # 텍스트 파일에 쓰기
    with open(f'./{output_dir}/logs.txt', 'a') as file:
        file.write(f"\nEpoch - {epoch}\n")
        file.write(f"Train loss : {epoch_loss/len(train_loader)}\n")
        file.write(f"valid_expresso_loss : {valid_expresso_loss}\n")
        file.write(f"valid_libri_loss : {valid_libri_loss}\n")
        
        file.write(f"Wer : {wer}\n")
        file.write(f"Cer : {cer}\n")

    del unwrapped_model
    torch.cuda.empty_cache()

