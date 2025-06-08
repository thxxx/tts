import os
import re
import torch
from transformers import T5EncoderModel, AutoTokenizer
from pathlib import Path
from f5_tts.infer.utils_infer import (
    infer_process,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model.cfm import T5Conditioner
from f5_tts.model import DiTPrepend, CFM
from tqdm import tqdm
from f5_tts.model.utils import get_tokenizer
from f5_tts.train.utils import make_html
from audiotools import AudioSignal


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("test")

# -----------------------------------------
mel_spec_type = "vocos"
vocoder_name = mel_spec_type

epoch = 'cfg_2'
output_dir = f"./gens_prepend_voicefx_{mel_spec_type}/"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

if vocoder_name == "vocos":
    ckpt_path = "/home/khj6051/tts_sfx/src/weights_1202_prepend_6/model_2.pt"
elif vocoder_name == "bigvgan":
    ckpt_path = "/home/khj6051/tts_sfx/src/weights_1203_prepend_bigvgan_3/model_0.pt"
    # ckpt_path = "/home/khj6051/tts_sfx/ckpts/model_1250000.pt"

cfg_strength = 2.0
scale_phi = 0.75

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32  # 16, 32
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None
vocab_file = "./f5_tts/infer/examples/vocab.txt"
tokenizer = "custom"
ode_method = "euler"

checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
use_ema = False

# -----------------------------------------

vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
if vocoder_name == "vocos":
    vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=True, local_path="/home/khj6051/tts_sfx/src/f5_tts/vocoder")
else:
    vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=False)

model_cls = DiTPrepend
model_cfg = dict(
    dim=1024, 
    depth=22, 
    heads=16, 
    ff_mult=2, 
    text_dim=512, 
    conv_layers=4
)
transformer = model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels)
text_conditioner = T5Conditioner(t5_model_name="t5-base", max_length=32).to(device)
text_conditioner.eval()

ode_method = "euler"

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
).to(device)
model.load_state_dict(checkpoint)
model.eval()

del checkpoint
torch.cuda.empty_cache()

test_prompts = [
    {
        'prefix_path': "/home/khj6051/tts_sfx/valid_data/evil-robot-says-upgrade-sound-effect-055388974_nw_prev.mp3",
        'prefix_script': "Upgrade. ",
        'caption': "evil robot says",
        'script': "Tell me everything. Apple is quiet delicious."
    },
    {
        'prefix_path': "/home/khj6051/tts_sfx/valid_data/alien-says-nothing-can-stop-sound-effect-057386747_nw_prev.mp3",
        'prefix_script': "nothing can stop me now",
        'caption': "alien says",
        'script': "To live or to die, that is the question."
    },
    {
        'prefix_path': "/home/khj6051/tts_sfx/valid_data/goblin-saying-ahaha-sfx-sound-effect-027830110_nw_prev.mp3",
        'prefix_script': "ahaha",
        'caption': "goblin saying",
        'script': "You can not beat me!"
    },
    {
        'prefix_path': "/home/khj6051/tts_sfx/valid_data/cartoon-robot-saying-powering-down-sound-effect-102512872_nw_prev.mp3",
        'prefix_script': "powering down",
        'caption': "cartoon robot saying",
        'script': "Welcome to the optimizer AI."
    },
    {
        'prefix_path': "/home/khj6051/tts_sfx/valid_data/cartoon-robot-saying-powering-down-sound-effect-102512872_nw_prev.mp3",
        'prefix_script': "powering down",
        'caption': "cartoon robot saying",
        'script': "Version two is coming.. keep your head up!"
    },
    {
        'prefix_path': "/home/khj6051/tts_sfx/valid_data/monster-saying-i-love-you-sound-effect-234404303_nw_prev.mp3",
        'prefix_script': "I love you.",
        'caption': "monster saying",
        'script': "This is my home. Do you want to get introduce?"
    },
    {
        'prefix_path': "/home/khj6051/tts_sfx/valid_data/dragon_you_are_come_from_lake_town.mp3",
        'prefix_script': "you are come from lake town.",
        'caption': "dragon saying",
        'script': "This is my home. Do you want to get introduce?"
    },
    {
        'prefix_path': "/home/khj6051/tts_sfx/valid_data/cartoon-robot-saying-powering-down-sound-effect-102512872_nw_prev.mp3",
        'prefix_script': "powering down",
        'caption': "cartoon robot saying",
        'script': "This is my home. Do you want to get introduce?"
    }
]


test_prompts = [
    {
        'prefix_path': "/home/khj6051/tts_sfx/valid_data/alien-saying-welcome-earthling-sound-effect-033846347_nw_prev.mp3",
        'prefix_script': "welcome earthling",
        'caption': "a alien says",
        'script': "This is my home. Do you want to get introduce? Ha people why come here."
    },
    {
        'prefix_path': "/home/khj6051/tts_sfx/valid_data/alien-saying-welcome-earthling-sound-effect-033846347_nw_prev.mp3",
        'prefix_script': "welcome earthling",
        'caption': "a alien says",
        'script': "This is my home. Do you want to get introduce? Ha people why come here."
    },
    {
        'prefix_path': "/home/khj6051/tts_sfx/valid_data/alien-saying-welcome-earthling-sound-effect-033846347_nw_prev.mp3",
        'prefix_script': "welcome earthling",
        'caption': "a alien says",
        'script': "Go back, boy."
    },
    {
        'prefix_path': "/home/khj6051/tts_sfx/valid_data/alien-saying-welcome-earthling-sound-effect-033846347_nw_prev.mp3",
        'prefix_script': "welcome earthling",
        'caption': "a alien says",
        'script': "Go back, boy."
    },
]

output_list = []

is_voice = True

for data in tqdm(test_prompts):
    caption = data['caption']
    script = data['script']

    prefix_path = data['prefix_path']
    prefix_script = data['prefix_script']
    main_voice = {
        "ref_audio": prefix_path,
        "ref_text": prefix_script,
    }
    voices = {
        "main": main_voice
    }
    for voice in voices:
        voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
            voices[voice]["ref_audio"], voices[voice]["ref_text"]
        )
    
    caption_embed, attention_mask = text_conditioner(caption, device=device)
    
    audio, final_sample_rate, spectragram = infer_process(
        voices[voice]["ref_audio"] if is_voice else None,
        voices[voice]["ref_text"] if is_voice else "Hello",
        script, 
        model, 
        vocoder, 
        mel_spec_type=mel_spec_type, 
        speed=speed,
        cfg_strength=cfg_strength,
        no_ref_audio=is_voice==False,
        caption_embed=caption_embed,
        attention_mask=attention_mask,
        scale_phi=scale_phi
    )

    prefix_audio = AudioSignal(prefix_path).to_mono().audio_data.squeeze()
    print(prefix_audio.shape)
    audio = torch.tensor(audio)
    audio = torch.concat([prefix_audio, audio], dim=0)
    if vocoder_name == "bigvgan":
        array = torch.stack((audio, audio), dim=0).squeeze()
    else:
        array = torch.stack((audio.unsqueeze(dim=0), audio.unsqueeze(dim=0)), dim=0).squeeze()
    
    output_list.append({
        'array': array,
        'caption': f"prefix + " + caption + " - " + script,
        'script': script
    })

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
make_html(epoch, output_dir, output_list)
