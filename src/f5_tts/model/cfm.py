"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

from random import random
from typing import Callable
from f5_tts.model.utils import convert_char_to_pinyin

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = MelSpec(**mel_spec_kwargs)
        num_channels = self.mel_spec.n_mel_channels
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        reference_audio: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],
        duration: int | int["b"],
        *,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096, # 약 40초
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
        batch_size=1
    ):
        # raw wave
        if reference_audio.ndim == 2:
            reference_audio = self.mel_spec(reference_audio)
            reference_audio = reference_audio.permute(0, 2, 1)
            assert reference_audio.shape[-1] == self.num_channels
        reference_audio = reference_audio.to(next(self.parameters()).dtype)

        cond_seq_len, device = reference_audio.shape[1], reference_audio.device
        cond_length = torch.full((batch_size,), cond_seq_len, device=device, dtype=torch.long) # batch개의 텐서를 만들고, 값을 cond_seq_len으로 준다. 나중에 마스킹용으로 사용

        text = convert_char_to_pinyin(text)
        text = list_str_to_idx(text, self.vocab_char_map).to(device)[0] # 이게 생성할 script의 indexes
        print(len(text), cond_length)
        condition_length = torch.maximum(len(text), cond_length) # condition의 최소 길이.

        # duration
        cond_mask = lens_to_mask(condition_length)

        duration = duration + cond_seq_len
        duration = torch.tensor(duration, device=device, dtype=torch.long)
        duration = torch.maximum(length + 1, duration)  # 당연히 duration이 더 커야한다.
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # 앞은 reference audio mel이고, 뒤는 비어있는 컨디션 생성
        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # test for no ref audio
        if no_ref_audio:
            cond = torch.zeros_like(cond)
            step_cond = torch.zeros_like(step_cond)

        # neural ode
        def fn(t, x):
            if cfg_strength < 1e-5:
                pred = self.transformer(x=x, cond=step_cond, text=text, time=t, mask=mask, drop_audio_cond=False, drop_text=False)
                return pred
            else:
                pred = self.transformer(x=x, cond=step_cond, text=text, time=t, mask=mask, drop_audio_cond=False, drop_text=False)
                null_pred = self.transformer(x=x, cond=step_cond, text=text, time=t, mask=mask, drop_audio_cond=True, drop_text=True)
                
                return pred + cfg_strength * (pred - null_pred)

        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0
        t = torch.linspace(t_start, 1, steps, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        # t는 0~1까지 사이 값들. 한쪽에 몰릴 수는 있음.

        # main inference ode solver
        trajectory = odeint(
            fn, 
            # torch.compile(fn), 
            y0, 
            t, 
            **self.odeint_kwargs
        )

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        *,
        lens: int["b"] | None = None,  # noqa: F821
        noise_scheduler: str | None = None,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels
        # bs, seq_len, dim
        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device) # tokenizer와 같은 역할
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch # bs, max length of texts (짧은 친구들은 -1로 패딩)
        
        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)
        
        mask = lens_to_mask(lens, length=seq_len)  # useless here, as collate_fn will pad to max length in batch

        # get a random span to mask out for training conditionally
        mask_prob = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask) # 마스크되는 비율. 70 ~ 100% 사이
        rand_span_mask = mask_from_frac_lengths(lens, mask_prob) # 항상 양 끝에 False, 중간에 길게 True가 있는 형태

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # if want rigourously mask out padding, record in collate_fn in dataset.py, and pass in here
        # adding mask will use more memory, thus also need to adjust batchsampler with scaled down threshold for long sequences
        pred = self.transformer(
            x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text)

        # print("Pred : ", pred.sum())
        # print("Flow : ", flow.sum())
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask.bool()] # masked loss

        # # return loss.mean(), cond, pred
        return loss.mean()

class T5Conditioner(nn.Module):
    T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl"]
    
    T5_MODEL_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "t5-xl": 2048,
        "t5-xxl": 4096,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
        "google/flan-t5-xl": 2048,
        "google/flan-t5-xxl": 4096,
    }

    def __init__(
        self,
        t5_model_name: str = "t5-base",
        max_length: str = 128,
        enable_grad: bool = False,
    ):
        assert t5_model_name in self.T5_MODELS, f"Unknown T5 model name: {t5_model_name}"
        super().__init__()
        
        from transformers import T5EncoderModel, AutoTokenizer

        self.max_length = max_length
        self.enable_grad = enable_grad

        # Suppress logging from transformers
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
            model = T5EncoderModel.from_pretrained(t5_model_name).train(enable_grad).requires_grad_(enable_grad).to(torch.float16)
        except:
            print("Load model error")
            
        if self.enable_grad:
            self.model = model
        else: 
            self.__dict__["model"] = model

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.model.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.eval()
            
        # with torch.cuda.amp.autocast(dtype=torch.float16) and torch.set_grad_enabled(self.enable_grad):
        embeddings = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )["last_hidden_state"]
        
        embeddings = embeddings * attention_mask.unsqueeze(-1).float()
        return embeddings, attention_mask