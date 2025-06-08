from f5_tts.infer.utils_infer import (
    infer_process,
    load_vocoder
)
from f5_tts.model.utils import default
import librosa
import torchaudio
from PIL import Image
from f5_tts.model.modules import MelSpec
import torch
import numpy as np
from tqdm import tqdm

hop_length=256
n_mel_channels=100
n_fft=1024
win_length=1024
target_sample_rate = 24000
mel_spec_type = 'bigvgan'
vocoder_name = "bigvgan"
mel_spec_module = None

preset = "/home/khj6051/tts_sfx"

audios = [
    'alien-saying-welcome-earthling-sound-effect-033846347_nw_prev.mp3',
    'zombie-or-monster-says-i-sound-effect-079567563_nw_prev.mp3',
    # 'alien-says-nothing-can-stop-sound-effect-057386747_nw_prev.mp3',
    # 'evil-robot-says-access-denied-sound-effect-055388964_nw_prev.mp3',
    # 'soldier-saying-attention-02-sound-effect-221151515_nw_prev.mp3',
    # 'goblin-saying-ahaha-sfx-sound-effect-027830110_nw_prev.mp3',
    # 'monster-says-ready-fight-07-sound-effect-157993476_nw_prev.mp3',
    # 'zombie-says-come-here-sound-effect-079567533_nw_prev.mp3',
    # 'monster-saying-i-love-you-sound-effect-234404303_nw_prev.mp3'
]

if vocoder_name == 'bigvgan':
    vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=False)
elif vocoder_name == 'vocos':
    vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=True, local_path=f"{preset}/src/f5_tts/vocoder")
mel_spectrogram = default(
    mel_spec_module,
    MelSpec(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    ),
)

for audio_path in tqdm(audios):
    pp = '../valid_data/'
    audio, sr = librosa.load(pp + audio_path, sr=24000)

    audio = torch.tensor(audio).unsqueeze(dim=0)
    mel_spec = mel_spectrogram(audio)
    mel_spec = mel_spec.squeeze(0)  # '1 d seq_len -> d seq_len'

    # NumPy 배열을 이미지로 변환
    # 3. 데이터 정규화 (0~255 범위로 스케일링)
    # mel_numpy = mel_spec.numpy()
    # mel_normalized = (mel_numpy - mel_numpy.min()) / (mel_numpy.max() - mel_numpy.min())  # 0~1로 정규화
    # mel_scaled = (mel_normalized * 255).astype(np.uint8)  # 0~255로 스케일링

    # # 4. PIL을 사용하여 이미지로 저장
    # image = Image.fromarray(mel_scaled)
    # image.save("mel_spectrogram.png")

    vocoder = vocoder.to('cuda')
    mel_spec = mel_spec.to('cuda')
    if vocoder_name == 'bigvgan':
        out = vocoder(mel_spec.unsqueeze(dim=0))
        torchaudio.save(pp + "bigvgan/" + audio_path, out.squeeze(dim=0).cpu().detach(), sr)
    elif vocoder_name == 'vocos':
        out = vocoder.decode(mel_spec.unsqueeze(dim=0))
        torchaudio.save(pp + "vocos/" + audio_path, out.cpu().detach(), sr)
