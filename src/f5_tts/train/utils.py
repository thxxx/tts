import matplotlib.pyplot as plt
import numpy as np
import os
import io
import base64
from pathlib import Path
import librosa
import torch
from einops import rearrange, repeat, reduce
import torch.nn.functional as F
import librosa.display
import torchaudio

def draw_plot(key, trainer, output_dir):
    plt.figure(figsize=(10, 6))
    plt.title(f"{key}")
    plt.plot(trainer[key])
    plt.savefig(f'./{output_dir}/{key}.png')
    plt.close()

def write_html(captions: list[str], audio_paths: list[Path], image_paths: list[Path]):
    html = """
    <html>
    <head>
        <title>Audio and Mel Preview</title>
    </head>
    <body>
        <table border="1">
            <tr>
                <th>Audio</th>
                <th>Mel</th>
            </tr>
    """

    # names = ["real", "pred", "gen"]
    for row_name, audio_path, image_path in zip(captions, audio_paths, image_paths):
        with open(audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        html += f"""
            <tr>
                <td>
                    <p>{row_name}</p>
                    <audio controls>
                        <source src="data:audio/flac;base64,{audio_base64}" type="audio/flac">
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td>
                    <img src="data:image/png;base64,{image_base64}" alt="{row_name} Mel Spectrogram" style="width:100%;">
                </td>
            </tr>
        """

    html += """
        </table>
    </body>
    </html>
    """

    return html

def make_html(epoch, output_dir, data):
    # 저장 디렉토리 생성
    audio_dir = f'./{output_dir}/epoch_{epoch}/audio_files'
    spectrogram_dir = f'./{output_dir}/epoch_{epoch}/spectrogram_images'
    total_dir = f'./{output_dir}/generations'
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(spectrogram_dir, exist_ok=True)
    os.makedirs(total_dir, exist_ok=True)

    audio_paths = []
    image_paths = []
    captions    = []
    sampling_rate = 24000
    n_mels        = 128  # Mel 필터의 개수
    hop_length    = 512  # Mel-spectrogram의 해상도를 결정하는 파라미터
    
    for i, item in enumerate(data):
        caption = item['caption']
        audio_array = item['array']
        
        # 오디오 파일로 저장
        audio_path = f"{audio_dir}/audio_{i}.wav"
        torchaudio.save(audio_path, audio_array, sampling_rate)
        audio_paths.append(audio_path)

        # Mel-spectrogram 생성
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_array.numpy()[0], sr=sampling_rate, n_mels=n_mels, hop_length=hop_length)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(mel_spectrogram_db, sr=sampling_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
        spectrogram_path = f"{spectrogram_dir}/spectrogram_{i}.png"
        plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        image_paths.append(spectrogram_path)
        captions.append(caption)
        
    # HTML 코드 종료 부분
    html_content = write_html(captions, audio_paths, image_paths)
    
    # HTML 파일 저장
    with open(f"{total_dir}/audio_s_{epoch}.html", "w", encoding="utf-8") as file:
        file.write(html_content)