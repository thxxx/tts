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
from f5_tts.model.utils import get_tokenizer
from f5_tts.train.utils import make_html
import torchaudio
import argparse
from supabase import create_client, Client
import requests


def download_file(url, save_path):
    try:
        # HTTP GET 요청으로 파일 가져오기
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 요청 성공 여부 확인
        
        # 디렉토리가 없는 경우 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 파일 저장
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"File downloaded successfully: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")

def delete_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)  # 파일 삭제
            print(f"File deleted: {file_path}")
        else:
            print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error deleting file: {e}")

# Supabase 프로젝트 URL과 API 키 설정
SUPABASE_URL = "https://hpxjdveijpuehyuykkos.supabase.co"
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhweGpkdmVpanB1ZWh5dXlra29zIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcwNDUyNzgxMSwiZXhwIjoyMDIwMTAzODExfQ.zo_ddufJU0SGR9ijLhzPFGZGJ6a46x7oByroj_qTkY8'
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

@torch.no_grad
def main(ckpt_path: Path, script:str, voice_type:str, input_audio_url:str, is_variation:bool, original_script:str, t_inter:float):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    
    cfg_strength = 2.0
    scale_phi = 0.75
    
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
    sway_sampling_coef = -1.0
    speed = 1.0
    fix_duration = None
    vocab_file = "./f5_tts/infer/examples/vocab.txt"
    tokenizer = "custom"
    ode_method = "euler"

    # load model
    model_cls = DiTPrepend
    model_cfg = dict(
        dim=1024, 
        depth=22, 
        heads=16, 
        ff_mult=2, 
        text_dim=512, 
        conv_layers=4
    )
    
    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
    if vocoder_name == "bigvgan":
        vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=False)
    else:
        vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=True, local_path=f"./f5_tts/vocoder")
    transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels)
    text_conditioner = T5Conditioner(t5_model_name="t5-base", max_length=32).to(device)
    text_conditioner.eval()
    
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

    dtype = torch.float32
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    # print(checkpoint)
    del checkpoint
    torch.cuda.empty_cache()

    # generate #
    t_inter = 0.1
    duplicate_test = False
    if input_audio_url is not None and not is_variation:
        # Use this voice
        file_name = input_audio_url.split("/")[-1]
        file_path = f'./reflist/{file_name}'
        download_file(input_audio_url, file_path)
        prefix_script = original_script
    
    if input_audio_url is not None and is_variation:
        # Random variation this voice
        file_name = input_audio_url.split("/")[-1]
        file_path = f'./reflist/{file_name}'
        download_file(input_audio_url, file_path)
        if t_inter is None:
            t_inter = 0.9
        else:
            t_inter = t_inter
        duplicate_test = True
        prefix_script = script
        print("Random variation test ", file_name, t_inter)
    if input_audio_url is None:
        response = supabase.table('voice_reference').select("*").eq("tag", voice_type).execute()
        file_name = response.data[0]['audio_url'].split("/")[-1]
        file_path = f"./reflist/{file_name}"
        prefix_script = response.data[0]['script']

        ref_list = os.listdir("reflist")
        if file_name not in ref_list:
            download_file(response.data[0]['audio_url'], file_path)
    
    main_voice = {"ref_audio": file_path, "ref_text": prefix_script}
    voices = {"main": main_voice}
    for voice in voices:
        voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
            voices[voice]["ref_audio"], voices[voice]["ref_text"]
        )
    
    audio, final_sample_rate, spectragram = infer_process(
        voices[voice]["ref_audio"],
        voices[voice]["ref_text"],
        script, 
        model, 
        vocoder, 
        mel_spec_type=mel_spec_type, 
        speed=speed,
        cfg_strength=cfg_strength,
        no_ref_audio=False,
        scale_phi=scale_phi,
        t_inter=t_inter,
        duplicate_test=duplicate_test,
        batch_size=5
    )

    print("Adua ", audio.shape)
    if vocoder_name == "bigvgan":
        array = torch.stack((torch.tensor(audio), torch.tensor(audio)), dim=0).squeeze()
    else:
        array = torch.stack((torch.tensor(audio).unsqueeze(dim=1), torch.tensor(audio).unsqueeze(dim=1)), dim=1).squeeze()
    print(array.shape)
    
    for idx, aud in enumerate(array.to(dtype).cpu().detach()):
        print(aud.shape)
        torchaudio.save(
            f"sample_{idx}.wav", aud, sample_rate=target_sample_rate, channels_first=True
        )
    delete_file(file_path)

    array = array.to(dtype).cpu().detach().numpy()
    return array # [batch_size, 2, audio_length]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--script", type=str, required=True)
    parser.add_argument("--voice_type", type=str, required=False)
    parser.add_argument("--input_audio_url", type=str, required=False) # if this is not None, it means reference audio is this.
    parser.add_argument("--original_script", type=str, required=False)
    parser.add_argument("--is_variation", type=bool, required=False) # if this is not None, it means reference audio is this.
    parser.add_argument("--t_inter", type=float, required=False) # if this is not None, it means reference audio is this.

    args = parser.parse_args()

    main(ckpt_path=args.path, script=args.script, voice_type=args.voice_type, input_audio_url=args.input_audio_url, is_variation=args.is_variation, original_script=args.original_script, t_inter=args.t_inter)










