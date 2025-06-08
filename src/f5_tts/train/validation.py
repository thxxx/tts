
import torch
from f5_tts.infer.utils_infer import (
    infer_process,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.train.custom_prompts import custom_prompts, creature_prompts, only_scripts
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from f5_tts.train.utils import make_html
import torchaudio
import numpy as np
from jiwer import wer
from difflib import SequenceMatcher
from torch.cuda.amp import autocast

def calculate_wer(reference, hypothesis):
    # Word Error Rate 계산
    return wer(reference, hypothesis)

def calculate_cer(reference, hypothesis):
    # Character Error Rate 계산
    # 레벤슈타인 거리 기반으로 일치 비율 계산
    matcher = SequenceMatcher(None, reference, hypothesis)
    cer = 1 - matcher.ratio()
    return cer

def validate(model, valid_loader, vocoder, mel_spec_type, output_dir, epoch, noise_scheduler, text_conditioner=None, is_caption=False, is_make_samples=False, is_whisper=False, device='cuda'):
    model.eval()
    valid_loss=0
    with torch.no_grad():
        for idx, batch in enumerate(valid_loader):
            mel = batch["mel"].to(device)
            mel_lengths = batch["mel_lengths"].to(device)
            scripts = batch["script"]
            caption = batch["caption"]
            mel_spec = mel.permute(0, 2, 1).to(device)
            
            # if is_caption:
            #     caption_embed, attention_mask = text_conditioner(caption, device=device)
            # else:
            caption_embed, attention_mask = None, None
            
            # with autocast():
            loss = model(
                mel_spec, text=scripts, lens=mel_lengths, noise_scheduler=noise_scheduler, caption_embed=caption_embed, attention_mask=attention_mask
            )
            valid_loss += loss.cpu().detach().item()
        print(valid_loss)

    wer = 0
    cer = 0
    if is_make_samples:
        speed = 1.0
        output_list = []
        for script in only_scripts:
            audio, final_sample_rate, spectragram = infer_process(
                None,
                "kill all. ", # 1초 짜리 zeros가 들어가니까
                script, 
                model, 
                vocoder, 
                mel_spec_type=mel_spec_type, 
                speed=speed,
                cfg_strength=2.0,
                no_ref_audio=True,
                caption_embed=None,
                attention_mask=None
            )
            output_list.append({
                'array': torch.stack((torch.tensor(audio).unsqueeze(dim=0), torch.tensor(audio).unsqueeze(dim=0)), dim=0).squeeze(),
                'caption': script,
                'script': script
            })
        
        for data in custom_prompts:
            caption = data['text']
            script = data['script']
            # caption_embed, attention_mask = text_conditioner(caption, device=device)
            caption_embed, attention_mask = None, None

            audio, final_sample_rate, spectragram = infer_process(
                None,
                "kill all. ", # 1초 짜리 zeros가 들어가니까
                script, 
                model, 
                vocoder, 
                mel_spec_type=mel_spec_type, 
                speed=speed,
                no_ref_audio=True,
                cfg_strength=2.0,
                caption_embed=caption_embed,
                attention_mask=attention_mask
            )
            output_list.append({
                'array': torch.stack((torch.tensor(audio).unsqueeze(dim=0), torch.tensor(audio).unsqueeze(dim=0)), dim=0).squeeze(),
                'caption': caption + " - " + script,
                'script': script
            })
        
        for data in creature_prompts:
            prefix_path = data['prefix_path']
            prefix_script = data['prefix_script']
            caption = data['caption']
            script = data['script']

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
            
            # caption_embed, attention_mask = text_conditioner(caption, device=device)
            caption_embed, attention_mask = None, None
            
            audio, final_sample_rate, spectragram = infer_process(
                voices[voice]["ref_audio"],
                voices[voice]["ref_text"],
                script, 
                model, 
                vocoder, 
                mel_spec_type=mel_spec_type, 
                speed=speed,
                no_ref_audio=False,
                cfg_strength=2.0,
                caption_embed=caption_embed,
                attention_mask=attention_mask
            )
            output_list.append({
                'array': torch.stack((torch.tensor(audio).unsqueeze(dim=0), torch.tensor(audio).unsqueeze(dim=0)), dim=0).squeeze(),
                'caption': "prefix + " + caption + " - " + script,
                'script': script
            })

        make_html(epoch, output_dir, output_list)
        
        if is_whisper:
            # load model and processor
            whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
            whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to(device)
            whisper_model.config.forced_decoder_ids = None
        
            resampler = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000)
            for i in range(len(output_list)):
                audio = output_list[i]['array'].squeeze()[0]
                script = output_list[i]['script']
                if audio.dtype == torch.float64:
                    continue
                resampled_audio = resampler(audio)
                
                input_features = whisper_processor(resampled_audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
                predicted_ids = whisper_model.generate(input_features)
                transcription = whisper_processor.batch_decode(predicted_ids.cpu(), skip_special_tokens=True)[0]
                
                wer_result = calculate_wer(script, transcription)
                cer_result = calculate_cer(script, transcription)
                wer += wer_result
                cer += cer_result
    
            del whisper_processor
            del whisper_model
            

    return valid_loss/len(valid_loader), wer, cer