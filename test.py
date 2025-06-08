from datasets import load_dataset

ds = load_dataset("ylacombe/expresso")

from tqdm import tqdm
styles = set()
for data in tqdm(ds['train']):
    styles.add(data['style'])

from tqdm import tqdm
import soundfile as sf
import numpy as np
import os

datas = [['audio_path', 'caption', 'duration', 'script']]
# 데이터셋 순회하면서 오디오 파일 저장
num = 0
for data in tqdm(ds['train']):
    audio_data = data['audio']['array']
    duration = data['audio']['array'].shape[0]/data['audio']['sampling_rate']
    save_directory = '../data/expresso'

    if data['speaker_id'] == 'ex01':
        gender = 'person'
    elif data['speaker_id'] == 'ex02':
        gender = 'woman'
    elif data['speaker_id'] == 'ex03':
        gender = 'man'
    elif data['speaker_id'] == 'ex04':
        gender = 'woman'

    text = data['text']
    if data['style'] == 'confused':
        caption = f'a {gender} saying in a confused tone'
    elif data['style'] == 'happy':
        caption = f'a {gender} happily saying'
    elif data['style'] == 'laughing':
        caption = f'a {gender} laughing and saying'
    elif data['style'] == 'sad':
        caption = f'a {gender} saying in a sad tone'
    elif data['style'] == 'whisper':
        num += 1
        caption = f'a {gender} whispering'
    elif data['style'] == 'enunciated':
        caption = f'a {gender} clearly saying'
    elif data['style'] == 'singing':
        caption = f'a {gender} singing'
    elif data['style'] == 'essentials':
        caption = f'a {gender} essentialy saying'
    elif data['style'] == 'emphasis':
        caption = f'a {gender} clearly saying'
    else:
        caption = f'a {gender} saying'

    sampling_rate = data['audio']['sampling_rate']

    # Mono인 경우 Stereo로 변환
    if audio_data.ndim == 1:
        audio_data = np.stack([audio_data, audio_data], axis=1)

    # 저장 경로 설정
    save_path = os.path.join(save_directory, data['audio']['path'])

    # 오디오 파일 저장
    sf.write(save_path, audio_data, sampling_rate)

    # datas.append([save_path, caption, duration, text])
print(num)
