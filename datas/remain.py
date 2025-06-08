import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
from audiotools import AudioSignal

# cres = {
#     'samurai': 108, 
#     'robot ': 19523, 
#     'monster ': 12515, 
#     'pirates': 74, 
#     'commander': 49, 
#     'dragon': 3094, 
#     'vampire': 585, 
#     'ghost ': 7842, 
#     'pirate': 932, 
#     'werewolf': 520, 
#     'zombie': 4896, 
#     'goblin': 883, 
#     ' orc ': 148, 
#     'troll': 1847, 
#     ' witch ': 682, 
#     'wizard': 1106, 
#     ' demon ': 1086, 
#     'ogre': 1618, 
#     'golem': 179, 'skeleton': 327, 'wraith': 85, 'ghoul': 138, 
#     'specter': 10, 'banshee': 196, 'fairy': 993, 'gremlin': 161, 'kraken': 75, 
#     'mermaid': 25, 'gargoyle': 52, 'mummy': 103, 'phantom': 277, 'harpy': 46, 'minotaur': 31, 'chimera': 7, 'griffin': 6, 'yeti ': 594, 'cyclops': 186, 'hobgoblin': 1, 
#     'darth vader': 4, 
#     'dinosaur': 2291, 
#     'satan': 57, 'gorilla': 763, ' mage ': 46, 'minions': 54,
#     ' man ': 16028,
#     'woman ': 14104
# }
# print(sum(cres.values()))

# df = pd.read_csv('/home/khj6051/data/csvs/filtered_0428_6.csv')
# new = defaultdict(int)
# for (k, v) in tqdm(cres.items()):
#     new[k] = len(df[df['caption'].str.contains(k)])
# print(new)

# filtered_df = df[df['caption'].apply(lambda x: any(keyword in x for keyword in cres.keys()))]
# filtered_df['script'] = ''
# filtered_df['audio_path'] = filtered_df['audio_path'].str.replace("/workspace/", "/home/khj6051/", regex=False)

# dds = [['audio_path', 'caption', 'duration', 'script']]
# for i in tqdm(range(len(filtered_df))):
#     d = filtered_df.iloc[i]
#     if not d['duration'] or pd.isna(d['duration']):
#         duration = AudioSignal(d['audio_path']).duration
#     else:
#         duration = d['duration']
    
#     if 'bbc-' not in d['audio_path'] and duration<25 and duration>0.3:
#         dds.append([
#             d['audio_path'],
#             d['caption'],
#             duration,
#             ''
#         ])
# print(len(dds))
# # 리스트를 DataFrame으로 변환 (첫 번째 행을 헤더로 설정)
# df = pd.DataFrame(dds[1:], columns=dds[0])

# # DataFrame을 CSV로 저장
# csv_path = 'verbals.csv'
# df.to_csv(csv_path, index=False)




# import torchaudio

# # CSV 파일 읽기
# csv_file = "/home/khj6051/data/csvs/libritrain.csv"  # 처리할 CSV 파일 경로
# output_csv = "libritrain.csv"  # 저장할 CSV 파일 경로
# base_path = "/home/khj6051/data/"

# # CSV 파일 로드
# df = pd.read_csv(csv_file)

# # 'train-clean-100' 텍스트를 포함한 행 필터링
# df_filtered = df[df['filename'].str.contains('train-clean-100')].copy()

# # 파일 경로 추가
# df_filtered['audio_path'] = base_path + df_filtered['filename']

# # 오디오 길이 계산 함수
# def calculate_duration(audio_path):
#     try:
#         waveform, sample_rate = torchaudio.load(audio_path)
#         duration = waveform.shape[1] / sample_rate
#         return duration
#     except Exception as e:
#         print(f"Error loading {audio_path}: {e}")
#         return None

# # 오디오 길이 계산 및 30초 미만 필터링
# df_filtered['duration'] = df_filtered['audio_path'].apply(calculate_duration)
# df_filtered = df_filtered[df_filtered['duration'] < 30].copy()

# # 칼럼 이름 변경
# df_filtered.rename(columns={'transcript': 'script'}, inplace=True)
# df_filtered['caption'] = ''

# # 결과 저장
# df_filtered.to_csv(output_csv, index=False)

# print(f"Filtered data saved to {output_csv}")




# df = pd.read_csv('verbals.csv')
# df2 = pd.read_csv('./train_expresso.csv')
# df3 = pd.read_csv('./voice_sfx.csv')
# df4 = pd.read_csv('./libritrain.csv')

# print(len(df2))
# print(len(df3))
# print(len(df4))

# merged_df = pd.concat([df, df2, df3, df4], ignore_index=True)
# merged_df.to_csv('train_all_expresso_libritrain_voicesfx_verbals.csv')

df = pd.read_csv('train_voicesfx_expresso.csv')
print(len(df[df['duration']>20]))
print(len(df[df['duration']>25]))
print(len(df))
df = df[df['duration']<=20]
# df['audio_path'] = df['audio_path'].str.replace("/workspace/", "/home/khj6051/", regex=False)
print(len(df))
df = df[~df['audio_path'].str.contains('/bbc-sfx')]
print(len(df))
df.to_csv('train_voicesfx_expresso_20.csv', index=False)