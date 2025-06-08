import pandas as pd

# CSV 파일 경로
# csv1_path = "train_expresso.csv"
# csv2_path = "voice_sfx.csv"
# output_path = "train_merged_csv.csv"

# # CSV 파일 읽기
# df1 = pd.read_csv(csv1_path)
# df2 = pd.read_csv(csv2_path)

# # 두 데이터프레임 합치기
# merged_df = pd.concat([df1, df2], ignore_index=True)

# # audio_path 컬럼에서 문자열 대체
# import pandas as pd
# df = pd.read_csv("/home/khj6051/tts_sfx/datas/libri_test_clean.csv")
# df['audio_path'] = df['audio_path'].str.replace("/workspace/", "/home/khj6051/", regex=False)

# # # 합친 데이터프레임 저장
# df.to_csv("/home/khj6051/tts_sfx/datas/libri_test_clean.csv", index=False)

# import os
# import pandas as pd
# from tqdm import tqdm

# # 데이터프레임 읽기
# df = pd.read_csv('/home/khj6051/tts_sfx/datas/train_all_expresso_libritrain_voicesfx_verbals_20.csv')

# # 업데이트된 경로를 저장할 리스트
# updated_audio_paths = []

# # wavcaps/ 폴더 경로
# wavcaps_root = "/home/khj6051/data/wavcaps"
# # 파일 경로를 딕셔너리에 저장
# print("Indexing all files in wavcaps folder...")
# file_dict = {}
# for root, dirs, files in tqdm(os.walk(wavcaps_root)):
#     for file in files:
#         file_dict[file] = os.path.join(root, file)

# # audio_path 업데이트
# print("Updating audio_path values in dataframe...")
# updated_audio_paths = []
# for audio_path in tqdm(df['audio_path']):
#     if "wavcaps/" in audio_path:
#         # 파일명 추출 (e.g., audio132.flac)
#         filename = os.path.basename(audio_path)
        
#         # 딕셔너리에서 경로 검색
#         updated_audio_paths.append(file_dict.get(filename, audio_path))
#     else:
#         updated_audio_paths.append(audio_path)

# # audio_path 칼럼 업데이트
# df['audio_path'] = updated_audio_paths

# print("길이 ", len(df))
# output_csv_path = "train_all_expresso_libritrain_voicesfx_verbals_20.csv"
# df.to_csv(output_csv_path, index=False)
# import os
# # print(f"Updated CSV saved to {output_csv_path}")
# import pandas as pd
# df=pd.read_csv('/home/khj6051/tts_sfx/datas/libri_test_clean.csv')
# for i in range(len(df)):
#     if not os.path.exists(df.iloc[i]['audio_path']):
#         print(df.iloc[i])
# print("길이 ", len(df))
# df = df[df['duration']>0.3]
# df = df[df['duration']<30.0]
# output_csv_path = "libri_test_clean.csv"
# df.to_csv(output_csv_path, index=False)
# print("새로운 길이 ", len(df))

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
#     ' man ': 100,
#     'woman ': 100
# }
# print(sum(cres.values()))

# from collections import defaultdict
# from tqdm import tqdm
# from audiotools import AudioSignal
# df = pd.read_csv('/home/khj6051/data/csvs/filtered_0428_6.csv')

# new = defaultdict(int)
# for (k, v) in tqdm(cres.items()):
#     new[k] = len(df[df['caption'].str.contains(k)])
# print(new)

# 필터링
# filtered_df = df[df['caption'].apply(lambda x: any(keyword in x for keyword in cres.keys()))]
# # filtered_df = pd.read_csv('/home/khj6051/data/csvs/creatures.csv')

# print(len(filtered_df))

# filtered_df['script'] = ''
# filtered_df['audio_path'] = filtered_df['audio_path'].str.replace("/workspace/", "/home/khj6051/", regex=False)

# dds = [['audio_path', 'caption', 'duration', 'script']]
# for i in tqdm(range(len(filtered_df))):
#     d = filtered_df.iloc[i]
#     if not d['duration'] or pd.isna(d['duration']):
#         duration = AudioSignal(d['audio_path']).duration
#     else:
#         duration = d['duration']
    
#     if 'bbc-' not in d['audio_path'] and duration<30:
#         dds.append([
#             d['audio_path'],
#             d['caption'],
#             duration,
#             ''
#         ])
# print(len(dds))
# filtered_df.to_csv('creatures.csv')

# # 리스트를 DataFrame으로 변환 (첫 번째 행을 헤더로 설정)
# df = pd.DataFrame(dds[1:], columns=dds[0])

# # DataFrame을 CSV로 저장
# csv_path = 'output.csv'
# df.to_csv(csv_path, index=False)

# import pandas as pd
# import torchaudio

# # CSV 파일 읽기
# csv_file = "/home/khj6051/data/csvs/libritrain.csv"  # 처리할 CSV 파일 경로
# output_csv = "output2.csv"  # 저장할 CSV 파일 경로
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

# df1 = pd.read_csv('./output.csv')
# df2 = pd.read_csv('./output2.csv')
# df3 = pd.read_csv('./train_merged_csv.csv')

# print(len(df1))
# print(len(df2))
# print(len(df3))

# merged_df = pd.concat([df1, df2, df3], ignore_index=True)
# merged_df.to_csv('train_all.csv')

import pandas as pd

df = pd.read_csv('/home/khj6051/tts_sfx/datas/train_voicesfx_expresso_20.csv')
print(len(df[df['caption'].str.contains("robot computer")]))