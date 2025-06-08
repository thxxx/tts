import os
import pandas as pd
from tqdm import tqdm

# df = pd.read_csv('/home/khj6051/data/csvs/filtered_0428_6.csv')
# print(len(df))

# cres = {
#     'samurai': 108, 
#     'robot ': 19523, 
#     'monster ': 12515, 
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
#     "cartoon": 10,
#     "evil ": 10,
#     "god ": 12,
# }
# creature_list = cres.keys()
# verbal_list = ['voice', " speak", " talk"]

# vdf = df[df['duration']<20][df['caption'].str.len() < 70][df['caption'].apply(lambda x: any(word in x for word in creature_list))][df['caption'].apply(lambda x: any(word in x for word in verbal_list))]
# print("vdf len : ", len(vdf))
# # vdf.to_csv("all_voice_creture.csv")

# vdf['audio_path'] = vdf['audio_path'].str.replace("/workspace/", "/home/khj6051/", regex=False)
# df = vdf[~vdf['audio_path'].str.contains('/bbc-sfx')]

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
# df = pd.DataFrame(dds)
# df.to_csv("svae.csv")

# df = pd.read_csv("svae.csv")

# # 데이터프레임 읽기
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
# print(df.head(10))
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
# output_csv_path = "all_voice_options.csv"
# df.to_csv(output_csv_path, index=False)


# from io import BytesIO
# from urllib.request import urlopen
# import librosa
# from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

# device = 'cuda'
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
# model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map=device)

# def get_answer(audio_path, prompt):
#     audio, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)

#     conversation = [
#         {"role": "user", "content": [
#             {
#                 "type": "audio", 
#                 "audio_array": audio,
#                 "audio_url": ""
#             },
#             {"type": "text", "text": prompt},
#         ]},
#     ]
#     text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
#     audios = []
#     for message in conversation:
#         if isinstance(message["content"], list):
#             for ele in message["content"]:
#                 if ele["type"] == "audio":
#                     audios.append(ele['audio_array'])

#     inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True, sampling_rate=sr).to('cuda')
#     inputs.input_ids = inputs.input_ids

#     generate_ids = model.generate(**inputs, max_length=256)
#     generate_ids = generate_ids[:, inputs.input_ids.size(1):]

#     response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

#     return response

# df = pd.read_csv("all_voice_options.csv")

# dds = [['audio_path', 'caption', 'duration', 'script']]
# # for i in tqdm(range(len(df))):
# for i in tqdm(range(len(df))):
#     data = df.iloc[i]
#     res = get_answer(data['audio_path'], "Is this sound speech? Return yes or no.")
#     if 'yes' in res.lower():
#         res2 = get_answer(data['audio_path'], "Please transcribe the english speech.")
#         print(res, data['audio_path'], 'caption - ', data['caption'], '\n', res2)
#         dds.append([
#             data['audio_path'],
#             data['caption'],
#             data['duration'],
#             res2,
#         ])
# print(len(dds))
# df = pd.DataFrame(dds)
# df.to_csv("testset.csv")

# df = pd.read_csv("testset.csv")
# df = df[df['script'] != "Sorry, but I can't assist with that."]
# df = df[df['script'] != "Sorry, I can't get what you mean."]
# df = df[~df['script'].str.contains("I'm sorry, but")]
# print(len(df))
# print(len(df[df['script'].str.count("'") == 2]))
# print(len(df[df['script'].str.count("'") == 2][df['script'].str.contains('speech')]))
# print(len(df[df['script'].str.count("'") == 2][df['script'].str.contains('audio')]))
# print(len(df[df['script'].str.count("'") == 0]))

# twospeech = df[df['script'].str.count("'") == 2][df['script'].str.contains('speech')]
# twoaudio = df[df['script'].str.count("'") == 2][df['script'].str.contains('audio')]
# three = df[df['script'].str.count("'") == 3]
# zero = df[df['script'].str.count("'") == 0]

# twospeech['script'] = twospeech['script'].str.extract(r"'(.*)'", expand=False)
# twoaudio['script'] = twoaudio['script'].str.extract(r"'(.*)'", expand=False)
# three['script'] = three['script'].str.extract(r"'(.*)'", expand=False)

# one = df[df['script'].str.count("'") == 1][~df['script'].str.contains('transcript')]

# merged_df = pd.concat([one, zero, twospeech, twoaudio, three], ignore_index=True)
# merged_df.to_csv('new_vocal_set.csv')

df1 = pd.read_csv("./train_voicesfx_expresso_20.csv")
print(len(df1))
print(len(df1[df1['audio_path'].str.contains('expresso')]))
print(len(df1[df1['caption'].str.contains("robot computer")]))
df1 = df1[~df1['caption'].str.contains("robot computer")]
df2 = pd.read_csv("./new_vocal_set.csv")
merged_df = pd.concat([df1, df2], ignore_index=True)
merged_df.to_csv('train_qwen_voicesfx_expresso_20_filterrobotvoice.csv')
print(len(merged_df))