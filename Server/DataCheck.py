import os
import pandas as pd

# CSV 파일 경로와 dataset 폴더 경로 설정
csv_file_path = './annotation.CSV'  # 업로드된 CSV 파일 경로
dataset_folder = './dataset'  # 'dataset' 폴더 경로

# CSV 파일 읽기
df = pd.read_csv(csv_file_path,encoding='euc-kr')

# 'name' 열에서 파일 목록 가져오기
csv_files = df['name'].tolist()

dataset_files = os.listdir(dataset_folder)

# dataset 폴더 안의 파일 목록 가져오기
dataset_files_no_ext = [os.path.splitext(file)[0] for file in dataset_files]

# 결과 확인


missingFiles = [file for file in csv_files if file not in dataset_files_no_ext]

print(missingFiles)

#%%
import numpy as np


# 저장된 label_list 로드
label_list = np.load('label2.npy')

# label_list가 이미 정수형 배열이라면, 바로 개수를 셀 수 있음
class_counts = np.bincount(label_list)

# 클래스명과 함께 출력
labels = ['어떤 사람이 흉기를 소지하고 있어요','어디선가 연기가 나요 불이 나고 있는 것 같아요',
          '가스가 새고 있는 것 같아요','도와주세요','빨리 와주세요','none']

for i, label in enumerate(labels):
    print(f"{label}: {class_counts[i]}개")
#%%
from transformers import pipeline

# 문장 생성 파이프라인 로드
text_generator = pipeline("text-generation", model="gpt2")

# 입력 텍스트
input_text = "나 다리 아프다"

# 문장 생성
output = text_generator(input_text, max_length=50, num_return_sequences=1, truncation=True)


# 생성된 텍스트 출력
generated_text = output[0]['generated_text']
print(generated_text)  # 자연스럽게 변환된 문장