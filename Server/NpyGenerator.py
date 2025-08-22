import os
import pandas as pd
import mediapipe as mp
import numpy as np
import cv2
import random

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# CSV 파일 경로 및 데이터셋 폴더
csv_file_path = './annotation.CSV'
video_folder_path = './dataset'

# CSV 파일 읽기
df = pd.read_csv(csv_file_path, encoding='EUC-KR')

labels = ['경찰','구급대','연락해주세요','도와주세요','빨리 와주세요']

label_indices = {label: i for i, label in enumerate(labels)}

# 비디오 파일 목록 불러오기
video_files = [f for f in os.listdir(video_folder_path) if f.endswith('.mp4')]


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = True
    
    return image, results

#아크코사인 각도 계산
def calculate_angle(v1, v2):
    v1 = v1.reshape(1, 3)
    v2 = v2.reshape(1, 3)
    v1=v1/np.linalg.norm(v1,axis=1)[:,np.newaxis]
    v2=v2/np.linalg.norm(v2,axis=1)[:,np.newaxis]
    jointangle = np.arccos(np.einsum('nt,nt->n',v1,v2))
    jointangle = np.degrees(jointangle)
   
    return jointangle

def extract_keypoints(results):
    angles = []

    # 상체 포즈 처리
    if results.pose_landmarks:
        upper_body_pose = np.array([[res.x, res.y, res.z] for idx, res in enumerate(results.pose_landmarks.landmark) if 11 <= idx <= 16])
        
        if upper_body_pose.shape[0] >= 2:
            # 왼쪽 팔 각도
            left_shoulder = upper_body_pose[0]
            left_elbow = upper_body_pose[1]
            left_wrist = upper_body_pose[2]
            v1 = left_elbow - left_shoulder
            v2 = left_wrist - left_elbow
            angles.append(calculate_angle(v1, v2))

            # 오른쪽 팔 각도
            right_shoulder = upper_body_pose[3]
            right_elbow = upper_body_pose[4]
            right_wrist = upper_body_pose[5]
            v1 = right_elbow - right_shoulder
            v2 = right_wrist - right_elbow
            angles.append(calculate_angle(v1, v2))

        # 상반신 포즈 벡터화
        vectors = np.diff(upper_body_pose, axis=0)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = np.divide(vectors, norms, where=norms > 0)
    else:
        vectors = np.zeros((12, 3))  # 상반신 포즈가 없음

    # 손 관절 처리
    lh_angles = []
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).reshape((21, 3))
        for i in range(len(lh) - 2):
            v1 = lh[i + 1] - lh[i]
            v2 = lh[i + 2] - lh[i + 1]
            angle = calculate_angle(v1, v2)
            lh_angles.append(angle)
    else:
        lh_angles = np.zeros(19)
    rh_angles = []
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).reshape((21, 3))
        for i in range(len(rh) - 2):
            v1 = rh[i + 1] - rh[i]
            v2 = rh[i + 2] - rh[i + 1]
            angle = calculate_angle(v1, v2)
            rh_angles.append(angle)
    else:
        rh_angles = np.zeros(19)

    # 리스트를 1차원 배열로 변환
    angles = np.array(angles).flatten()
    lh_angles = np.array(lh_angles).flatten()
    rh_angles = np.array(rh_angles).flatten()
   
    # 최종 키포인트 데이터
    keypoints = np.concatenate([vectors.flatten(), angles, lh_angles, rh_angles])
    
  
    
    return keypoints

def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_out = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    image_out.flags.writeable = True
    return image_out, results




def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def normalize_keypoints(keypoints, image_width, image_height):
    keypoints = keypoints.astype(np.float64)  # 배열을 float64로 변환
    keypoints[:, 0] /= image_width  # x 좌표 정규화
    keypoints[:, 1] /= image_height  # y 좌표 정규화
    return keypoints

# 이미지 밝기 조정
def adjust_brightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# 회전 및 기울이기
def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

# 노이즈 추가
def add_noise(image, noise_factor=0.05):
    noise = np.random.randn(*image.shape) * noise_factor
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

# 스케일링
def scale_image(image, scale_factor):

    height, width = image.shape[:2]
    new_dim = (int(width * scale_factor), int(height * scale_factor))
    return cv2.resize(image, new_dim, interpolation=cv2.INTER_LINEAR)



width = height = None


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    keypoints_list = []
    label_list = []
    max_len = 0
    
    for idx, row in df.iterrows():
        video_name = row['name'] + '.mp4'
        video_label = row['translation']
        
        if video_name not in video_files:
            continue

        video_path = os.path.join(video_folder_path, video_name)
        cap = cv2.VideoCapture(video_path)
        
        
        # 원본 + 증강 데이터를 각각 처리
        for augmentation in ['original', 'brightness', 'rotation', 'noise', 'scaling']:
            video_keypoints = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 비디오를 처음부터 다시 읽기
            
            # Set consistent augmentation parameters for the whole video
            if augmentation == 'brightness':
                brightness_factor = random.uniform(0.5, 1.5)
            elif augmentation == 'rotation':
                rotation_angle = random.uniform(-10, 10)
            elif augmentation == 'scaling':
                scale_factor = random.uniform(0.8, 1.2)
            elif augmentation == 'noise':
                noise_factor = random.uniform(0.01, 0.1)
        
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                # Apply consistent augmentation
                if augmentation == 'brightness':
                    frame = adjust_brightness(frame, factor=brightness_factor)
                elif augmentation == 'rotation':
                    frame = rotate_image(frame, angle=rotation_angle)
                elif augmentation == 'noise':
                    frame = add_noise(frame, noise_factor=noise_factor)
                elif augmentation == 'scaling':
                    frame = scale_image(frame, scale_factor=scale_factor)
                
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
                    array = extract_keypoints(results)
                    video_keypoints.append(array)
                
                cv2.imshow("test", image)
                cv2.waitKey(1)
            print(len(video_keypoints))
            # 현재 비디오의 프레임 수 확인
            if video_keypoints:
                video_keypoints = np.array(video_keypoints)            
                video_keypoints = normalize_keypoints(video_keypoints, width, height)
                keypoints_list.append(video_keypoints)
                label_list.append(label_indices.get(video_label, -1))        
                max_len = max(max_len, video_keypoints.shape[0])
            print(f"{video_name} {video_label} {labels[label_indices.get(video_label, -1)]} Augmentation: {augmentation}")
        cap.release() 
        

# 패딩을 적용하여 모든 비디오의 키포인트 배열을 동일한 길이로 맞춤
padded_keypoints_list = [np.pad(kp, ((0, max_len - kp.shape[0]), (0, 0)), mode='constant', constant_values=-1)
                         for kp in keypoints_list]

# 모든 비디오의 keypoints를 하나의 배열로 결합
keypoints_array = np.stack(padded_keypoints_list, axis=0)
label_array = np.array(label_list)

# NumPy 배열 저장
if keypoints_array.size > 0 and label_array.size > 0:
    np.save('keypoints.npy', keypoints_array)
    np.save('label.npy', label_array)
    print("Keypoints and labels saved successfully.")
else:
    print("No keypoints were extracted.")


print(keypoints_array.shape)
print(label_array.shape)
#%%
import numpy as np
k = np.load('keypoints.npy')
l = np.load('label.npy')
print(k.shape)
print(l.shape)
#%%
import numpy as np
from collections import Counter

# 저장된 라벨 파일 로드
labels = np.load('label_modify.npy')

# 각 클래스의 데이터 개수 세기
label_counts = Counter(labels)

# 클래스별 데이터 개수 출력
for label, count in label_counts.items():
    print(f"Class {label}: {count} samples")

# 총 데이터 수 출력
print(f"Total samples: {len(labels)}")
