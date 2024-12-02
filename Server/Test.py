import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
#현재 가장 정확한거 SGD2
model = load_model('classify_Conv1D_Adam.h5')


# 비디오 파일 경로
video_path = './uploads/20241112_120201.mp4'
#video_path = './testvideo.mp4'
#video_path = './dataset/index5_5.mp4'
 
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = True
    return image, results

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



def normalize_keypoints(keypoints, image_width, image_height):
    print(keypoints.shape)
    keypoints[:, 0] /= image_width  # x 좌표 정규화
    keypoints[:, 1] /= image_height  # y 좌표 정규화
    return keypoints

# Initialize MediaPipe
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    cap = cv2.VideoCapture(video_path)

    keypoints_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        image, results = mediapipe_detection(frame, holistic)
        if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
            array = extract_keypoints(results)
            keypoints_list.append(array)
       
        cv2.waitKey(1)
    
    cap.release()
    cv2.destroyAllWindows()
    # Convert to NumPy array
    keypoints_array = np.array(keypoints_list)
    print(keypoints_array.shape)
    # Normalize keypoints array
    keypoints_array = normalize_keypoints(keypoints_array, width, height)
    keypoints_array = np.pad(keypoints_array,((0, 225 - keypoints_array.shape[0]), (0, 0)), mode='constant', constant_values=-1)
                           

   

    # Predict
    predictions = model.predict(keypoints_array[np.newaxis, ...])
    print(predictions)
    predicted_class_index = np.argmax(predictions, axis=1)

    # Define labels
    labels = ['경찰','구급대','연락해주세요','도와주세요','빨리 와주세요']
    predicted_label = labels[predicted_class_index[0]]
    print(f"Predicted Label: {predicted_label}")
#%%
import numpy as np
k = np.load('keypoints_modify.npy')
l = np.load('label_modify.npy')

print(l)