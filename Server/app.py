from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from flask_cors import CORS

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="classify_LSTM_SGD3.tflite")
interpreter.allocate_tensors()

# Initialize MediaPipe for holistic pose
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    # BGR -> RGB, no-write, process, back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image.flags.writeable = True
    return image, results

def calculate_angle(v1, v2):
    v1 = v1.reshape(1, 3)
    v2 = v2.reshape(1, 3)
    v1 = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    v2 = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
    jointangle = np.arccos(np.einsum('nt,nt->n', v1, v2))
    jointangle = np.degrees(jointangle)
    return jointangle

def extract_keypoints(results):
    # === 기존 feature 구성 유지 ===
    angles = []
    if results.pose_landmarks:
        upper_body_pose = np.array(
            [[res.x, res.y, res.z] for idx, res in enumerate(results.pose_landmarks.landmark) if 11 <= idx <= 16]
        )

        if upper_body_pose.shape[0] >= 2:
            # 왼팔
            left_shoulder, left_elbow, left_wrist = upper_body_pose[0], upper_body_pose[1], upper_body_pose[2]
            v1 = left_elbow - left_shoulder
            v2 = left_wrist - left_elbow
            angles.append(calculate_angle(v1, v2))

            # 오른팔
            right_shoulder, right_elbow, right_wrist = upper_body_pose[3], upper_body_pose[4], upper_body_pose[5]
            v1 = right_elbow - right_shoulder
            v2 = right_wrist - right_elbow
            angles.append(calculate_angle(v1, v2))

        vectors = np.diff(upper_body_pose, axis=0)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = np.divide(vectors, norms, where=norms > 0)
    else:
        vectors = np.zeros((12, 3))

    # 손 각도
    lh_angles = []
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).reshape((21, 3))
        for i in range(len(lh) - 2):
            v1 = lh[i + 1] - lh[i]
            v2 = lh[i + 2] - lh[i + 1]
            lh_angles.append(calculate_angle(v1, v2))
    else:
        lh_angles = np.zeros(19)

    rh_angles = []
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).reshape((21, 3))
        for i in range(len(rh) - 2):
            v1 = rh[i + 1] - rh[i]
            v2 = rh[i + 2] - rh[i + 1]
            rh_angles.append(calculate_angle(v1, v2))
    else:
        rh_angles = np.zeros(19)

    angles = np.array(angles).flatten()
    lh_angles = np.array(lh_angles).flatten()
    rh_angles = np.array(rh_angles).flatten()

    keypoints = np.concatenate([vectors.flatten(), angles, lh_angles, rh_angles])
    return keypoints  # shape: (D,)

# NOTE: normalize_keypoints는 호출하지 않습니다 (MediaPipe 좌표는 이미 정규화, 각도/벡터엔 무의미)

def predict_with_tflite_model(keypoints_array):
    """
    keypoints_array: (T, D)  # T는 모델 기대 길이(예: 225)에 맞춰 준비되어야 함
    returns: predicted class index (int)
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 입력 텐서 shape 확인 (예: (1, 225, D))
    # 필요 시 reshape
    input_data = np.asarray(keypoints_array, dtype=np.float32)
    if input_details[0]['shape'].ndim == 3:
        # TFLite가 (1, T, D) 형태를 기대한다고 가정
        interpreter.set_tensor(input_details[0]['index'], input_data[np.newaxis, ...])
    else:
        # 다른 형태라면 맞춰서 reshape 필요
        interpreter.set_tensor(input_details[0]['index'], input_data[np.newaxis, ...])

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])  # (1, C)
    predictions = output_data[0]
    predicted_class_index = int(np.argmax(predictions))
    return predicted_class_index

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/upload_video', methods=['POST'])
def file_down():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    labels = ['경찰', '구급대', '연락해주세요', '도와주세요', '빨리 와주세요']

    # 슬라이딩 윈도우 설정
    MICRO_WIN = 3   # 창 길이(프레임)
    HOP = 3         # 매 프레임 갱신
    MODEL_WIN = 225 # 모델이 기대하는 시퀀스 길이(패딩용)


    with mp_holistic.Holistic(static_image_mode=False,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        cap = cv2.VideoCapture(file_path)
        keypoints_seq = []  # 전체 시퀀스(다운샘플 후)
        frame_skip = 3
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue  # 다운샘플
            image = frame  
            image, results = mediapipe_detection(image, holistic)

            if (results.pose_landmarks or
                results.left_hand_landmarks or
                results.right_hand_landmarks):
                feat = extract_keypoints(results)  # (D,)
                keypoints_seq.append(feat)

        cap.release()
        cv2.destroyAllWindows()

    # 추출 실패 처리
    if not keypoints_seq:
        return jsonify({'error': 'No keypoints extracted'}), 400

    seq = np.asarray(keypoints_seq, dtype=np.float32)  # (T, D)
    T, D = seq.shape[0], seq.shape[1]
    # 슬라이딩 윈도우 설정


    
    # 3프레임 슬라이딩 윈도우로 창별 추론 → 다수결
    preds = []
    if T < MICRO_WIN:
        # 프레임이 3보다 적으면, 있는 만큼으로 창을 만들고 MODEL_WIN으로 패딩 후 1회 추론
        pad_len = MODEL_WIN - T
        pad_block = np.full((max(0, pad_len), D), -1.0, dtype=np.float32) 
        win_in = np.vstack([seq, pad_block]) if pad_len > 0 else seq
        pred = predict_with_tflite_model(win_in)
        preds.append(pred)
    else:
        for i in range(0, T - MICRO_WIN + 1, HOP):
            win = seq[i:i+MICRO_WIN]  # (3, D)
            # 모델 입력 길이(225)에 맞게 뒤쪽 -1 패딩
            pad_len = MODEL_WIN - MICRO_WIN
            pad_block = np.full((pad_len, D), -1.0, dtype=np.float32)
            win_in = np.vstack([win, pad_block])  # (225, D)
            pred = predict_with_tflite_model(win_in)
            preds.append(pred)

    # 최종 예측: 다수결
    counts = np.bincount(preds, minlength=len(labels))
    predicted_class_index = int(np.argmax(counts))
    predicted_label = labels[predicted_class_index]

    return jsonify({'predict': predicted_label}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
