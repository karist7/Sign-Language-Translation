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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = True
    return image, results

def calculate_angle(v1, v2):
    v1 = v1.reshape(1, 3)
    v2 = v2.reshape(1, 3)
    v1 = v1 / np.linalg.norm(v1, axis=1)[:, np.newaxis]
    v2 = v2 / np.linalg.norm(v2, axis=1)[:, np.newaxis]
    jointangle = np.arccos(np.einsum('nt,nt->n', v1, v2))
    jointangle = np.degrees(jointangle)
    return jointangle

def extract_keypoints(results):
    angles = []
    if results.pose_landmarks:
        upper_body_pose = np.array([[res.x, res.y, res.z] for idx, res in enumerate(results.pose_landmarks.landmark) if 11 <= idx <= 16])
        
        if upper_body_pose.shape[0] >= 2:
            left_shoulder = upper_body_pose[0]
            left_elbow = upper_body_pose[1]
            left_wrist = upper_body_pose[2]
            v1 = left_elbow - left_shoulder
            v2 = left_wrist - left_elbow
            angles.append(calculate_angle(v1, v2))

            right_shoulder = upper_body_pose[3]
            right_elbow = upper_body_pose[4]
            right_wrist = upper_body_pose[5]
            v1 = right_elbow - right_shoulder
            v2 = right_wrist - right_elbow
            angles.append(calculate_angle(v1, v2))

        vectors = np.diff(upper_body_pose, axis=0)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = np.divide(vectors, norms, where=norms > 0)
    else:
        vectors = np.zeros((12, 3))

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

    angles = np.array(angles).flatten()
    lh_angles = np.array(lh_angles).flatten()
    rh_angles = np.array(rh_angles).flatten()
    
    keypoints = np.concatenate([vectors.flatten(), angles, lh_angles, rh_angles])
    return keypoints

def normalize_keypoints(keypoints, image_width, image_height):
    keypoints[:, 0] /= image_width
    keypoints[:, 1] /= image_height
    return keypoints

# Function to predict using the TensorFlow Lite model
def predict_with_tflite_model(keypoints_array):
    # Get model input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare input data
    input_data = np.array(keypoints_array, dtype=np.float32)  # Ensure correct dtype
    interpreter.set_tensor(input_details[0]['index'], input_data[np.newaxis, ...])

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predictions = output_data[0]  # This will be your prediction vector
    print(predictions)
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)  # Safely get the predicted class index
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
    with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        cap = cv2.VideoCapture(file_path)

        # Enable GPU (using cv2.cuda module)
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:

            print("CUDA is enabled!")

            # GPU에서 비디오 캡처
            gpu_frame = cv2.cuda_GpuMat()

        keypoints_list = []
        frame_skip = 3  # Skip every 5th frame to speed up processing
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames to reduce processing load
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            # GPU에서 작업을 하기 위해서는 이미지를 GpuMat으로 전환
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                gpu_frame.upload(frame)
                # GPU에서 작업을 할 수 있는 형태로 변환
                image = gpu_frame.download()
            else:
                image = frame  # CPU에서 작업할 때는 일반 프레임 사용

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            image, results = mediapipe_detection(image, holistic)
            if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
                array = extract_keypoints(results)
                keypoints_list.append(array)
           
        cap.release()
        cv2.destroyAllWindows()

        keypoints_array = np.array(keypoints_list)
        keypoints_array = normalize_keypoints(keypoints_array, width, height)
        keypoints_array = np.pad(keypoints_array, ((0, 225 - keypoints_array.shape[0]), (0, 0)), mode='constant', constant_values=-1)

        # Use TensorFlow Lite model for prediction
        predicted_class_index = predict_with_tflite_model(keypoints_array)
        

        labels = ['경찰', '구급대', '연락해주세요', '도와주세요', '빨리 와주세요']
        predicted_label = labels[predicted_class_index]

    return jsonify({'predict': predicted_label}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
