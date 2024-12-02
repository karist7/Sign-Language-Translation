import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, GRU,Masking
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import EarlyStopping
import mediapipe as mp
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.python.client import device_lib

#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#tf.config.list_physical_devices('GPU')

key_points = np.load('keypoints_modify.npy')
label_list = np.load('label_modify.npy')

labels = ['경찰','구급대','연락해주세요','도와주세요','빨리 와주세요']
label_indices = {label: i for i, label in enumerate(labels)}

num_classes = len(labels)
label_list = to_categorical(label_list, num_classes=num_classes)

X_train, X_temp, y_train, y_temp = train_test_split(key_points, label_list, test_size=0.2, random_state=42)

# 나머지 데이터(20%)를 다시 검증(10%)과 테스트(10%)로 나눔
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


model = Sequential([
    Masking(mask_value=-1,input_shape=(225,55)),
    LSTM(512,activation='tanh',return_sequences=False),
    Dense(1024, activation='relu'),
    BatchNormalization(), 
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Flatten(),
    Dense(num_classes, activation='softmax')
        
    ])
model.summary()

earlystopping = EarlyStopping(
    monitor='val_loss', 
    min_delta=0.001, 
    patience=10, 
    verbose=1, 
    mode='auto', 
    restore_best_weights=True)


#optimizer = Adam(learning_rate=0.0001)
optimizer = SGD(learning_rate=0.00005, momentum=0.7, decay=1e-4)
model.compile(
    optimizer=optimizer, 
    loss='categorical_crossentropy', 
    metrics=['accuracy'])


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=300,
   
    callbacks=[earlystopping])
# 모델 저장
model.save('classify_LSTM_SGD3.h5')

# 테스트 세트로 모델 평가
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

   
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Model accuracy')
plt.show()

plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.show()

predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# 예측 결과 확인
from sklearn.metrics import classification_report
print(classification_report(true_classes, predicted_classes, target_names=labels))

#%%
import tensorflow as tf

# 모델을 로드하고 변환기 설정
model = tf.keras.models.load_model('classify_LSTM_SGD3.h5')

# TFLiteConverter를 사용하여 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# TensorFlow 연산을 선택하도록 설정
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

# 실험적 플래그 설정 (TensorListReserve 관련 문제 해결)
converter._experimental_lower_tensor_list_ops = False

# 모델을 TFLite 형식으로 변환
tflite_model = converter.convert()

# 변환된 모델을 파일로 저장
with open('classify_LSTM_SGD3.tflite', 'wb') as f:
    f.write(tflite_model)
