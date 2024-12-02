import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, Masking, BatchNormalization, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# 데이터 로드
key_points = np.load('keypoints_modify.npy')
label_list = np.load('label_modify.npy')

labels = ['경찰','구급대','연락해주세요','도와주세요','빨리 와주세요']
label_indices = {label: i for i, label in enumerate(labels)}

num_classes = len(labels)
label_list = to_categorical(label_list, num_classes=num_classes)

# 데이터 분할
X_train, X_temp, y_train, y_temp = train_test_split(key_points, label_list, test_size=0.2, random_state=42)

# 나머지 데이터(20%)를 다시 검증(10%)과 테스트(10%)로 나눔
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Conv1D 모델
model = Sequential([
    Masking(mask_value=-1, input_shape=(X_train.shape[1], X_train.shape[2])),  # (시퀀스 길이, 특성 수)
   
   # Conv1D 레이어로 지역적 패턴 추출
   Conv1D(128, 3, activation='relu', padding='same'),  # Conv1D 레이어
   MaxPooling1D(2),  # MaxPooling 레이어
   Dropout(0.3),
   Conv1D(256, 3, activation='relu', padding='same'),  # Conv1D 레이어
   MaxPooling1D(2),  # MaxPooling 레이어
   Dropout(0.3),

  
   
   Flatten(),  # 평탄화
   
   # Fully connected 레이어
   Dense(1024, activation='relu'),  
   BatchNormalization(),
   Dropout(0.3),  # Dropout
   Dense(256, activation='relu'),  # Fully connected 레이어
   Dense(num_classes, activation='softmax')  # 출력 레이어
])

model.summary()

# 조기 종료 콜백 설정
earlystopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=10,
    verbose=1,
    mode='auto',
    restore_best_weights=True
)

# 옵티마이저 설정
optimizer = Adam(learning_rate=0.0001)

# 모델 컴파일
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 학습
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    callbacks=[earlystopping]
)

# 모델 저장
model.save('classify_Conv1D_Adam.h5')

# 테스트 세트로 모델 평가
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# 정확도와 손실 그래프
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

# 예측 및 성능 평가
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

from sklearn.metrics import classification_report
print(classification_report(true_classes, predicted_classes, target_names=labels))
