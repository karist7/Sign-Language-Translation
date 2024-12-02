import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, Masking, TimeDistributed,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

# 데이터 로드
key_points = np.load('keypoints_modify.npy')
label_list = np.load('label_modify.npy')

print(key_points.shape)
print(label_list.shape)

# 레이블 정의
labels = ['어떤 사람이 흉기를 소지하고 있어요','어디선가 연기가 나요 불이 나고 있는 것 같아요',
          '가스가 새고 있는 것 같아요','도와주세요','빨리 와주세요']
num_classes = len(labels)

# 레이블을 one-hot 인코딩
label_list = to_categorical(label_list, num_classes=num_classes)

# 데이터를 학습 및 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(key_points, label_list, test_size=0.3, random_state=42)

# 마지막 차원에 채널을 추가 (배치 크기, 프레임 수, 482, 1)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
print(X_train.shape)

#%%
# 모델 정의
model = Sequential([
    # Masking을 통해 특정 값 패딩 무시 (선택적)
    Masking(mask_value=-1, input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    
    # TimeDistributed로 Conv2D 적용 (시간 축에 대해 2D Convolution 수행)
    TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01))),
    TimeDistributed(MaxPooling2D(pool_size=2)),
    TimeDistributed(BatchNormalization()),
    
    TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01))),
    TimeDistributed(MaxPooling2D(pool_size=2)),
    TimeDistributed(BatchNormalization()),
    
    # Flatten을 사용하여 LSTM 입력을 3D로 변환
    TimeDistributed(GlobalAveragePooling2D()),
    
    # LSTM Layers
    LSTM(64, return_sequences=True, activation='tanh', recurrent_dropout=0.25, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    
    LSTM(32, activation='tanh', recurrent_dropout=0.25, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    
    # Dense Layers
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(num_classes, activation='softmax')
])

model.summary()

# EarlyStopping 설정
earlystopping = EarlyStopping(
    monitor='val_loss', 
    min_delta=0.001, 
    patience=3, 
    verbose=1, 
    mode='auto', 
    restore_best_weights=True)

# 모델 컴파일
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy'])

# 모델 학습
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=64,
    callbacks=[earlystopping])

# 모델 저장
model.save('cnn_LSTM_modify.h5')

# 테스트 세트로 모델 평가
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# 학습 과정 시각화
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

# 예측 및 결과 확인
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

from sklearn.metrics import classification_report
print(classification_report(true_classes, predicted_classes, target_names=labels))
