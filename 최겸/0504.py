from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

# VGG16 모델 로드, 최상위 레이어는 포함하지 않음
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# 모델의 파라미터를 동결
for layer in base_model.layers:
    layer.trainable = False

# 커스텀 레이어 추가
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

# 전체 모델 구성
model = Model(inputs=base_model.input, outputs=output)

import os

# 디렉토리 생성 함수
def create_directories(base_path, dir_names):
    for dir_name in dir_names:
        dir_path = os.path.join(base_path, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory created at: {dir_path}")

# 학습 및 검증 데이터셋 디렉토리 생성
base_path = '/content'
directories = ['train_dataset', 'validation_dataset']
create_directories(base_path, directories)

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지 경로 설정
train_dir = '/content/train_dataset'
validation_dir = '/content/validation_dataset'

# 데이터 증강을 위한 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 픽셀 값을 0~1 범위로 스케일링
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)  # 검증 데이터에는 증강 적용 안 함

# 훈련용 데이터 로더
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # 모델에 맞는 이미지 크기
    batch_size=32,
    class_mode='binary'  # 이진 분류 문제
)

# 검증용 데이터 로더
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16

# VGG16 기본 모델 로드
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# 기존 레이어는 학습되지 않도록 설정
for layer in base_model.layers:
    layer.trainable = False

# 새로운 분류 레이어 추가
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# history = model.fit(
#     train_generator,
#     steps_per_epoch=100,  # 한 에폭에 100개 배치를 사용 (2000 이미지 기준)
#     epochs=10,
#     validation_data=validation_generator,
#     validation_steps=50  # 검증 단계에서 50개의 배치를 사용
# )
history = model.fit(
    train_generator,
    steps_per_epoch=31,  # 수정된 스텝 수
    epochs=10,
    validation_data=validation_generator,
    validation_steps=6  # 수정된 검증 스텝 수
)

# 모델 평가
val_loss, val_acc = model.evaluate(validation_generator, steps=50)
print(f"Validation Accuracy: {val_acc}")
print(f"Validation Loss: {val_loss}")

# 새로운 이미지에 대한 예측
# pred = model.predict(new_image_processed)  # new_image_processed는 예측을 위해 처리된 이미지

