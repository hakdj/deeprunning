import numpy as np
import pandas as pd
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATA_PATH = './csv_data/nocolorinfo'

train_df = pd.read_csv(DATA_PATH + '/train.csv')
val_df = pd.read_csv(DATA_PATH + '/val.csv')
test_df = pd.read_csv(DATA_PATH + '/test.csv')

train_df.head()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지 제네레이터를 정의합니다.
train_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)

def get_steps(num_samples, batch_size):
    if (num_samples % batch_size) > 0 :
        return (num_samples // batch_size) + 1
    else :
        return num_samples // batch_size

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

model = Sequential()

# 입력 데이터의 형태를 꼭 명시해야 합니다.
model.add(Flatten(input_shape = (112, 112, 3))) # (112, 112, 3) -> (112 * 112 * 3)
model.add(Dense(128, activation = 'relu')) # 128개의 출력을 가지는 Dense 층
model.add(Dense(64, activation = 'relu')) # 64개의 출력을 가지는 Dense 층
model.add(Dense(11, activation = 'sigmoid')) # 11개의 출력을 가지는 신경망

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['acc'])

batch_size = 32
class_col = ['black', 'blue', 'brown', 'green', 'red', 'white',
             'dress', 'shirt', 'pants', 'shorts', 'shoes']

# Make Generator

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col = 'image',
    y_col = class_col,
    target_size = (112, 112),
    color_mode='rgb',
    class_mode='raw',
    batch_size=batch_size,
    shuffle = True,
    seed=42
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=None,
    x_col = 'image',
    y_col = class_col,
    target_size = (112, 112),
    color_mode='rgb',
    class_mode='raw',
    batch_size=batch_size,
    shuffle=True
)
print('--------------------------------------------------------------',len(train_generator));
model.fit(train_generator,
         validation_data = val_generator,
         epochs = 5, verbose=0)
test_datagen = ImageDataGenerator(rescale = 1./255)

# y_col: None, class_mode: None이므로
# test_generator는 image만 반환하고, label은 반환하지 않습니다.
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,
    x_col = 'image',
    y_col = None,
    target_size = (112, 112),
    color_mode='rgb',
    class_mode=None,
    batch_size=batch_size,
    shuffle = False
)

result = model.predict(test_generator)

print(result.shape)
print(result)

# 테스트 데이터 예측

import matplotlib.pyplot as plt
import cv2  # pip install opencv-python

image = cv2.imread(test_df['image'][7])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image);

desc = zip(class_col,list(result[7]))
desc_list = list(desc);
type = desc_list[0:6];
color = desc_list[6:11];
type = sorted(type, key=lambda z:z[1], reverse=True)
color = sorted(color, key=lambda z:z[1], reverse=True)

print(type[0][0],type[0][1])
print(color[0][0], color[0][1])

plt.title(type[0][0]);
plt.show();

