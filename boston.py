import pandas as pd;
import numpy as np;
import tensorflow as tf;
from sklearn import datasets;


#보스턴 주택 데이터셋 로딩

b_data = datasets.load_boston();
x_data = b_data.data;
y_data = b_data.target;
print(x_data.shape);
print(y_data.shape);



# 데이터 정규화
from sklearn.preprocessing import MinMaxScaler;
x_data_scaled= MinMaxScaler().fit_transfrom(x_data);
print(x_data);
print(x_data_scaled);

# 학습 데이터셋 분할
from sklearn.model_selection import train_test_split, KFold;
x_train,x_test,y_train,y_test=train_test_split(
    x_data_scaled, y_data,
    test_size=0.2,
    shuffle=True,
    random_state=12
);
print(x_train.shape,y_train.shape);
print(x_test.shape,y_test.shape);

# 심층 신경말 구축
from tensorflow.keras import Sequential;
from tensorflow.keras.layers import Dense;

def build_model(num_input=1):
    model = Sequential();
    model.add(Dense(64, activation='relu',input_dim=num_input)); # 히든 레이어1
    model.add(Dense(32, activation='relu')); # 히든 레이어2 -> 이는 다중 어쩌구 그거임
    model.add(Dense(1, activation='linear')); # 최종적인 아웃풋
    model.compile(optimizer='adam',loss='mse',metrics=['mae']);
    return model;

model = build_model(num_input=13);
model.summary();

# 모델 훈련
history1 = model.fit(x_train,y_train,epochs=300,verbose=1);
print(model.evaluate(x_test,y_test));

# 교차 검증 1
model2=build_model(13);
history2 = model.fit(x_train,y_train,validation_split=0.25,epochs=300,verbose=1)
print(model2.evaluate(x_test,y_test));

# 교차 검증2
# 여기서는 훈련 데이터셋을 3개로 등분하여
k=3;
kfold=KFold(n_splits=k,random_state=777, shuffle=True);

mae_list=[];

for train_index,val_index in kfold.split(x_train):
    x_train_fold, x_val_fold =x_train[train_index], x_train[val_index];
    y_train_fold, y_val_fold =y_train[train_index], y_train[val_index];
    kmodel = build_model(13);
    kmodel.fit(x_train_fold,y_train_fold,epochs=300,validation_data=(x_val_fold,y_val_fold));
    result=kmodel.evaluate(x_test,y_test);
    mae_list.append(kmodel);

print(mae_list);