from mpl_toolkits.mplot3d import axes3d
import numpy as np
import sys
import io
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc , style
import pandas as pd
import plotly
import matplotlib as mpl
import scipy as sp
from scipy.stats import weibull_min
from scipy import stats
import math
import  tensorflow as tf
from sklearn.preprocessing import minmax_scale
from tensorflow.keras.layers import Dropout, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# gpus = tf.config.experimental.list_physical_devices('GPU')
#
# if gpus :
#     try :
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError as e:
#         print(e)

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
mpl.rcParams['axes.unicode_minus']=False

df=pd.read_csv('C:/Users/LUNABIT/Desktop/　/open/train.csv',dtype='unicode',sep=',')



columns = list(df.columns[1:])
total_columns = columns
str_col = columns[2:6]

for i in str_col :
    columns.remove(i)

float_col = columns[:-1]

df[float_col] = df[float_col].astype('int')

df['가격'] =df['가격'].astype('float64')
y_data = df['가격']


del df['차량모델명']
del df['브랜드']
del df['판매도시']
del df['판매구역']
del df['가격']
del df['ID']
x_data = minmax_scale(df.values)
x_data = np.array(df)
x_data = x_data.reshape(57920,1,9)

# x_data = pd.DataFrame(x_data)

#val 데이터 나누기

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(9,activation='relu' , input_shape = (1,9)))
model.add(tf.keras.layers.Dense(9,activation='relu'))
model.add(tf.keras.layers.Dense(1))
adam = tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss = 'mse' , optimizer= adam  ,metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=5)
model.summary()
history = model.fit(x_data , y_data , epochs=5 ,verbose=2 ,batch_size=1 ,callbacks = [early_stop]  )

#Make a test data set

df_test=pd.read_csv('C:/Users/LUNABIT/Desktop/　/open/test.csv',dtype='unicode',sep=',')

columns_test = list(df_test.columns[1:])
total_columns_test = columns_test
str_col_test = columns_test[2:6]

for i in str_col_test :
    columns_test.remove(i)

float_col = columns_test[:-1]

df_test[float_col] = df_test[float_col].astype('int')

df_id = df_test['ID']


del df_test['차량모델명']
del df_test['브랜드']
del df_test['판매도시']
del df_test['판매구역']

del df_test['ID']
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(df_test)
x_data_test = scaler.transform(df_test)
x_data_test = np.array(df_test)
x_data_test = x_data_test.reshape(14480,1,9)

prediction = model.predict(x_data_test)

print(prediction)

pd_list= []

# pre_list = np.array(prediction)
# pre_list = pre_list.reshape(-1,1)
# pre_list = pre_list.tolist()
# for i in range(14480) :
#     re = pre_list[i][0]
#     pd_list.append(re)

# inverse_transformed = scaler.inverse_transform(pre_list)
# X= dict()
# result = pd.DataFrame(X)
#
# result['ID'] = df_id
# result['결과'] = pre_list
# # result['결과'] = scaler.inverse_transform(result['결과'])
# print(result)

# fig, loss_ax = plt.subplots()
#
# acc_ax = loss_ax.twinx()
#
# loss_ax.plot(hist.history['loss'],'y',label='train loss')
# loss_ax.plot(hist.history['val_loss'],'r',label='val loss')
# acc_ax.plot(hist.history['acc'],'b',label='train acc')
# acc_ax.plot(hist.history['val_acc'],'g',label='val acc')
#
# loss_ax.set_xlabel('epoch')
# loss_ax.set_ylabel('loss')
# acc_ax.set_ylabel('accuracy')
#
# loss_ax.legend(loc='upper left')
# acc_ax.legend(loc='lower left')
# plt.show()
# LSTM 할려면 3차원으로 변경해야함 // GPU 사용 가능함