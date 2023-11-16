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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


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


list_train = []

for i in df['가격'] :
    list_train.append(i*100)
df['가격'] = list_train

y_train = df['가격']

car_modelname = df['차량모델명'].unique()
brand = df['브랜드'].unique()
sell_city = df['판매도시'].unique()
sell_zone = df['판매구역'].unique()

for i , j in zip(car_modelname , range(len(car_modelname))) :
    df['차량모델명'] = df['차량모델명'].replace(i,j)

for i , j in zip(brand , range(len(brand))) :
    df['브랜드'] = df['브랜드'].replace(i,j)

for i , j in zip(sell_city , range(len(sell_city))) :
    df['판매도시'] = df['판매도시'].replace(i,j)

for i , j in zip(sell_zone , range(len(sell_zone))) :
    df['판매구역'] = df['판매구역'].replace(i,j)

empty_carmodel ={}
data_carmodel = pd.DataFrame(empty_carmodel)
data_carmodel['차량모델명'] = car_modelname
data_carmodel['치환숫자'] = range(len(car_modelname))

empty_brand ={}
data_brand = pd.DataFrame(empty_brand)
data_brand['브랜드'] = brand
data_brand['치환숫자'] = range(len(brand))

empty_sell_city ={}
data_sell_city = pd.DataFrame(empty_sell_city)
data_sell_city['판매도시'] = sell_city
data_sell_city['치환숫자'] = range(len(sell_city))

empty_sell_zone ={}
data_sell_zone = pd.DataFrame(empty_sell_zone)
data_sell_zone['판매구역'] = sell_zone
data_sell_zone['치환숫자'] = range(len(sell_zone))


del df['가격']
del df['ID']

scaler = MinMaxScaler()
scaler.fit(df)
x_train = scaler.transform(df)


df_test=pd.read_csv('C:/Users/LUNABIT/Desktop/　/open/test.csv',dtype='unicode',sep=',')
columns_test = list(df_test.columns[1:])
total_columns_test = columns_test
str_col_test = columns_test[2:6]

for i in str_col_test :
    columns_test.remove(i)

float_col = columns_test[:-1]

df_test[float_col] = df_test[float_col].astype('int')

df_id = df_test['ID']

df_test_sell_city = list(df_test['판매도시'].unique())

df_test_sell_zone = list(df_test['판매구역'].unique())

print(len(df_test_sell_zone))

for i in list(sell_city) :
   try:
    df_test_sell_city.remove(i)
   except:
       continue





for i , j in zip(list(df_test_sell_city) , range(3224 ,3224+len(df_test_sell_city) )) :
    data_sell_city = data_sell_city.append({'판매도시' : i , '치환숫자' : j} , ignore_index=True)


for i , j in zip(data_carmodel['차량모델명'] , data_carmodel['치환숫자']) :
    df_test['차량모델명'] = df_test['차량모델명'].replace(i,j)

for i , j in zip(data_brand['브랜드'] , data_carmodel['치환숫자']) :
    df_test['브랜드'] = df_test['브랜드'].replace(i,j)

for i , j in zip(data_sell_city['판매도시'] , data_sell_city['치환숫자'] ) :
    df_test['판매도시'] = df_test['판매도시'].replace(i,j)


for i , j in zip(data_sell_zone['판매구역'] , data_sell_zone['치환숫자']) :
    df_test['판매구역'] = df_test['판매구역'].replace(i,j)




del df_test['ID']
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(df_test)
x_test = scaler.transform(df_test)

# X_train , X_val , Y_train , Y_val = train_test_split(x_train , y_train , test_size=0.3 , random_state=5)



# clf  =RandomForestClassifier(n_estimators=200 , max_depth=15 , random_state=7)
clf  =RandomForestRegressor(n_estimators=200 , max_depth=35 , random_state=5)
clf.fit(x_train, y_train)
relation_square = clf.score(x_train, y_train)
print(relation_square)
#
predict_ = clf.predict(x_test)
#
pre_list = []
#
for i in predict_ :
    pre_list.append(i/100)


dic_ = dict()

result = pd.DataFrame(dic_)
result['ID'] = df_id
result['가격'] = pre_list

result.to_csv('C:/Users/LUNABIT/Desktop/　/open/gogo_1.csv' ,encoding='ms949',header=True, index=False)

