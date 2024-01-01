import numpy as np
import pandas as pd
import os
import math
from prophet import Prophet

#-*- coding: utf-8 -*-

df = pd.read_csv('C:/Users/LUNABIT/Desktop/open/train.csv',dtype='unicode',sep=',')

del df['일사합']
del df['일조율']

df['강수량'] = df['강수량'].fillna(0)
df = df.dropna(axis=0)


df = df.rename(columns={'일시' : 'ds' , '평균기온' : 'y'})





model = Prophet(growth='linear',
                    seasonality_mode='additive',
                    seasonality_prior_scale=0.003,
                    yearly_seasonality='auto',
                    weekly_seasonality='auto',
                    daily_seasonality='auto',
                    interval_width=0.80 #신뢰구간
                    ,changepoint_prior_scale=0.698)
## changepoint scale 0.7이상 올리면 오히려 떨어짐 신뢰구간은 0.8이나 0.95나 똑같음
## seasonality_prior_scale 0.003일때 최고
# 더 건들게.... 없다요
# 최근 10년치만 해볼까? changepoint_prior_scale 은 0.7 근방에서는 다 괜춘 확 떨굴려면 ...
# 10년치 한번 해보고 12월만 ?
model.fit(df)
pred = pd.DataFrame()
pred['ds'] = pd.date_range(start='2023-01-01' , periods=358 , freq='D')
forecast = model.predict(pred)
pred['y'] = forecast['yhat'].values

sub = pd.read_csv('C:/Users/LUNABIT/Desktop/open/sample_submission.csv',dtype='unicode',sep=',')
sub['평균기온'] = pred['y'].values
sub.to_csv('C:/Users/LUNABIT/Desktop/gogo_1.csv' ,encoding='UTF-8',header=True, index=False)
