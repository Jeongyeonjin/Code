import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import holidays


df =pd.read_csv('C:/Users/LUNABIT/Desktop/open/open/international_trade.csv',dtype='unicode',sep=',')

df_train=pd.read_csv('C:/Users/LUNABIT/Desktop/open/open/train.csv',dtype='unicode',sep=',')
df_test= pd.read_csv('C:/Users/LUNABIT/Desktop/open/open/test.csv',dtype='unicode',sep=',')
df_train = df_train.rename(columns={'timestamp': 'ds', 'price(원/kg)': 'y'})
df_train['ID'] = df_train['ID'].str.replace(r'_\d{8}$', '', regex=True)
df_train['y'] = df_train['y'].astype('float64')

df_train = df_train[['ID','ds' , 'y']]
df= df_train
holidays = holidays.KR()

holiday_df = pd.DataFrame(columns=['ds' , 'holiday'])
holiday_df['ds'] = df['ds']
holiday_df['holiday'] = holiday_df.ds.apply(lambda x : holidays.get(x) if x in holidays else 'non-holiday')


pred_list= []

for code in df['ID'].unique() :
    d = df[df['ID'] == code].reset_index().drop(['ID'], axis=1).sort_values('ds')

    model = Prophet(growth='linear',
                    seasonality_mode='additive',
                    seasonality_prior_scale=8,
                    yearly_seasonality=True,
                    weekly_seasonality='auto',
                    daily_seasonality=True,
                    interval_width=0.80 ,
                    changepoint_prior_scale=0.5
                    # holidays=holiday_df,
                    # holidays_prior_scale= 10
                    )
    model.fit(d)
    pred = pd.DataFrame()
    pred['ds'] = pd.date_range(start='2023-03-04', periods=28, freq='D')
    forecast = model.predict(pred)
    pred_y = forecast['yhat'].values
    pred_code = [str(code)] * len(pred_y)
    for y_val, id_val in zip(pred_y, pred_code):
        pred_list.append({'ID': id_val, 'y': y_val})
    pred = pd.DataFrame(pred_list)



pred['y'] = pred['y'].apply(lambda  x:0 if x < 0 else x)

pred.to_csv('C:/Users/LUNABIT/Desktop/gogo.csv' ,encoding='cp949',header=True, index=False)

