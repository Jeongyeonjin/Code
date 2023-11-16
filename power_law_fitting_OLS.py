#-*- coding: utf-8 -*-

from mpl_toolkits.mplot3d import axes3d
import numpy as np
import sys
import io
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc , style
import pandas as pd
import plotly
import matplotlib as mpl
import scipy.stats
from scipy.optimize import curve_fit
import statsmodels.api as sm

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
mpl.rcParams['axes.unicode_minus']=False


list=['20161_평당가격', '20162_평당가격', '20163_평당가격', '20164_평당가격', '20165_평당가격', '20166_평당가격', '20167_평당가격', '20168_평당가격', '20169_평당가격', '201610_평당가격', '201611_평당가격', '201612_평당가격', '20171_평당가격', '20172_평당가격', '20173_평당가격', '20174_평당가격', '20175_평당가격', '20176_평당가격', '20177_평당가격', '20178_평당가격', '20179_평당가격', '201710_평당가격', '201711_평당가격', '201712_평당가격', '20181_평당가격', '20182_평당가격', '20183_평당가격', '20184_평당가격', '20185_평당가격', '20186_평당가격', '20187_평당가격', '20188_평당가격', '20189_평당가격', '201810_평당가격', '201811_평당가격','201812_평당가격']

index1=[	'201601'	,'201602'	,'201603'	,'201604'	,'201605'	,'201606'	,'201607'	,'201608'	,'201609'	,'201610'	,'201611'	,'201612'	,'201701'	,'201702'	,'201703'	,'201704'	,'201705'	,'201706'	,'201707'	,'201708'	,'201709'	,'201710',	'201711'	,'201712'	,'201801'	,'201802'	,'201803'	,'201804'	,'201805'	,'201806'	,'201807'	,'201808'	,'201809'	,'201810'	,'201811'	,'201812']


def mode(e,N_bins) :
    ns,a  = np.histogram(e,N_bins )
    mode = np.argmax(ns)
    return a[mode + 2]

def func_powerlaw(x, a ,c  ):
    return x**(-a) * c

def mse(y, t):
    return  ((y-t)**2).mean(axis=None)

def gini(sorted_list):

    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(sorted_list) / 2.
    return (fair_area - area) / fair_area
alpha = []

df=pd.read_csv('C:/Users/User/Desktop/data2016-2018.csv',dtype='unicode',sep=',',encoding='ms949')

gini_coefficient = []
result = {}




for ab , zd in zip(list,index1) :
    dada = {}
    d=[]
    e=[]
    for id in df[ab]:
        d.append(float(id))
    for cd in df[ab]:
        if type(cd) == str :
            e.append(float(cd))
    df[ab] = d

    data = sorted(e)

    da=np.array(e)
    abc=df[ab].describe()
    gini_coefficient.append(gini(data))

    N_bins = 60
    #pl_for_data = df.loc[df[ab]>abc[6]][ab]
    lenth = len(data)
    alpha = []
    MSE = []

    for a in range (90 , 95 , 5):

        pl_for_data= []
        aa= lenth * a * 0.01
        for dd in data :
            if dd > data[int(aa) -1] :
                pl_for_data.append(dd)

        counts , bin_edges  = np.histogram(pl_for_data,N_bins , normed =True )



        popt, pcov = curve_fit(func_powerlaw , bin_edges[:-1] , counts , maxfev = 50000 )
        alpha.append(popt[0])

        #MSE.append(mse(counts , func_powerlaw(bin_edges[:-1] , *popt )))
        #print(alpha[0])


    #result[zd] = [gini(data) , 1/MSE[0] ,1/MSE[1] ,1/MSE[2] ,1/MSE[3] , alpha[0] ,alpha[1] ,alpha[2],alpha[3]]

#df_for_index = ['gini_coefficient' , '80% 1/ MSE' , '85% 1/ MSE' ,'90% 1/MSE' , '95% 1/MSE' ,'80% alpha' , '85% alpha' ,'90% alpha' , '95% alpha']
#dataframe = pd.DataFrame(result) #, index = df_for_index )
#dataframe.to_csv('C:/Users/User/Desktop/gini_MSE_alpha_1.csv',encoding='ms949',header=True, index=True)

#for aa , bb in zip(range(0,10) , df_for_index) :
#    fig = plt.figure()
#    ax = fig.add_subplot(1,1,1)
#    ax.set_xlabel('$Time$' , size = 15)
#    ax.set_ylabel('$f$', size = 15)
#    #ax.set_ylim([2,6.5])
#    ax.tick_params(axis='both', direction='in')
#    plt.plot(dataframe.iloc[aa] , color='b',marker='o',markerfacecolor='None',markeredgecolor='b')
#    plt.title(bb , size =20)
    #plt.show()
    #plt.savefig('C:/Users/User/Desktop/솔개/ %s ' %aa)
#print(dataframe)
