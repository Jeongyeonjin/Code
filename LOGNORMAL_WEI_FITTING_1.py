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
import scipy as sp
from scipy.stats import weibull_min
from scipy import stats
import math
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
mpl.rcParams['axes.unicode_minus']=False

class MLE():

    def __init__(self, samples, m, std, learning_rate, epochs, verbose=False):
        """
        :param samples: samples for get MLE
        :param learning_rate: alpha on weight update
        :param epochs: training epochs
        :param verbose: print status
        """
        self._samples = samples
        self._m = m
        self._std = std
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._verbose = verbose


    def likelihood(self, x, M):
        """
        Probability Density Function is Normal distribution
        PDF's y is same as likelihood

        :param x:
        :return: likelihood of input x (likelihood of input x is same as y of pdf)
        """

        #return  1/(x * self._std * math.sqrt(2*math.pi)) * np.exp( -(np.log(x) - M)**2/(2*self._std **2))
        return M * (x/self._std)**(M-1) * np.exp(-(x/self._std)**M)

    def fit(self):
        """
        training estimator
        M, which minimizes Likelihood, is obtained by the gradient descent method.
        M is the MLE of the samples
        """

        # init M
        self._estimator = np.random.normal(self._m, self._std, 1)

        # train while epochs
        self._training_process = []
        for epoch in range(self._epochs):
            likelihood = np.prod(self.likelihood(self._samples, self._m))
            prediction = np.prod(self.likelihood(self._samples, self._estimator))
            cost = self.cost(likelihood, prediction)
            self._training_process.append((epoch, cost))
            self.update(self._samples, self._estimator)

            # print status
            #if self._verbose == True and ((epoch + 1) % 10 == 0):
            #    print("Iteration: %d ; cost = %.4f" % (epoch + 1, cost))


    def cost(self, likelihood, prediction):
        """
        cost function
        :param likelihood: likelihood of population
        :param prediction: likelihood in samples
        :return: the cost of optimizing the parameters
        """
        return math.sqrt(likelihood - prediction)


    def update(self, x, M):
        """
        update in gradient descent
        gradient is approximated
        :param x: samples
        :param M: estimator
        """
        gradient = np.sum(np.exp(-(np.power(x - M, 2) / (2*math.pow(self._std, 2)))))
        if self._m > self._estimator:
            self._estimator += self._learning_rate * gradient
        else:
            self._estimator -= self._learning_rate * gradient


    def get_mle(self):
        """
        parameter getter
        :return: estimator of MLE
        """
        return self._estimator


list=['20161_평당가격', '20162_평당가격', '20163_평당가격', '20164_평당가격', '20165_평당가격', '20166_평당가격', '20167_평당가격', '20168_평당가격', '20169_평당가격', '201610_평당가격', '201611_평당가격', '201612_평당가격', '20171_평당가격', '20172_평당가격', '20173_평당가격', '20174_평당가격', '20175_평당가격', '20176_평당가격', '20177_평당가격', '20178_평당가격', '20179_평당가격', '201710_평당가격', '201711_평당가격', '201712_평당가격', '20181_평당가격', '20182_평당가격', '20183_평당가격', '20184_평당가격', '20185_평당가격', '20186_평당가격', '20187_평당가격', '20188_평당가격', '20189_평당가격', '201810_평당가격', '201811_평당가격','201812_평당가격']

list1=['20161_평당가격', '20162_평당가격', '20163_평당가격', '20164_평당가격', '20165_평당가격', '20166_평당가격', '20167_평당가격', '20168_평당가격', '20169_평당가격', '201610_평당가격', '201611_평당가격', '201612_평당가격', '20171_평당가격', '20172_평당가격', '20173_평당가격', '20174_평당가격', '20175_평당가격', '20176_평당가격', '20177_평당가격', '20178_평당가격', '20179_평당가격', '201710_평당가격', '201711_평당가격', '201712_평당가격', '20181_평당가격', '20182_평당가격', '20183_평당가격', '20184_평당가격', '20185_평당가격', '20186_평당가격', '20187_평당가격', '20188_평당가격', '20189_평당가격', '201810_평당가격', '201811_평당가격']

list2=['20162_평당가격', '20163_평당가격', '20164_평당가격', '20165_평당가격', '20166_평당가격', '20167_평당가격', '20168_평당가격', '20169_평당가격', '201610_평당가격', '201611_평당가격', '201612_평당가격', '20171_평당가격', '20172_평당가격', '20173_평당가격', '20174_평당가격', '20175_평당가격', '20176_평당가격', '20177_평당가격', '20178_평당가격', '20179_평당가격', '201710_평당가격', '201711_평당가격', '201712_평당가격', '20181_평당가격', '20182_평당가격', '20183_평당가격', '20184_평당가격', '20185_평당가격', '20186_평당가격', '20187_평당가격', '20188_평당가격', '20189_평당가격', '201810_평당가격', '201811_평당가격', '201812_평당가격']

index1=[	'201601'	,'201602'	,'201603'	,'201604'	,'201605'	,'201606'	,'201607'	,'201608'	,'201609'	,'201610'	,'201611'	,'201612'	,'201701'	,'201702'	,'201703'	,'201704'	,'201705'	,'201706'	,'201707'	,'201708'	,'201709'	,'201710',	'201711'	,'201712'	,'201801'	,'201802'	,'201803'	,'201804'	,'201805'	,'201806'	,'201807'	,'201808'	,'201809'	,'201810'	,'201811'	,'201812']

#list=['20161_평당가격']
#index1=[	'201601']
df=pd.read_csv('C:/Users/User/Desktop/data2016-2018.csv',dtype='unicode',sep=',',encoding='ms949')

def mse(y, t):
    return  ((y-t)**2).mean(axis=None)



for ab,zd in zip(list,index1) :
    d=[]
    e=[]
    for id in df[ab]:
        d.append(float(id))
    for cd in df[ab]:
        if type(cd) == str :
            e.append(float(cd))


    data = sorted(e)

    N_bins = 45
    lenth = len(data)

    for a in range (90 , 95 , 5):

        pl_for_data= []
        aa= lenth * a * 0.01
        for dd in data :
            if dd < data[int(aa) -1] :
                pl_for_data.append(dd)
        d=np.array(pl_for_data)



        shape, loc , scale = sp.stats.lognorm.fit(d, floc = 0)
        shape_w , loc_w, scale_w = sp.stats.weibull_min.fit(d ,floc = 0)






        weights = np.ones_like(d)/float(len(d))
        x_for_sample_result = np.linspace(0,d.max(),N_bins)
        y_l=(x_for_sample_result - loc)/scale
        y=(x_for_sample_result - loc_w)/scale_w
        counts , bin_edges  = np.histogram(d , bins = N_bins  ,normed= True )


        counts_N , bin_edges_N  = np.histogram(d , bins = N_bins  )

        samples_result = sp.stats.lognorm.pdf(x_for_sample_result, shape , loc = loc , scale= scale)
        samples_result_wei = sp.stats.weibull_min.pdf( y,shape_w) /scale_w


        mx=counts_N.max()
        #print(mx)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        #test_stat=stats.kstest(counts,'lognorm',sp.stats.lognorm.fit(d, floc = 0))
        #print(test_stat[1])
        #MSE =mse(counts , samples_result)
        #print('MSE = %.10f , 1/MSE = %f' %(MSE,1/MSE))




        plt.plot(bin_edges[:-1], counts,color='b',marker='o',linestyle='None',markeredgecolor='b')

        plt.plot( x_for_sample_result, samples_result ,linewidth=4,color='r',label='lognormal')
        #plt.plot( x_for_sample_result, samples_result_wei ,linewidth=4,color='k',label='Weibull')
        ax.set_xlabel('$p$' , size = 20)
        ax.set_ylabel('$f$', size = 20)
        ax.tick_params(axis='both', direction='in')
        ax.set_xlim([0,1500])
        ax.set_ylim([0,0.004])
        plt.title(zd , size =28)
        plt.legend()
        #plt.show()
        plt.savefig('C:/Users/User/Desktop/Fitting/%s body lognorm pdf fitting' %zd)

        estimator = MLE(d, scale, shape, learning_rate=0.1, epochs=30, verbose=True)
        estimator.fit()
        result = estimator.get_mle()

        estimator_w = MLE(d, scale_w, shape_w, learning_rate=0.1, epochs=30, verbose=True)
        estimator_w.fit()
        result_w = estimator_w.get_mle()
        #print(result_w[0])
