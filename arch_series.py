# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.arima_model import ARMA
import scipy.stats as scs
from arch import arch_model
from arch.univariate import EGARCH
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Garch:通过检验数据的序列相关性建立一个均值方程，如有必要，
# 对收益率序列建立一个计量经济模型（如ARMA，此处用AR）来消除任何线形依赖。
# 对均值方程的残差进行ARCH效应检验
# 如果具有ARCH效应，则建立波动率模型
# 检验拟合的模型，如有必要则进行改进
# https://pypi.org/project/arch/3.0/
# adf单位根检验，检验平稳性，是否需要差分
# adf_res['Lags Used']为使用阶数
# 这里是对收益率序列的处理(data)


def adf_test(data, autolag='AIC'):
    adftest = ts.adfuller(data, autolag='%s' % autolag)
    adf_res = pd.Series(adftest[0:4], index=['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used'])
    adf_res['Autolag'] = autolag
    for key, value in adftest[4].items():
        adf_res['Critical Value (%s)' % key] = value
    fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(111)
    fig = sm.graphics.tsa.plot_pacf(data, lags=20, ax=ax1)
    plt.show()
    return adf_res

# 根据阶数确定均值AR模型并得到残差序列
# data仍为收益率序列
# order=(lags_AR,0), lags_AR=adf_res['Lags Used']


def residual(data, order):
    model = sm.tsa.ARMA(data, order).fit()
    at = data - model.fittedvalues
    at2 = np.square(at)
    plt.figure(figsize=(10,6))
    plt.subplot(211)
    plt.plot(at, label='at')
    plt.legend()
    plt.subplot(212)
    plt.plot(at2, label='at^2')
    plt.legend(loc=0)
    plt.show()
    return at2

# 自相关检验（混成检验），m代表检验多少个自相关系数
# 若具有自相关性（拒绝原假设），则认为具有ARCH效应
# 对残差平方序列（at^2）检验


def lbq_test(m, n):
    r, q, p = sm.tsa.acf(n, nlags=m, qstat=True)
    data = np.c_[range(1, m+1), r[1:], q, p]
    table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
    print(table.set_index('lag'))
    return table

# ARCH模型的阶次用at^2的偏自相关函数确定
# 若使用GARCH(m,s)模型则不必对ARCH(m)定阶，得到均值方程AR(n)即可
# 此处使用Garch（1,1）,n测试集的长度
# 貌似garch模型可以直接确定阶数，不需要提前定阶
# http://nbviewer.jupyter.org/github/bashtage/arch/blob/master/examples/univariate_volatility_modeling.ipynb


def garch(data, lags_ar, n,p=1,o=0,q=1,power=2.0,vol='GARCH'):
    train = data[:-n]
    test = data[-n:]
    am = arch_model(train, mean='AR', lags=lags_ar, vol=vol, dist='StudentsT',p=p,o=o,q=q,power=power)
    res = am.fit()
    print(res.summary())
    print(res.params)
    res.plot()
    plt.plot(data,label = 'origin_data')
    # res.hedgehog_plot()
    ini = res.resid[-lags_ar:]
    a = np.array(res.params[1:lags_ar+1])
    w = a[::-1]  # 系数
    for i in range(n):  # 预测后n个残差at
        new = test[i] - (res.params[0] + w.dot(ini[-lags_ar:]))
        ini = np.append(ini, new)
    # print(len(ini))
    at_pre = ini[-n:]
    at_pre2 = at_pre**2
    # print(at_pre2)
    ini2 = res.conditional_volatility[-2:]  # 上两个条件异方差值
    for i in range(n):
        new = res.params[-3] + res.params[-2]*at_pre2[i] + res.params[-1]*ini2[-1]
        ini2 = np.append(ini2, new)
    vol_pre = ini2[-n:]
    # print(vol_pre)
    plt.figure(figsize=(15,5))
    # plt.plot(data, label='origin_data')
    # plt.plot(res.conditional_volatility,label='conditional_volatility')
    plt.plot(np.sqrt(res.conditional_volatility),label='conditional_sqrt_vol')
    x = range(len(train), len(train)+n)
    #plt.plot(x, vol_pre, '.r', label='predict_volatility')
    plt.legend(loc=0)
    return np.sqrt(res.conditional_volatility)

# TODO
# 得到数据后完成EGARCH
if __name__ == '__main__':
    tteest = pd.read_csv(filepath_or_buffer='hxsz50ETF.csv',index_col=None)
    close_array = np.array(tteest['收盘价'])[::-1]
    sigma00 = np.array(tteest['sigma00'])[::-1]
    r_array = np.log([close_array[i]/close_array[i-1] for i in range(1,len(close_array))])
    nlag = 2
    #
    # at2_array = residual(r_array,order=(nlag,0))
    #
    # print(adf_test(r_array))
    # print(pd.DataFrame(r_array).describe(include='all'))
    # print(pd.DataFrame(sigma00).describe(include='all'))
    # #adf_test(r_array).to_csv('adf_test.csv',encoding='UTF_8')
    # lbq_test(25,at2_array).to_csv('lbq_test.csv',index=None,encoding='UTF_8')
    garch(r_array, nlag, 1,vol='GARCH')
    plt.plot(sigma00,label ='sigma00')
    plt.show()
    garch(r_array, nlag, 1,o=1,vol='EGARCH')
    # data_other = pd.read_csv(filepath_or_buffer='result_all.csv', index_col=None)
    # sigma_ivx = np.array(data_other['sigma_ivx'])
    # ivx = np.array(data_other['ivx'])
    # sigma_gvix = np.array(data_other['sigma_gvix'])
    # gvix = np.array(data_other['gvix'])
    # sigma_bsm = np.array(data_other['sigma_bsm'])
    # print(data_other['sigma_ivx'].describe(include='all'))
    # print(data_other['ivx'].describe(include='all'))
    # print(data_other['sigma_gvix'].describe(include='all'))
    # print(data_other['gvix'].describe(include='all'))
    # print(data_other['sigma_bsm'].describe(include='all'))
    # plt.show()
    # plt.plot(sigma_ivx, label = 'sigma_ivx')
    # plt.plot(ivx/100, label = 'ivx')
    # plt.plot(sigma_gvix, label = 'sigma_gvix')
    # plt.plot(gvix/100, label = 'gvix')
    # plt.plot(sigma_bsm, label = 'sigma_bsm')
    plt.plot(sigma00,label ='sigma00')
    # plt.ylim([0,1])
    plt.show()
    garch(r_array, nlag, 1,o=1,power=1.0,vol='GARCH')
    plt.plot(sigma00,label ='sigma00')
    plt.show()
