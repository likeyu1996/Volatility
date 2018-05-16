# -*- coding: UTF-8 -*-
# 要用到sklearn或者statsmodels
# https://blog.csdn.net/jiede1/article/details/78245597
# https://blog.csdn.net/lulei1217/article/details/49386295
# https://blog.csdn.net/xuxiatian/article/details/55002412


# 是否需要做两两之间的线性关系？当然要做，单变量回归和多变量回归都要做
# 单变量回归用来检测该波动率的预测能力和包含信息
# 多变量回归用来解释不同波动率的重要性
# scikit-learn,X要求x是DataFrame，y是Series
# reshape(-1,1)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = pd.read_csv(filepath_or_buffer='result_all.csv', index_col=None)
sigma_ivx = np.array(data['sigma_ivx']).reshape(-1,1)
ivx = np.array(data['ivx']).reshape(-1,1)
sigma_gvix = np.array(data['sigma_gvix']).reshape(-1,1)
gvix = np.array(data['gvix']).reshape(-1,1)
sigma_bsm = np.array(data['sigma_bsm']).reshape(-1,1)
sigma_00 = np.array(data['sigma00']).reshape(-1,1)


def one_linear(X,y):
    model = LinearRegression()
    model.fit(X, y)
    print('coef',model.coef_)
    print('intercept_',model.intercept_)
    print('R-squared: %.2f' % model.score(X,y))

    # X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
    # y_test = [[11], [8.5], [15], [18], [11]]
    # predictions = model.predict(X_test)
    # for i, prediction in enumerate(predictions):
    #     print('Predicted: %s, Target: %s' % (prediction, y_test[i]))
    # print('R-squared: %.2f' % model.score(X_test, y_test))


# sigma_ivx_bsm = np.array(data[['sigma_ivx','sigma_bsm']]).reshape(-1,2)
# ivx_bsm = np.array(data[['ivx','sigma_bsm']]).reshape(-1,2)
# sigma_gvix_bsm = np.array(data[['sigma_gvix','sigma_bsm']]).reshape(-1,2)
# gvix_bsm = np.array(data[['gvix','sigma_bsm']]).reshape(-1,2)
# # sigma_bsm = np.array(data['sigma_bsm','']).reshape(-1,1)
#
# one_linear(sigma_bsm,sigma_00)
# one_linear(sigma_ivx,sigma_00)
# one_linear(ivx,sigma_00)
# one_linear(sigma_gvix,sigma_00)
# one_linear(gvix,sigma_00)
# one_linear(sigma_ivx_bsm,sigma_00)
# one_linear(ivx_bsm,sigma_00)
# one_linear(sigma_gvix_bsm,sigma_00)
# one_linear(gvix_bsm,sigma_00)

# ivx_mf = np.array(data[['ivx','sigma_ivx']]).reshape(-1,2)
# sigma_gvix_mf = np.array(data[['sigma_gvix','sigma_ivx']]).reshape(-1,2)
# gvix_mf = np.array(data[['gvix','sigma_ivx']]).reshape(-1,2)
# one_linear(ivx_mf,sigma_00)
# one_linear(sigma_gvix_mf,sigma_00)
# one_linear(gvix_mf,sigma_00)

# sigma_gvix_ivx = np.array(data[['sigma_gvix','ivx']]).reshape(-1,2)
# gvix_mf_ivx = np.array(data[['gvix','ivx']]).reshape(-1,2)
#
# one_linear(sigma_gvix_ivx,sigma_00)
# one_linear(gvix_mf_ivx,sigma_00)

# gvix_gmf = np.array(data[['gvix','sigma_gvix']]).reshape(-1,2)
# one_linear(gvix_gmf,sigma_00)

# gvix_gmf = np.array(data[['sigma_bsm','sigma_ivx','ivx','sigma_gvix','gvix']]).reshape(-1,5)
# one_linear(gvix_gmf,sigma_00)
import arch_series

tteest = pd.read_csv(filepath_or_buffer='hxsz50ETF.csv',index_col=None)
close_array = np.array(tteest['收盘价'])[::-1]
sigma00 = np.array(tteest['sigma00'])[::-1]
r_array = np.log([close_array[i]/close_array[i-1] for i in range(1,len(close_array))])
nlag = 2
sigma_garch = np.array(arch_series.garch(r_array, nlag, 1,vol='GARCH'))[2:]
sigma_egarch = np.array(arch_series.garch(r_array, nlag, 1,o=1,vol='EGARCH'))[2:]
sigma_tarch = np.array(arch_series.garch(r_array, nlag, 1,o=1,power=1.0,vol='GARCH'))[2:]
# one_linear(sigma_garch.reshape(-1,1),sigma_00[1:])
# one_linear(sigma_egarch.reshape(-1,1),sigma_00[1:])
# one_linear(sigma_tarch.reshape(-1,1),sigma_00[1:])

# garch_ivx = np.array(np.c_[sigma_garch,np.array(data['ivx'])[1:]]).reshape(-1,2)
# garch_sigma_gvix = np.array(np.c_[sigma_garch,np.array(data['sigma_gvix'])[1:]]).reshape(-1,2)
# garch_gvix = np.array(np.c_[sigma_garch,np.array(data['gvix'])[1:]]).reshape(-1,2)

# one_linear(garch_ivx,sigma_00[1:])
# one_linear(garch_sigma_gvix,sigma_00[1:])
# one_linear(garch_gvix,sigma_00[1:])

# g_eg =np.array(np.c_[sigma_garch,sigma_egarch]).reshape(-1,2)
# g_t = np.array(np.c_[sigma_garch,sigma_tarch]).reshape(-1,2)
# eg_t =np.array(np.c_[sigma_egarch,sigma_tarch]).reshape(-1,2)
# one_linear(g_eg,sigma_00[1:])
# one_linear(g_t,sigma_00[1:])
# one_linear(eg_t,sigma_00[1:])

g_eg_t=np.array(np.c_[sigma_garch,sigma_egarch,sigma_tarch]).reshape(-1,3)
one_linear(g_eg_t,sigma_00[1:])

sigma_all = np.array(np.c_[np.array(data['ivx'])[1:],np.array(data['sigma_gvix'])[1:],np.array(data['gvix'])[1:],sigma_garch,sigma_egarch,sigma_tarch]).reshape(-1,6)
one_linear(sigma_all,sigma_00[1:])
