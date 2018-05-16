# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import time, datetime, calendar
import bsm
import vix
import get_file_name


def ivx_gvix():
    date_array = []
    sigma_ivx = []
    ivx_array = []
    sigma_gvix = []
    gvix_array = []
    for date in get_file_name.get_filename(path=r'F:\Workspace\python\volatility\date_arr',n=12):
        print(date)
        db_ivx_temp = pd.read_csv(filepath_or_buffer='date_arr/%s' % date,index_col=None)
        if date[0:4] == 2015:
            r0 = 0.030872
        elif date[0:4] == 2016:
            r0 = 0.022296
        elif date[0:4] == 2017:
            r0 = 0.031164
        else:
            r0 = 0.030061
        sm = vix.IVX(db_ivx_temp, r0)
        date_array.append(db_ivx_temp['日期'][0])
        sigma_ivx.append(np.sqrt(sm.outer_sigma2()[0]))
        ivx_array.append(sm.ivx())
        sigma_gvix.append(np.sqrt(sm.outer_sigma2_gvix()[0]))
        gvix_array.append(sm.gvix())
    data = np.c_[date_array,sigma_ivx,ivx_array,sigma_gvix,gvix_array]
    column = ['date','sigma_ivx','ivx','sigma_gvix','gvix']
    frame = pd.DataFrame(data,columns=column,)
    frame.to_csv(path_or_buf='result.csv',index=False,encoding='UTF-8')


def bsm_sigma():
    db_hxsz = pd.read_csv(filepath_or_buffer='hxsz50ETF.csv',index_col=None)
    date_array = []
    code_id_array = []
    option_type_array = []
    K_array = []
    T_array = []
    sigma_bsm = []
    for date in get_file_name.get_filename(path=r'F:\Workspace\python\volatility\date',n=12):
        print(date)
        # 两个地方的路径都要改，注意一下
        db_bsm_temp = pd.read_csv(filepath_or_buffer='date/%s' % date,index_col=None)
        if db_bsm_temp['日期'][0] not in list(db_hxsz['日期']):
            print('%s is not in ETF database' % db_bsm_temp['日期'][0])
            continue
        if date[0:4] == 2015:
            r0 = 0.030872
        elif date[0:4] == 2016:
            r0 = 0.022296
        elif date[0:4] == 2017:
            r0 = 0.031164
        else:
            r0 = 0.030061
        # 选取持仓量最大的期权
        interest_array = [int(i.replace(',','')) for i in db_bsm_temp['持仓量']]
        interest_arr = pd.DataFrame(interest_array,columns=['interest_arr'])
        db_bsm_temp = db_bsm_temp.merge(interest_arr,left_index=True,right_index=True,how='outer')
        max_interest = np.max(interest_array)
        db_max_interest = db_bsm_temp[(db_bsm_temp['interest_arr'] == max_interest)]
        index_max = db_max_interest.index.tolist()
        K = db_bsm_temp['K'][index_max[0]]
        c0 = db_bsm_temp['收盘价'][index_max[0]]
        code_id = db_bsm_temp['期权代码'][index_max[0]]
        option_type = db_bsm_temp['option_type'][index_max[0]]
        date_max = time.strptime(db_bsm_temp['日期'][index_max[0]], "%Y-%m-%d")
        maturity_max = time.strptime(db_bsm_temp['T'][index_max[0]], "%Y-%m-%d")
        date_max = datetime.datetime(date_max[0],date_max[1],date_max[2])
        maturity_max = datetime.datetime(maturity_max[0],maturity_max[1],maturity_max[2])
        if calendar.isleap(date_max.year):
            remains_inyear = (maturity_max -date_max).days/366.0
        else:
            remains_inyear = (maturity_max -date_max).days/365.0
        db_s = db_hxsz[(db_hxsz['日期'] == db_bsm_temp['日期'][0])]
        index_s = db_s.index.tolist()
        s = db_s['收盘价'][index_s[0]]
        sigma00 = db_s['sigma00'][index_s[0]]
        bsm_sigma = bsm.BSM(S=s,K=K,T=remains_inyear,r=r0,sigma=sigma00,option_type=option_type).imp_sigma(c0=c0,sigma_est=sigma00)
        date_array.append(db_bsm_temp['日期'][0])
        code_id_array.append(code_id)
        option_type_array.append(option_type)
        K_array.append(K)
        T_array.append(db_bsm_temp['T'][index_max[0]])
        sigma_bsm.append(bsm_sigma)
    data = np.c_[date_array,code_id_array,option_type_array,K_array,T_array,sigma_bsm]
    frame = pd.DataFrame(data,columns=['date','code_id','option_type','K','T','sigma_bsm'])
    frame.to_csv(path_or_buf='result_bsm.csv',index=False,encoding='UTF-8')
ivx_gvix()
bsm_sigma()
