# -*- coding: UTF-8 -*-

# TODO
# Data
# 数据获取方式（读取csv或爬虫，后期考虑换为sql）
# 获取内容（上证50ETF[华夏基金]价格和交易量序列，上证50ETF期权价格和交易量，同时期国债利率）
# 数据清洗（对期权，保留主力期权（交易量最大的期权），平价期权）
# 收益率计算（对数化价格序列，得到收益率序列，特别考虑期权跨期问题）
# 数据截取（根据预设的窗口期，截取相应长度的序列，似乎也要注意期权跨期问题）
# 数据方面（main）最终得到窗口长度的收益率序列

# TODO
# 波动率计算与估计
# 自相关性检验，Ljung-Box检验，ARCH效应检验
# GARCH,EGARCH,GARCH-IV,EGARCH-IV(输入收益率序列，输出波动率)(单窗口）
# 用循环来估计所有窗口的波动率，得到波动率序列
# 已实现波动率、历史波动率、BS隐含波动率、无模型波动率
# 总共得到8种波动率序列

# TODO
# 统计分析
# 样本数量、均值、标准差、偏度、峰度、最小值、最大值
# 同期不同波动率之间的相关系数
# 将总时长分段（自定义时段下）不同波动率之间的相关系数

# TODO
# 包含回归
# 日期	期权代码	交易代码	期权简称	涨跌幅(%)	开盘价	收盘价	成交量	持仓量
# date,option_code,trading_code,code,type,T,status,K,name,range,open,close,volume,open_Interest
# date	option_code	type	T	K	range	open	close	volume	open_Interest

# https://blog.csdn.net/jt1123/article/details/50086595 dataframe数据筛选
import numpy as np
import pandas as pd
import datetime,calendar
import time
import get_file_name
import bsm
from scipy import interpolate

# 日期列表与期权代码列表，分别对应无模型和BS两种波动率的计算
# 用条件筛选的方法取出对无模型和BS所需数据，甚至单个数据
# print(db[(db['日期']=='2015-06-24') & (db['期权代码']==10000027)])
# 算无模型要逐日计算，先获取日期set，按先后排序，每个日期作为索引获得一份当日dataframe
# BS要跟踪主力期权，因此需要先通过交易量判断主力期权（或者使用黄薏舟等）的办法，再通过X方案计算持续的BS波动率


def trans_date(str_date):
    date = np.array([datetime.date(int(i[0:4]), int(i[5:7]), int(i[8:10])).strftime("%Y-%m-%d") for i in str_date])
    return pd.DataFrame(date,columns=['日期'])


def trading_code_slice(trading_code):
    asset_id = []
    option_type = []
    T_year = []
    T_month = []
    T_maturity = []
    status = []
    K = []
    # 分离
    for i in trading_code:
        asset_id.append(i[0:6])
        option_type.append(i[6:7])
        T_year.append(2000+int(i[7:9]))# 考虑过eval
        T_month.append(int(i[9:11]))
        status.append(i[11:12])
        # 考虑到调整后的期权执行价不会反映在交易代码中，故此处的K不纳入计算
        K.append(float(i[12:]))
    K = [i/1000.0 for i in K]

    def cal_T(year,month):
        _weekday,_lastday = calendar.monthrange(year,month)
        _wednesday = [i for i in range(2-_weekday+1,_lastday+1,7)]
        # standard_day = datetime.date.today()
        # oneday = datetime.timedelta(days = 1)
        # d = '%04d/%02d/%02d' % (year, month, _wednesday[3])
        # pd.Timestamp('20171121')也可以表示时间
        # 注意，这样筛选出来的是第四个周的星期三，而不一定是第四个星期三，故还要增加条件进行判定
        # 22日之前的周三只有1,8,15,22,因此小于22的周三必定不是第四个周三
        if _wednesday[3] < 22:
            d = datetime.date(year, month, _wednesday[4])
        else:
            d = datetime.date(year, month, _wednesday[3])
        return d.strftime("%Y-%m-%d")
    if len(T_year) == len(T_month):
        for i in range(len(T_year)):
            T_maturity.append(cal_T(T_year[i],T_month[i]))
    else:
        raise ValueError("year is not equal to month")
    result = pd.DataFrame(np.array([option_type,T_maturity,K,status]).T,columns=["option_type", "T", "K_ori","status"])
    return result


def k_real(name):
    # 50ETF购6月2.651A
    # 50ETF购6月2.80
    # 50ETF沽2017年6月2.299A
    K = []
    for i in name:
        for j in range(len(i)):
            if i[j] == '月':
                if i[-1] == 'A' or i[-1] == 'B':
                    # if i[-1] == 'A' or 'B': 写成这样就完了，因为'B'一直是True
                    K.append(float(i[j+1:-1]))
                else:
                    K.append(float(i[j+1:]))
    result = pd.DataFrame(np.array(K),columns=['K'])
    print(result)
    return result


def new_vix_db(db):
    date_list = list(set(db['日期']))
    date_list.sort()
    code_list = list(set(db['期权代码']))
    code_list.sort()
    for i in date_list:
        db_date = db[(db['日期'] == i)]
        filename = i[0:4] + i[5:7] + i[8:10]
        pd.DataFrame.to_csv(db_date, path_or_buf="date/%s.csv" % filename,index=False,encoding='UTF-8')
    for i in code_list:
        db_code = db[(db['期权代码'] == i)]
        pd.DataFrame.to_csv(db_code, path_or_buf="code/%s.csv" % i,index=False,encoding='UTF-8')


def spline_vix_db():
    db_hxsz = pd.read_csv(filepath_or_buffer='hxsz50ETF.csv',index_col=None)
    # 20161129是上证50ETF的首个除息日，期权合约调整，市场上出现了同执行价同类型同到期日的期权，导致interp1d报X错，splrep出现nan
    # 解决方法为，由于之前的标准期权合约变成了非标合约，流动性会变差，故直接不考虑这类期权
    # 由于计算标的资产华夏上证时使用的是复权单位基金，故暂不考虑分红带来的影响，调整后的合约也依旧使用调整前的执行价
    # 因此当市场上出现同执行价同类型同到期日的期权时，判断交易代码中的A与M
    datesave = []
    for date in get_file_name.get_filename(path=r'F:\Workspace\python\volatility\date', n=12):
        print(date)
        db_ivx_temp = pd.read_csv(filepath_or_buffer='date/%s' % date,index_col=None)
        if db_ivx_temp['日期'][0] not in list(db_hxsz['日期']):
            print('%s is not in ETF database' % db_ivx_temp['日期'][0])
            continue
        T_list = list(set(db_ivx_temp['T']))
        T_list.sort()
        K_list = list(set(db_ivx_temp['K']))
        K_list.sort()
        T_0 = T_list[0]
        T_1 = T_list[1]
        date0 = time.strptime(T_0, "%Y-%m-%d")
        date1 = time.strptime(T_1, "%Y-%m-%d")
        date_ori = time.strptime(db_ivx_temp['日期'][0], "%Y-%m-%d")
        date0 = datetime.datetime(date0[0], date0[1], date0[2])
        date1 = datetime.datetime(date1[0], date1[1], date1[2])
        date_ori = datetime.datetime(date_ori[0], date_ori[1], date_ori[2])
        remain0 = date0-date_ori
        remain1 = date1-date_ori
        # 针对市场上交易已过期期权（如5.21时仍有5.20到期的期权在交易）的情况，直接不考虑这一类期权
        if remain0.days <= 0:
            T_0 = T_list[1]
            T_1 = T_list[2]
            date0 = time.strptime(T_0, "%Y-%m-%d")
            date1 = time.strptime(T_1, "%Y-%m-%d")
            date_ori = time.strptime(db_ivx_temp['日期'][0], "%Y-%m-%d")
            date0 = datetime.datetime(date0[0], date0[1], date0[2])
            date1 = datetime.datetime(date1[0], date1[1], date1[2])
            date_ori = datetime.datetime(date_ori[0], date_ori[1], date_ori[2])
            remain0 = date0-date_ori
            remain1 = date1-date_ori
        if date_ori.year == 2015:
            r0 = 0.030872
        elif date_ori.year == 2016:
            r0 = 0.022296
        elif date_ori.year == 2017:
            r0 = 0.031164
        else:
            r0 = 0.030061
        # print(remain0,remain1)# 类型为datetime.timedelta
        # 获取剩余天数在一年中的比例(T)
        remain0 = remain0.days/365.0
        remain1 = remain1.days/365.0
        # 分别建立T_0,T_1到期的期权对K的映射，要求满射
        # S, K, T, r, sigma, option_type
        # 对近月和次近月分别进行数据筛选，得到两组对应的数组，并对其三次样条插值拟合函数
        # 不分离看涨和看跌的话，会因为执行价（x）中有相同的值而导致插值失败
        # 在dataframe中排序可以保留对应关系
        db_temp_0 = db_ivx_temp[(db_ivx_temp['T'] == T_0)]
        db_temp_0 = db_temp_0.sort_values(by='K')
        db_temp_0_c = db_temp_0[(db_temp_0['option_type'] == 'C')]
        db_temp_0_p = db_temp_0[(db_temp_0['option_type'] == 'P')]
        index_0_c = db_temp_0_c.index.tolist()
        index_0_p = db_temp_0_p.index.tolist()
        db_temp_1 = db_ivx_temp[(db_ivx_temp['T'] == T_1)]
        db_temp_1 = db_temp_1.sort_values(by='K')
        db_temp_1_c = db_temp_1[(db_temp_1['option_type'] == 'C')]
        db_temp_1_p = db_temp_1[(db_temp_1['option_type'] == 'P')]
        index_1_c = db_temp_1_c.index.tolist()
        index_1_p = db_temp_1_p.index.tolist()
        k_0_c = np.array([db_temp_0_c['K'][i] for i in index_0_c])
        k_1_c = np.array([db_temp_1_c['K'][i] for i in index_1_c])
        k_0_p = np.array([db_temp_0_p['K'][i] for i in index_0_p])
        k_1_p = np.array([db_temp_1_p['K'][i] for i in index_1_p])
        # 针对16.11.29除息日的情况
        count_0_c = np.unique(k_0_c,return_counts=True)
        count_1_c = np.unique(k_1_c,return_counts=True)
        count_0_p = np.unique(k_0_p,return_counts=True)
        count_1_p = np.unique(k_1_p,return_counts=True)
        # 取出重复元素,这里只适用于最多两种合约，若同时出现同类型同到期日同执行价的MAB三种期权则此结构无效
        diff_k_0_c = [count_0_c[0][i] for i in range(len(count_0_c[0])) if count_0_c[1][i] != 1]
        diff_k_1_c = [count_1_c[0][i] for i in range(len(count_1_c[0])) if count_1_c[1][i] != 1]
        diff_k_0_p = [count_0_p[0][i] for i in range(len(count_0_p[0])) if count_0_p[1][i] != 1]
        diff_k_1_p = [count_1_p[0][i] for i in range(len(count_1_p[0])) if count_1_p[1][i] != 1]
        # print(diff_k_0_c,diff_k_1_c,diff_k_0_p,diff_k_1_p)

        if len(diff_k_0_c) or len(diff_k_1_c) or len(diff_k_0_p) or len(diff_k_1_p):
            print('%s is in MAB' % date)
            datesave.append(date[:-4])
            drop_temp_0_c = []
            drop_temp_1_c = []
            drop_temp_0_p = []
            drop_temp_1_p = []
            for i in diff_k_0_c:
                drop_temp_0_c += db_temp_0_c[(db_temp_0_c['K'] == i) & (db_temp_0_c['status'] == 'M')].index.tolist()
            for i in diff_k_1_c:
                drop_temp_1_c += db_temp_1_c[(db_temp_1_c['K'] == i) & (db_temp_1_c['status'] == 'M')].index.tolist()
            for i in diff_k_0_p:
                drop_temp_0_p += db_temp_0_p[(db_temp_0_p['K'] == i) & (db_temp_0_p['status'] == 'M')].index.tolist()
            for i in diff_k_1_p:
                drop_temp_1_p += db_temp_1_p[(db_temp_1_p['K'] == i) & (db_temp_1_p['status'] == 'M')].index.tolist()
            db_temp_0_c.drop(drop_temp_0_c, inplace=True)
            db_temp_1_c.drop(drop_temp_1_c, inplace=True)
            db_temp_0_p.drop(drop_temp_0_p, inplace=True)
            db_temp_1_p.drop(drop_temp_1_p, inplace=True)
            # 更新数据 db_temp和K_list
            # db_temp_0_c = db_temp_0[(db_temp_0['option_type'] == 'C')]
            # db_temp_0_p = db_temp_0[(db_temp_0['option_type'] == 'P')]
            # db_temp_1_c = db_temp_1[(db_temp_1['option_type'] == 'C')]
            # db_temp_1_p = db_temp_1[(db_temp_1['option_type'] == 'P')]
            index_0_c = db_temp_0_c.index.tolist()
            index_0_p = db_temp_0_p.index.tolist()
            index_1_c = db_temp_1_c.index.tolist()
            index_1_p = db_temp_1_p.index.tolist()
            k_0_c = np.array([db_temp_0_c['K'][i] for i in index_0_c])
            k_1_c = np.array([db_temp_1_c['K'][i] for i in index_1_c])
            k_0_p = np.array([db_temp_0_p['K'][i] for i in index_0_p])
            k_1_p = np.array([db_temp_1_p['K'][i] for i in index_1_p])
            # 对于ndarray，+代表元素相加而非拼接，拼接用r_与c_
            # 不用set去重，也不用sort排序，直接unique搞定
            K_list = np.unique(np.r_[k_0_c, k_1_c, k_0_p, k_1_p])
        db_s = db_hxsz[(db_hxsz['日期'] == db_ivx_temp['日期'][0])]
        index_s = db_s.index.tolist()
        s = db_s['收盘价'][index_s[0]]
        sigma00 = db_s['sigma00'][index_s[0]]
        # 本打算计算sigma时使用的是累计复权净值的30日样本标准差，后来仍然使用收盘价30日样本标准差
        # 注意 db的对应 被坑过
        c_0 = np.array([db_temp_0_c['收盘价'][i] for i in index_0_c])  # 期权价格
        c_1 = np.array([db_temp_1_c['收盘价'][i] for i in index_1_c])
        p_0 = np.array([db_temp_0_p['收盘价'][i] for i in index_0_p])  # 期权价格
        p_1 = np.array([db_temp_1_p['收盘价'][i] for i in index_1_p])
        # option_type_0 = np.array([db_temp_0['option_type'][i] for i in index_0])
        # option_type_1 = np.array([db_temp_1['option_type'][i] for i in index_1])
        sigma_0_c = np.array([bsm.BSM(s, k_0_c[i], remain0, r=r0, sigma=sigma00, option_type='C').imp_sigma(c0=c_0[i], sigma_est=sigma00) for i in range(len(index_0_c))])
        sigma_1_c = np.array([bsm.BSM(s, k_1_c[i], remain1, r=r0, sigma=sigma00, option_type='C').imp_sigma(c0=c_1[i], sigma_est=sigma00) for i in range(len(index_1_c))])
        sigma_0_p = np.array([bsm.BSM(s, k_0_p[i], remain0, r=r0, sigma=sigma00, option_type='P').imp_sigma(c0=p_0[i], sigma_est=sigma00) for i in range(len(index_0_p))])
        sigma_1_p = np.array([bsm.BSM(s, k_1_p[i], remain1, r=r0, sigma=sigma00, option_type='P').imp_sigma(c0=p_1[i], sigma_est=sigma00) for i in range(len(index_1_p))])
        # 不明白interp1d和splrep的区别
        # func_0_c = interpolate.splrep(x=k_0_c, y=sigma_0_c, k=3)
        # func_1_c = interpolate.splrep(x=k_1_c, y=sigma_1_c, k=3)
        # func_0_p = interpolate.splrep(x=k_0_p, y=sigma_0_p, k=3)
        # func_1_p = interpolate.splrep(x=k_1_p, y=sigma_1_p, k=3)
        # sigma_dummy_0_c = interpolate.splev(K_list, func_0_c, der=0)
        # sigma_dummy_1_c = interpolate.splev(K_list, func_1_c, der=0)
        # sigma_dummy_0_p = interpolate.splev(K_list, func_0_p, der=0)
        # sigma_dummy_1_p = interpolate.splev(K_list, func_1_p, der=0)
        func_0_c = interpolate.interp1d(x=k_0_c, y=sigma_0_c, kind='cubic')
        func_1_c = interpolate.interp1d(x=k_1_c, y=sigma_1_c, kind='cubic')
        func_0_p = interpolate.interp1d(x=k_0_p, y=sigma_0_p, kind='cubic')
        func_1_p = interpolate.interp1d(x=k_1_p, y=sigma_1_p, kind='cubic')
        # 断点之外的波动率要用截断点处的波动率估计，否则，会出现A value in x_new is below the interpolation的错误
        # 对于interp1d来说，似乎splrep处理了这类问题，但是出错不报就很蠢
        # 取出区间内的K，用来解sigma_dummy，再统计两端k的数量，用端点处的sigma_dummy代替
        K_list_0_c = [i for i in K_list if k_0_c[0] <= i <= k_0_c[-1]]
        K_list_1_c = [i for i in K_list if k_1_c[0] <= i <= k_1_c[-1]]
        K_list_0_p = [i for i in K_list if k_0_p[0] <= i <= k_0_p[-1]]
        K_list_1_p = [i for i in K_list if k_1_p[0] <= i <= k_1_p[-1]]
        sigma_dummy_0_c = func_0_c(K_list_0_c)
        sigma_dummy_1_c = func_1_c(K_list_1_c)
        sigma_dummy_0_p = func_0_p(K_list_0_p)
        sigma_dummy_1_p = func_1_p(K_list_1_p)
        # 思考一次生成三个列表的生成式
        left_0_c = len([i for i in K_list if i < k_0_c[0]])
        right_0_c = len([i for i in K_list if i > k_0_c[-1]])
        left_1_c = len([i for i in K_list if i < k_1_c[0]])
        right_1_c = len([i for i in K_list if i > k_1_c[-1]])
        left_0_p = len([i for i in K_list if i < k_0_p[0]])
        right_0_p = len([i for i in K_list if i > k_0_p[-1]])
        left_1_p = len([i for i in K_list if i < k_1_p[0]])
        right_1_p = len([i for i in K_list if i > k_1_p[-1]])
        sigma_dummy_0_c = np.r_[[sigma_dummy_0_c[0] for i in range(left_0_c)], sigma_dummy_0_c, [sigma_dummy_0_c[-1] for j in range(right_0_c)]]
        sigma_dummy_1_c = np.r_[[sigma_dummy_1_c[0] for i in range(left_1_c)], sigma_dummy_1_c, [sigma_dummy_1_c[-1] for j in range(right_1_c)]]
        sigma_dummy_0_p = np.r_[[sigma_dummy_0_p[0] for i in range(left_0_p)], sigma_dummy_0_p, [sigma_dummy_0_p[-1] for j in range(right_0_p)]]
        sigma_dummy_1_p = np.r_[[sigma_dummy_1_p[0] for i in range(left_1_p)], sigma_dummy_1_p, [sigma_dummy_1_p[-1] for j in range(right_1_p)]]
        # 迭代，缺失的sigma用插值得到
        # 再通过BS公式得到完整的合约
        c_dummy_0 = np.array([bsm.BSM(s, K_list[i], remain0, r=r0, sigma=sigma_dummy_0_c[i], option_type='C').call_value() for i in range(len(K_list))])
        c_dummy_1 = np.array([bsm.BSM(s, K_list[i], remain1, r=r0, sigma=sigma_dummy_1_c[i], option_type='C').call_value() for i in range(len(K_list))])
        p_dummy_0 = np.array([bsm.BSM(s, K_list[i], remain0, r=r0, sigma=sigma_dummy_0_p[i], option_type='P').put_value() for i in range(len(K_list))])
        p_dummy_1 = np.array([bsm.BSM(s, K_list[i], remain1, r=r0, sigma=sigma_dummy_1_p[i], option_type='P').put_value() for i in range(len(K_list))])
        T_0_list = [T_0 for i in range(len(K_list))]
        T_1_list = [T_1 for i in range(len(K_list))]
        c_option_type_list = ['C' for i in range(len(K_list))]
        p_option_type_list = ['P' for i in range(len(K_list))]
        date_list = [db_ivx_temp['日期'][0] for i in range(len(K_list))]
        # 这里采用CP(0)CP(1)的顺序
        date_list_arr = np.r_[date_list, date_list, date_list, date_list]
        T_list_arr = np.r_[T_0_list, T_0_list, T_1_list, T_1_list]
        option_type_list = np.r_[c_option_type_list, p_option_type_list, c_option_type_list, p_option_type_list]
        close_list = np.r_[c_dummy_0, p_dummy_0, c_dummy_1, p_dummy_1]
        K_list_arr = np.r_[K_list, K_list, K_list, K_list]
        # 这一行暂且想不清了
        # status_arr = np.r_[list(db_temp_0_c['status']),list(db_temp_0_p['status']),list(db_temp_1_c['status']),list(db_temp_1_p['status'])]
        sigma_list_arr = np.r_[sigma_dummy_0_c, sigma_dummy_0_p, sigma_dummy_1_c, sigma_dummy_1_p]
        data = np.c_[date_list_arr, close_list, option_type_list, T_list_arr, K_list_arr, sigma_list_arr]
        spline_db = pd.DataFrame(data, columns=['日期', '收盘价', 'option_type', 'T', 'K', 'sigma'])
        spline_db.to_csv(path_or_buf='date_arr/%s' % date, index=False, encoding='UTF-8')
    print(datesave)

if __name__ == '__main__':
    # path = 'sz50ETF.csv'
    # db = pd.read_csv(filepath_or_buffer=path, index_col=None)
    # db = db[["日期","期权代码","交易代码",'期权简称',"涨跌幅(%)","开盘价","收盘价","持仓量"]]
    # db = db.merge(trading_code_slice(db['交易代码']),left_index=True,right_index=True,how='outer')
    # db = db.merge(k_real(db['期权简称']),left_index=True,right_index=True,how='outer')
    # new_vix_db(db=db)
    spline_vix_db()
