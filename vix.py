# -*- coding: UTF-8 -*-

import numpy as np
import time
import datetime
import calendar


class IVX:
    # 导入结构为dataframe
    # 剩余时间T，无风险利率r由外部导入
    # NT=T*m_year
    # 每次只需计算一种到期，近月和次近月，故面板里到期日是一样的
    # 内部函数一般返回两个值，分别为近月和次近月
    # 默认了近月和次近月都有相同的执行价格，故先筛选的K再筛选的近与次近，对近月和次近月分别求K_list更保险
    # GG，换月的时候市场上仍有四个期权，即次近月、再下月和后两个季月的，但新加入的下月合约的执行价会与原有合约不同
    def __init__(self, data, r):
        self.data = data
        self.r = r
        self.T_list = list(set(self.data['T']))
        self.T_list.sort()
        self.K_list = list(set(self.data['K']))
        self.K_list.sort()
        self.T_0 = self.T()[0]
        self.T_1 = self.T()[1]
        self.T1 = self.t_remain()[0]
        self.T2 = self.t_remain()[1]
        self.NT1 = self.t_remain()[2]
        self.NT2 = self.t_remain()[3]
        self.N30 = self.t_remain()[4]
        self.N365 = self.t_remain()[5]
        # 近月合约
        self.s_0 = self.s_0()
        self.f_0 = self.f_0()
        self.k00 = self.k00()
        self.sum_u0 = self.k_sum_0()[0]
        self.sum_v0 = self.k_sum_0()[1]
        self.sum_w0 = self.k_sum_0()[2]
        self.sum_x0 = self.k_sum_0()[3]
        self.sigma2_0 = self.sigma2_0()
        self.u_0 = self.u_0()
        self.v_0 = self.v_0()
        self.w_0 = self.w_0()
        self.x_0 = self.x_0()
        self.sigma2_gvix_0 = self.sigma2_gvix_0()
        # 次近月合约，注意移仓换月问题
        self.s_1 = self.s_1()
        self.f_1 = self.f_1()
        self.k01 = self.k01()
        self.sum_u1 = self.k_sum_1()[0]
        self.sum_v1 = self.k_sum_1()[1]
        self.sum_w1 = self.k_sum_1()[2]
        self.sum_x1 = self.k_sum_1()[3]
        self.sigma2_1 = self.sigma2_1()
        self.u_1 = self.u_1()
        self.v_1 = self.v_1()
        self.w_1 = self.w_1()
        self.x_1 = self.x_1()
        self.sigma2_gvix_1 = self.sigma2_gvix_1()

    def T(self):
        # 返回近月合约或次近月合约的到期日
        j0 = self.T_list[0]
        j1 = self.T_list[1]
        return j0,j1

    def t_remain(self):
        # 返回值为公式中的T和nt，共四个
        date0 = time.strptime(self.T_0,"%Y-%m-%d")
        date1 = time.strptime(self.T_1,"%Y-%m-%d")
        date_ori = time.strptime(self.data['日期'][0],"%Y-%m-%d")
        date0 = datetime.datetime(date0[0],date0[1],date0[2])
        date1 = datetime.datetime(date1[0],date1[1],date1[2])
        date_ori = datetime.datetime(date_ori[0],date_ori[1],date_ori[2])
        remain0 = date0-date_ori
        remain1 = date1-date_ori
        # print(remain0,remain1)# 类型为datetime.timedelta
        # 获取剩余天数
        remain0 = remain0.days
        remain1 = remain1.days
        # 由于是日数据，所以只考虑开盘时间8:30，收盘时间15:30
        def m_leap(year):
            if calendar.isleap(year):
                m_year = 366*24*60
            else:
                m_year = 365*24*60
            return m_year
        m_year_0 = m_leap(date0.year)
        m_year_1 = m_leap(date1.year)
        m_curr = 8.5*60  # 15:30闭市
        m_sett = 8.5*60  # 8:30开市
        m_other_0 = (remain0-1)*24*60
        m_other_1 = (remain1-1)*24*60
        nt_remain0 = m_curr+m_sett+m_other_0
        nt_remain1 = m_curr+m_sett+m_other_1
        remain0 = nt_remain0/m_year_0
        remain1 = nt_remain1/m_year_1
        # 添加返回N30与N365的功能
        now_year = date_ori.year
        now_month = date_ori.month + 1  # 下个月
        if now_month == 13:
            now_year += 1
            now_month = 1
        n30 = calendar.monthrange(now_year, now_month)[1]*24*60
        # 输出的是一个元组，第一个元素是上一个月的最后一天为星期几(0-6),星期天为0;
        # 第二个元素是这个月的天数。
        # 直接用作时间差的话，不好处理2.29的情况
        n365 = m_leap(now_year)
        return remain0, remain1, nt_remain0, nt_remain1, n30, n365

    def s_0(self):
        min_delta_0 = 65536.0
        min_K_0 = 0
        for i in self.K_list:
            db_temp_0 = self.data[(self.data['K'] == i) & (self.data['T'] == self.T_0)]
            index_0 = db_temp_0.index.tolist()
            delta_0 = abs(db_temp_0['收盘价'][index_0[0]]-db_temp_0['收盘价'][index_0[1]])
            if delta_0 < min_delta_0:
                min_delta_0 = delta_0
                min_K_0 = i
            #else: # 为什么这句raise总是被触发
                #raise ValueError('delta is lager than 65536!')
        return min_K_0

    def s_1(self):
        min_delta_1 = 65536.0
        min_K_1 = 0
        for i in self.K_list:
            db_temp_1 = self.data[(self.data['K'] == i) & (self.data['T'] == self.T_1)]
            index_1 = db_temp_1.index.tolist()
            delta_1 = abs(db_temp_1['收盘价'][index_1[0]]-db_temp_1['收盘价'][index_1[1]])
            if delta_1 < min_delta_1:
                min_delta_1 = delta_1
                min_K_1 = i
            #else: # 为什么这句raise总是被触发
                #raise ValueError('delta is lager than 65536!')
        return min_K_1
# 表示认购期权价格与认沽期权价格相差最小的执行价
# for循环，计算每个执行价对应的abs(data[i][c]-data[i][p])
# 字典法：
# prices{'HPQ': 37.2, 'FB': 10.75, 'AAPL': 612.78, 'IBM': 205.55, 'ACME': 45.23}
# min(prices, key=lambda k: prices[k])

    def f_0(self):
        db_temp_0 = self.data[(self.data['K'] == self.s_0) & (self.data['T'] == self.T_0)]
        index_0 = db_temp_0.index.tolist()
        f_value_0 = self.s_0 + np.exp(self.r*self.T1)*(db_temp_0['收盘价'][index_0[0]]-db_temp_0['收盘价'][index_0[1]])
        # 默认看涨的代码小于看跌的
        return f_value_0

    def f_1(self):
        db_temp_1 = self.data[(self.data['K'] == self.s_1) & (self.data['T'] == self.T_1)]
        index_1 = db_temp_1.index.tolist()
        f_value_1 = self.s_1 + np.exp(self.r*self.T2)*(db_temp_1['收盘价'][index_1[0]]-db_temp_1['收盘价'][index_1[1]])
        # 默认看涨的代码小于看跌的
        return f_value_1

    def k00(self):
        k00 = 0
        if self.K_list[0] == self.f_0:
            k00 = self.K_list[0]
        else:
            for i in range(len(self.K_list)):
                if self.K_list[i] >= self.f_0:
                    k00 = self.K_list[i-1]
                    break
                else:
                    k00 = self.K_list[-1]
        return k00

    def k01(self):
        k01 = 0
        if self.K_list[0] == self.f_1:
            k01 = self.K_list[0]
        else:
            for j in range(len(self.K_list)):
                if self.K_list[j] >= self.f_1:
                    k01 = self.K_list[j-1]
                    break
                else:
                    k01 = self.K_list[-1]
        return k01

    def k_sum_0(self):
        delta_Ki = []
        sum_0 = 0
        sum_v0 = 0
        sum_w0 = 0
        sum_x0 = 0
        for i in range(len(self.K_list)):
            if i == 0:
                delta_Ki.append(self.K_list[i+1]-self.K_list[i])
            elif i == len(self.K_list)-1:
                delta_Ki.append(self.K_list[i] - self.K_list[i-1])
            else:
                delta_Ki.append((self.K_list[i+1] - self.K_list[i-1])/2.0)

        def pki(Ki):
            db_temp_0 = self.data[(self.data['K'] == Ki) & (self.data['T'] == self.T()[0])]
            index_0 = db_temp_0.index.tolist()
            if Ki < self.k00:
                pki_0 = db_temp_0['收盘价'][index_0[1]]
            elif Ki == self.k00:
                pki_0 = (db_temp_0['收盘价'][index_0[0]]+db_temp_0['收盘价'][index_0[1]])/2.0
            else:
                pki_0 = db_temp_0['收盘价'][index_0[0]]
            return pki_0
        for i in range(len(self.K_list)):
            sum_0 += (delta_Ki[i]/self.K_list[i]**2)*pki(self.K_list[i])
            sum_v0 += (delta_Ki[i]/self.K_list[i]**2)*pki(self.K_list[i])*(1-np.log(self.K_list[i]/self.s_0))
            sum_w0 += (delta_Ki[i]/self.K_list[i]**2)*pki(self.K_list[i])*(2*np.log(self.K_list[i]/self.s_0)-np.log(self.K_list[i]/self.s_0)**2)
            sum_x0 += (delta_Ki[i]/self.K_list[i]**2)*pki(self.K_list[i])*(3*np.log(self.K_list[i]/self.s_0)**2-np.log(self.K_list[i]/self.s_0)**3)
        return sum_0, sum_v0, sum_w0, sum_x0

    def k_sum_1(self):
        delta_Ki = []
        sum_1 = 0
        sum_v1 = 0
        sum_w1 = 0
        sum_x1 = 0
        for i in range(len(self.K_list)):
            if i == 0:
                delta_Ki.append(self.K_list[i+1]-self.K_list[i])
            elif i == len(self.K_list)-1:
                delta_Ki.append(self.K_list[i] - self.K_list[i-1])
            else:
                delta_Ki.append((self.K_list[i+1] - self.K_list[i-1])/2.0)

        def pki(Ki):
            db_temp_1 = self.data[(self.data['K'] == Ki) & (self.data['T'] == self.T()[1])]
            index_1 = db_temp_1.index.tolist()
            if Ki < self.k01:
                pki_1 = db_temp_1['收盘价'][index_1[1]]
            elif Ki == self.k01:
                pki_1 = (db_temp_1['收盘价'][index_1[0]]+db_temp_1['收盘价'][index_1[1]])/2.0
            else:
                pki_1 = db_temp_1['收盘价'][index_1[0]]
            return pki_1
        for i in range(len(self.K_list)):
            sum_1 += (delta_Ki[i]/self.K_list[i]**2)*pki(self.K_list[i])
            sum_v1 += (delta_Ki[i]/self.K_list[i]**2)*pki(self.K_list[i])*(1-np.log(self.K_list[i]/self.s_1))
            sum_w1 += (delta_Ki[i]/self.K_list[i]**2)*pki(self.K_list[i])*(2*np.log(self.K_list[i]/self.s_1)-np.log(self.K_list[i]/self.s_1)**2)
            sum_x1 += (delta_Ki[i]/self.K_list[i]**2)*pki(self.K_list[i])*(3*np.log(self.K_list[i]/self.s_1)**2-np.log(self.K_list[i]/self.s_1)**3)
        return sum_1, sum_v1, sum_w1, sum_x1

    def sigma2_0(self):
        sigma2_0 = (2.0/self.T1)*np.exp(self.r*self.T1) * self.sum_u0 - \
                (1.0/self.T1)*(self.f_0/self.k00-1)**2
        return sigma2_0

    def sigma2_1(self):
        sigma2_1 = (2.0/self.T2)*np.exp(self.r*self.T2) * self.sum_u1 - \
                (1.0/self.T2)*(self.f_1/self.k01-1)**2
        return sigma2_1

    def u_0(self):
        u_0 = np.log(self.k00/self.s_0)+(self.f_0/self.k00-1) - \
              np.exp(self.r*self.T1)*self.sum_u0
        return u_0

    def v_0(self):
        v_0 = np.log(self.k00/self.s_0)**2 + \
              2*np.log(self.k00/self.s_0)*(self.f_0/self.k00-1) + \
              2*np.exp(self.r*self.T1)*self.sum_v0
        return v_0

    def w_0(self):
        w_0 = np.log(self.k00/self.s_0)**3 + \
              3*np.log(self.k00/self.s_0)**2*(self.f_0/self.k00-1) + \
              3*np.exp(self.r*self.T1)*self.sum_w0
        return w_0

    def x_0(self):
        x_0 = np.log(self.k00/self.s_0)**4 + \
              4*np.log(self.k00/self.s_0)**3*(self.f_0/self.k00-1) + \
              4*np.exp(self.r*self.T1)*self.sum_x0
        return x_0

    def u_1(self):
        u_1 = np.log(self.k01/self.s_1)+(self.f_1/self.k01-1) - \
              np.exp(self.r*self.T2)*self.sum_u1
        return u_1

    def v_1(self):
        v_1 = np.log(self.k01/self.s_1)**2 + \
              2*np.log(self.k01/self.s_1)*(self.f_1/self.k01-1) + \
              2*np.exp(self.r*self.T2)*self.sum_v1
        return v_1

    def w_1(self):
        w_1 = np.log(self.k01/self.s_1)**3 + \
              3*np.log(self.k01/self.s_1)**2*(self.f_1/self.k01-1) + \
              3*np.exp(self.r*self.T2)*self.sum_w1
        return w_1

    def x_1(self):
        x_1 = np.log(self.k01/self.s_1)**4 + \
              4*np.log(self.k01/self.s_1)**3*(self.f_1/self.k01-1) + \
              4*np.exp(self.r*self.T2)*self.sum_x1
        return x_1

    def sigma2_gvix_0(self):
        sigma2_gvix_0 = (1.0/self.T1)*self.v_0-self.u_0**2
        return sigma2_gvix_0

    def sigma2_gvix_1(self):
        sigma2_gvix_1 = (1.0/self.T2)*self.v_1-self.u_1**2
        return sigma2_gvix_1

    def ivx(self):
        if self.NT1/(60.0*24.0) >= 30.0:
            ivx = 100*np.sqrt(self.sigma2_0)
        else:
            ivx = 100*np.sqrt((self.T1*self.sigma2_0*(self.NT2-self.N30)/(self.NT2-self.NT1)+self.T2*self.sigma2_1*(self.N30-self.NT1)/(self.NT2-self.NT1))*self.N365/self.N30)
        return ivx

    def gvix(self):
        if self.NT1/(60.0*24.0) >= 30.0:
            gvix = 100*np.sqrt(self.sigma2_gvix_0)
        else:
            gvix = 100*np.sqrt((self.T1*self.sigma2_gvix_0*(self.NT2-self.N30)/(self.NT2-self.NT1)+self.T2*self.sigma2_gvix_1*(self.N30-self.NT1)/(self.NT2-self.NT1))*self.N365/self.N30)
        return gvix
    
    def outer_sigma2(self):
        return self.sigma2_0,self.sigma2_1

    def outer_sigma2_gvix(self):
        return self.sigma2_gvix_0,self.sigma2_gvix_1
