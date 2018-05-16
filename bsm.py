# -*- coding: UTF-8 -*-
import numpy as np
from scipy import stats
from scipy.optimize import fsolve


class BSM:
    def __init__(self, S, K, T, r, sigma, option_type):
        self.S = float(S)
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.type = option_type

    def d1(self):
        return (np.log(self.S/self.K)+(self.r+0.5*self.sigma**2)*self.T)/(self.sigma*self.T**0.5)

    def d2(self):
        return (np.log(self.S/self.K)-(self.r+0.5*self.sigma**2)*self.T)/(self.sigma*self.T**0.5)

    def call_value(self):
        c = self.S*stats.norm.cdf(self.d1(), 0.0, 1.0)\
            - self.K*np.exp(-self.r*self.T)*stats.norm.cdf(self.d2(), 0.0, 1.0)
        if c < 0:
            c = 0
        return c

    def put_value(self):
        p = -self.S*stats.norm.cdf(-self.d1(), 0.0, 1.0)\
            + self.K*np.exp(-self.r*self.T)*stats.norm.cdf(-self.d2(), 0.0, 1.0)
        if p < 0:
            p = 0
        return p
    '''
    If you take a look at Vega in BS which is:
    SN'(d1)sqrt(T-t), where d1=(log(S/K)-0.5*sigma^2(T-t))/(sigma*sqrt(T-t)), 
    N'() reaches the maximum point when d1=0. So when sigma or T-t is not so big,
    S=K approximately can make d1 very close to zero. 
    But this does not hold if T-t or sigma is very huge.
    '''
    def vega(self):
        vega = self.S*stats.norm.pdf(self.d1(), 0.0, 1.0)*(self.T**0.5)
        return vega

    def imp_sigma(self, c0, sigma_est=0.2):
        option = BSM(self.S, self.K, self.T, self.r, sigma_est, self.type)

        def difference(sigma):
            option.sigma = sigma
            if self.type == "C":
                option_value = option.call_value()
            elif self.type == "P":
                option_value = option.put_value()
            else:
                raise ValueError("Default Option Type")
            return option_value-c0
        iv = fsolve(difference, sigma_est)[0]
        # fsolve函数就是这么用的
        return iv
