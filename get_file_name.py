# -*- coding: UTF-8 -*-
import os
import pandas as pd
import numpy as np
from pandas import DataFrame
def iterbrowse(path):
    for home, dirs, files in os.walk(path):
        for filename in files:
            yield os.path.join(home, filename)

#遍历文件夹获取文件名
def get_filename(path=r"F:\Workspace\python\Project\FE_practice\data", n=10):
    filename=[]
    for fullname in iterbrowse(path):
        filename.append(fullname[-n:])
    return filename
def main1():
    for code_id in get_filename():
        data_path='stock_data/'+code_id
        table=pd.read_csv(data_path)
        length=len(table)
        lograte=np.log([float(table.iat[i-1,3])/float(table.iat[i,3]) for i in range(length-1,1,-1)])
        data=DataFrame(np.c_[lograte],index=table['date'][:-2],columns=[code_id[:6]])
        print(data)
