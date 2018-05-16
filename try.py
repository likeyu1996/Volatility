'''
import xlrd
from xlrd import xldate_as_tuple

book = xlrd.open_workbook('sz50ETF.xls')
sheet = book.sheet_by_name('Sheet1')
for row in range(sheet.nrows):
    for col in range(sheet.ncols):
        value = sheet.cell(row,col).value
        if sheet.cell(row,col).ctype == 3:
            date = xldate_as_tuple(sheet.cell(row,col).value,0)
            value = datetime.datetime(*date)
        print(value)
'''


l=[1,2,3,4,5,6,7]
ll=[1,2,2,3,4,5,6,7,7]
# if l !=list(set(ll)):
#     print('不等于')
# else:
#     print('等于')
#
# if l !=set(ll):
#     print('不等于')
# else:
#     print('等于')
# print(set(ll))
import numpy as np
import pandas as pd
l2 = [2,3,4,4.5,5,6,7,8]
l3 = [3,4,5,6,7,7.5,8,9]
# data = np.c_[l,l2,l3]
# frame = pd.DataFrame(data,columns=['l','l2','l3'])
# print(frame)
# print(frame.drop([1,2]))
# print(frame.drop([1,2],inplace=True))
# print(frame.index)
# l4 = []
# l4.append(j for j in l2)
# print(l4)
# print(l2+l3)
# print(frame.drop([]))
l5=[1,1,2,2,3,4,5,5,5,6,7]
# result = np.unique(l5,return_counts=True)
# print([result[0][i] for i in range(len(result[0])) if result[1][i] != 1])
print(np.unique(l5))
l6 = list(set(l2+l3))
ll2 = [i for i in l6 if l2[0]<=i<=l2[-1]]
ll3 = [i for i in l6 if l3[0]<=i<=l3[-1]]
print(ll2)
print(ll3)
print(l[0:4])
print(l5[-1],l5[5:],l5[5:-1])
