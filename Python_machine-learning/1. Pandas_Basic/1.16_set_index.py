# -*- coding: utf-8 -*-

import pandas as pd

# DataFrame() 함수로 데이터프레임 변환. 변수 df에 저장 
exam_data = {'이름' : [ '서준', '우현', '인아'],
             '수학' : [ 90, 80, 70],
             '영어' : [ 98, 89, 95],
             '음악' : [ 85, 95, 100],
             '체육' : [ 100, 90, 90]}
df = pd.DataFrame(exam_data)
print(df)
print('\n')
'''
 이름  수학  영어   음악   체육
0  서준  90  98   85  100
1  우현  80  89   95   90
2  인아  70  95  100   90'''
# 특정 열(column)을 데이터프레임의 행 인덱스(index)로 설정 
ndf = df.set_index(['이름'])
print(ndf)
'''    수학  영어   음악   체육
이름                  
서준  90  98   85  100
우현  80  89   95   90
인아  70  95  100   90'''
print('\n')
ndf2 = ndf.set_index('음악')
print(ndf2)
'''
수학  영어   체육
음악              
85   90  98  100
95   80  89   90
100  70  95   90'''
print('\n')
ndf3 = ndf.set_index(['수학', '음악'])
print(ndf3)
'''
   영어   체육
수학 음악          
90 85   98  100
80 95   89   90
70 100  95   90'''
#%%

ndf5 = ndf3.loc[(80,95), ['체육','영어']]
print("체육",'\n', ndf5)


print("체육,영어:")
print(ndf6)