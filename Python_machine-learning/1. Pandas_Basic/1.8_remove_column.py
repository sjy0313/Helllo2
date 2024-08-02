# -*- coding: utf-8 -*-

import pandas as pd

# DataFrame() 함수로 데이터프레임 변환. 변수 df에 저장 
#  dict -> dataframe
exam_data = {'수학' : [ 90, 80, 70], '영어' : [ 98, 89, 95],
             '음악' : [ 85, 95, 100], '체육' : [ 100, 90, 90]}

df = pd.DataFrame(exam_data, index=['서준', '우현', '인아']) # 인덱스 지정 
print(df) 
print('\n')
'''수학  영어   음악   체육
서준  90  98   85  100
우현  80  89   95   90
인아  70  95  100   90'''

# 데이터프레임 df를 복제하여 변수 df4에 저장. df4의 1개 열(column)을 삭제
df4 = df.copy()
df4.drop('수학', axis=1, inplace=True)
print(df4)
'''    영어   음악   체육
서준  98   85  100
우현  89   95   90
인아  95  100   90'''
print('\n')

# 데이터프레임 df를 복제하여 변수 df5에 저장. df5의 2개 열(column)을 삭제
df5 = df.copy()
df5.drop(['영어', '음악'], axis=1, inplace=True)
print(df5)
'''수학   체육
서준  90  100
우현  80   90
인아  70   90
'''
