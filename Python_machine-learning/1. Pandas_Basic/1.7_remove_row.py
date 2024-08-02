# -*- coding: utf-8 -*-

import pandas as pd

# DataFrame() 함수로 데이터프레임 변환. 변수 df에 저장 
exam_data = {'수학' : [ 90, 80, 70], '영어' : [ 98, 89, 95],
             '음악' : [ 85, 95, 100], '체육' : [ 100, 90, 90]}

df = pd.DataFrame(exam_data, index=['서준', '우현', '인아'])
print(df)
print('\n')

'''
   수학  영어   음악   체육
서준  90  98   85  100
우현  80  89   95   90
인아  70  95  100   90
'''
# 데이터프레임 df를 복제하여 변수 df2에 저장. df2의 1개 행(row)을 삭제
'''SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame'''
df2 = df # 원본을 참조하는 사본 [inplace=False]
df2 = df[:]
# df2= df[:].copy() 
#[권고사항] 복사해서 써라

df2.drop('우현', inplace=True)
print(df2)
''' 수학  영어   음악   체육
서준  90  98   85  100
인아  70  95  100   90
'''
print('\n')

# 데이터프레임 df를 복제하여 변수 df3에 저장. df3의 2개 행(row)을 삭제
df3 = df[:]
df3.drop(['우현', '인아'], axis=0, inplace=True)

print(df3)
''' 수학  영어  음악   체육
서준  90  98  85  100
'''
#%%
# axis = 1 : 컬럼을 삭제
# df4 = df[:]
# KeyError: "['우현', '인아'] not found in axis"
df4 = df[:].copy()
df4.drop(['우현', '인아'], axis=1, inplace=True)
print(df4)
'''
수학  영어   음악   체육
서준  90  98   85  100
우현  80  89   95   90
인아  70  95  100   90'''
#%%
#df5 = df[:]
df5 = df[:].copy()
# axis : 기본값 1, 인덱스 삭제
# KeyError: "['수성', '인수'] not found in axis"
df5.drop(['수성', '인수'], inplace=True)
print(df5)
















