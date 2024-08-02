# -*- coding: utf-8 -*-

import pandas as pd

# read_csv() 함수로 df 생성
df = pd.read_csv('./auto-mpg.csv', header=None)

# 열 이름을 지정
df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

# 데이터프레임 df의 각 열이 가지고 있는 원소 개수 확인 
print(df.count())
'''
mpg             398
cylinders       398
displacement    398
horsepower      398
weight          398
acceleration    398
model year      398
origin          398
name            398
dtype: int64
'''
print('\n')

# df.count()가 반환하는 객체 타입 출력
print(type(df.count())) # <class 'pandas.core.series.Series'>

print('\n')

# 데이터프레임 df의 특정 열이 가지고 있는 고유값 확인 
unique_values = df['origin'].value_counts() 
print(unique_values)
'''
'origin
1    249
3     79
2     70
Name: count, dtype: int64
'''
print('\n')

# value_counts 메소드가 반환하는 객체 타입 출력
print(type(unique_values)) # <class 'pandas.core.series.Series'>







