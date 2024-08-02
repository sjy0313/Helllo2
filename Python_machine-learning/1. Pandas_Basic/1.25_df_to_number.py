# -*- coding: utf-8 -*-

# 라이브러리 불러오기
import pandas as pd
import seaborn as sns

 
# 데이터프레임을 연산
# 시리즈 연산을 확장하는 개념으로 접근
# 행/열 인덱스를 기준으로 정렬하고 일대일 대응되는 원소끼리 연산

# pip install seaborn
#%%


# titanic 데이터셋에서 age, fare 2개 열을 선택하여 데이터프레임 만들기
titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age','fare']]
print(df.head())   #첫 5행만 표시
print('\n')
print(type(df))
print('\n')
'''
 age     fare
0  22.0   7.2500
1  38.0  71.2833
2  26.0   7.9250
3  35.0  53.1000
4  35.0   8.0500


<class 'pandas.core.frame.DataFrame'>
'''
# 데이터프레임에 숫자 10 더하기
addition = df + 10
print(addition.head())   #첫 5행만 표시
print('\n')
print(type(addition))
'''
  age     fare
0  32.0  17.2500
1  48.0  81.2833
2  36.0  17.9250
3  45.0  63.1000
4  45.0  18.0500


<class 'pandas.core.frame.DataFrame'>
'''