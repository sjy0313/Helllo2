# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:01:34 2024

@author: Shin
"""

import seaborn as sns
import pandas as pd 
import numpy as np

# titanic 데이터셋에서 age, fare 2개 열을 선택하여 데이터프레임 만들기
titanic = sns.load_dataset('titanic')

print(titanic.dtypes)
'''
age             float64
fare            float64
'''
# 만약 만 18세 이상 fare + 10  / 50세 이상 fare - 10
#%%
# nan처리를 위해 str 타입으로 변환

df = titanic.loc[:, ['age','fare']]

df['age'] = df['age'].astype('str')     
print(df['age'].dtypes) 


# nan 값 np.nan변환
df['age'] = df['age'].replace({ 'nan' : np.nan}) 
df.dropna(subset=['age'], axis=0, inplace=True) # # 누락데이터 행을 삭제


df['age'] = df['age'].astype('category')     
print(df['age'].dtypes) 

print(df.dtypes)
# age            category

# 범주형 age열 -> 순서가 있는 열로 변환 
df['age'] = df['age'].cat.as_ordered()
# np.histogram 함수로 3개의 bin으로 나누는 경계 값의 리스트 구하기
count, bin_dividers = np.histogram(df['age'], bins=3)
# horsepower의 구간을 3개로 나누어 작업 실행 각bin값에 속한 값의 수를 계산하고
# 각 구간의  bin_dividers(구간분할자)도 반환
print(bin_dividers) 
# [ 0.42       26.94666667 53.47333333 80.        ]
# 0~26세 : 319명 / 27~53 : 345명 / 53세 이상 : 50명


age_min = df['age'].min() # 0.42
age_max = df['age'].max() # 80.00
age_bin = (age_max - age_min) / 3 # 26.52
print("age_min: ", df['age'].min)

bin_names = ['청년층', '중년층', '고령층']

# pd.cut 함수로 각 데이터를 3개의 bin에 할당
df['연령층'] = pd.cut(x=df['age'],     # 데이터 배열
                      bins=bin_dividers,      # 경계 값 리스트
                      labels=bin_names,       # bin 이름
                      include_lowest=True)    # 첫 경계값 포함 

# age 열, hp_bin 열의 첫 15행을 출력
print(df[['age', '연령층']].head(15))

print(df['연령층'].dtypes) # category
# 청년층에 대해 요금추가 부과 fare + 10
# 500이상의 fare를 낸 중년층에 대해 할인 fare - 100 
# 노년층에 대해 요금할인 fare - 5
#%%

# 문제
# 함수이름 : user_dataset_apply
# 파라미터 : df, 컬럼목록, 새로운컬럼, 처리함수, 전달인자
# 기능설명 : 
#   데이터프레임 : 처리대상
#   컬럼목록 : 처리대상
#   새로운컬럼 : 컬럼목록으로 처리 함수를 적용한 결과 컬럼
#   처리함수 : 처리해야할 기능 함수
#   전달인자 : 처리함수에 전달할 인자
# 처리함수 : 처리로직은 



# [ 0.42       26.94666667 53.47333333 80.        ]
# 0~26세 : 319명 / 27~53 : 345명 / 53세 이상 : 50명
# 청년층에 대해 요금추가 부과 fare + 10
# 500이상의 fare를 낸 중년층에 대해 할인 10%
# 노년층에 대해 요금할인 fare - 5


# 예 ) 

def user_dataset_apply(df,cols,newcols,func,args):
    ndf=df.copy()
    return ndf
#%%
      
df_age = df.sort_values(by='age')
df['age'] = df['age'].astype('float')

# 연령층 열 df_byage 지정
byage = df_age['연령층']
# 나이 열 ages 지정
ages = df_age['age']
# 요금 열 ages 지정
fare = df_age['fare']    


# 연령층 별 요금 변동을 계산해주는 함수 
def user_dataset_apply(df_age,cols,newcols,func,args):    
     
    ndf = df_age[cols].copy()
    
    # 사용자 정의 함수 적용
    ndf[newcols] = func(*args)
    
    return ndf

# 예제로 사용할 함수

def fare_function(byage):
    if byage == '청년층':
        return fare + 10
    
    elif byage == '중년층' and fare >= 500:
        return byage - (fare // 10)
    elif byage == '노년층':
        return fare - 5
    else:
        return fare

# 새로운 열에 요금 변동 적용
df_age['연령층별요금'] = df_age.apply(fare_function, axis=1)

#print(titanic[['age', 'fare', '연령층', '연령층별요금']].head())

    
young_df = user_dataset_apply(df_age, ['age', 'fare', '연령층'], 
                               ['fare_in_different_ages'], fare_function, byage)
print(young_df.head())





middle_df = user_dataset_apply(df_age, ['age', 'fare', '연령층'], 
                               ['fare_in_different_ages'], fare_function, [ages] )
print(middle_df.head())

elder_df = user_dataset_apply(df_age, ['age', 'fare', '연령층'], 
                               ['fare_in_different_ages'], fare_function, [ages] )
print(elder_df.head())
     
     
     