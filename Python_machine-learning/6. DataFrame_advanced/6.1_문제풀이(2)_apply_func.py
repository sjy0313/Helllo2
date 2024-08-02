# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:30:03 2024

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

#%%
# nan처리를 위해 str 타입으로 변환

df = titanic.loc[:, ['age','fare']]

# nan 값 np.nan변환
# IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer
# NaN값이 포함 되어있으므로 정수형으로 변환 전에 누락값 삭제
df['age'] = df['age'].replace({ 'nan' : np.nan}) 
df.dropna(subset=['age'], axis=0, inplace=True) # # 누락데이터 행을 삭제

#%%

# 정수형으로 변환
df['age'] = df['age'].round().astype(int)   
print(df['age'].dtypes) 
# 카테고리형으로 변환
df['age'] = df['age'].astype('category')     
print(df['age'].dtypes) 
print(df.dtypes)
'''
age     category
fare     float64
dtype: object
'''
#%%

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

# age 열, 연령층 열의 첫 15행을 출력
print(df[['age', '연령층']].head(15))

print(df['연령층'].dtypes) # category
# 청년층에 대해 요금추가 부과 fare + 10
# 500이상의 fare를 낸 중년층에 대해 할인 10%
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
# 처리함수 : 처리로직은 선택
# 리턴 : 새로운 데이터프레임 



# [ 0.42       26.94666667 53.47333333 80.        ]
# 0~26세 : 319명 / 27~53 : 345명 / 53세 이상 : 50명
# 청년층에 대해 요금추가 부과 fare + 10
# 500이상의 fare를 낸 중년층에 대해 할인 10%
# 노년층에 대해 요금할인 fare - 5


# 예 ) 
'''
def user_dataset_apply(df,cols,newcols,func,args):
    ndf=df.copy()
    return ndf
'''


s1 = df.loc[0, ['age', 'fare']]
'''
age     22.00
fare     7.25
Name: 0, dtype: float64
'''
print(type(s1)) # seires 

df.loc[0, 'new_fare'] 
s1.iloc[0]
#%%


byage = df['연령층']
ages = df['age']
fare = df['fare']

def user_fare_apply(df, age, fare, newcol, func):  
    cols = [age, fare]
    ndf = df.loc[:, cols] # 새로운 df 
    ndf[newcol] = 0.0 # 새로운 컬럼명
    
    for n in range(len(df)):
        lx = ndf.index[n] # df의 index정의 
        ncols = ndf.loc[lx,cols] # age, fare 값 출력
        ndf.loc[lx, newcol] = func(ncols.iloc[0], ncols.iloc[1]) 
        # ncols의 첫쨰와 두번쨰값 func에 넣어줘
        
    return ndf

#%%
#선생님 방법: 
def user_fare_apply(df, age, fare, newcol, func):  
    cols = [age, fare]
    ndf = df.loc[:, cols] # 새로운 df 
    ndf[newcol] = ndf.apply(func,axis=1)
    return ndf

# axis =1  : apply function to each row
# 행 단위로 데이터를 시리즈로 받음. 
def user_fare(ncols):
    age = ncols.loc['age']
    fare = ncols.loc['fare']
    if age <= 20:
        return fare + 0.7
    elif age <= 50:
        return fare
    else:
        return fare + 0.5
    
#%%

def fare_func(age, fare):
    if age == '청년층':
        return fare + 10
    elif age == '중년층' and fare >= 500:
        return fare - (fare // 10)
    elif age == '노년층':
        return fare - 5
    else:
        return fare + 1 # 이외에 조건에 대하여 +1 부과
    
fare_byage = user_fare_apply(df, '연령층', 'fare', 'fare_by_age', fare_func)
print(fare_byage.head())

 




#%%    
# 방법 1) 
# 연령층/age/fare 열 이름 지정 
byage = df['연령층']
ages = df['age']
fare = df['fare']

def user_dataset_apply(df, cols, newcol, func):
    ndf = df.loc[:, cols].copy()  # Make a copy of selected columns
    ndf[newcol] = ndf.apply(lambda row: func(row[cols[0]], row[cols[1]]), axis=1)
    return ndf

def fare_function(age, fare):
    if age == '청년층':
        return fare + 10
    elif age == '중년층' and fare >= 500:
        return fare - (fare // 10)
    elif age == '노년층':
        return fare - 5
    else:
        return fare + 1 

age_df = user_dataset_apply(df, ['연령층', 'fare'], 'fare_in_different_ages', fare_function)
print(age_df.head())


