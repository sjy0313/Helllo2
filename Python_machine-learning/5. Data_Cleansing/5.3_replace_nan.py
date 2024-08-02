# -*- coding: utf-8 -*-

# 라이브러리 불러오기
import seaborn as sns

# titanic 데이터셋 가져오기
df = sns.load_dataset('titanic')

# age 열의 첫 10개 데이터 출력 (5 행에 NaN 값)
print(df['age'].head(10))
print('\n')

# age 열의 NaN값을 다른 나이 데이터의 평균으로 변경하기

#[[]] -> dataframe으로 변경
# age/fare열의 평균 - >  age평균으로 변경하기 
mean_age = df[['age','fare']].mean(axis=0) # 나이,요금의 평균 계산 (NaN 값 제외)
mean_age_fare = mean_age.mean(axis=0) # age,fare의 평균
#df['age'].fillna(mean_age_fare, inplace=True)


# 나이의 평균값으로 nan값 변경하기
mean_age = df['age'].mean(axis=0)   # age 열의 평균 계산 (NaN 값 제외)
print(mean_age)
# df['age'].fillna(mean_age, inplace=True)
df_age = df.fillna({'age' : mean_age})
df['age'].fillna({'age' : mean_age}, inplace=True)


# age 열의 첫 10개 데이터 출력 (5 행에 NaN 값이 평균으로 대체)
print(df['age'].head(10))

age_isnull = df['age'].head(10)