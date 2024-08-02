# -*- coding: utf-8 -*-

# 라이브러리 불러오기
import seaborn as sns

# titanic 데이터셋 가져오기
df = sns.load_dataset('titanic')


print(df.info)
'''
<bound method DataFrame.info of      survived  pclass     sex   age  ...  deck  embark_town  alive  alone
0           0       3    male  22.0  ...   NaN  Southampton     no  False
1           1       1  female  38.0  ...     C    Cherbourg    yes  False
2           1       3  female  26.0  ...   NaN  Southampton    yes   True
3           1       1  female  35.0  ...     C  Southampton    yes  False
4           0       3    male  35.0  ...   NaN  Southampton     no   True
..        ...     ...     ...   ...  ...   ...          ...    ...    ...
886         0       2    male  27.0  ...   NaN  Southampton     no   True
887         1       1  female  19.0  ...     B  Southampton    yes   True
888         0       3  female   NaN  ...   NaN  Southampton     no  False
889         1       1    male  26.0  ...     C    Cherbourg    yes   True
890         0       3    male  32.0  ...   NaN   Queenstown     no   True

[891 rows x 15 columns]>
'''
#%%
# deck 열의 NaN 개수 계산하기
nan_deck = df['deck'].value_counts(dropna=False) 
# nan값의 개수
# dropna=False 결측값을 유지하고 해당 결측값의 빈도수 까지 계산
print(nan_deck)

# isnull() 메서드로 누락 데이터 찾기
print(df.head().isnull())
#누락데이터인 NAN값에 대해서 True 반환


# notnull() 메서드로 누락 데이터 찾기
print(df.head().notnull())
# 누락데이터인 NAN값에 대해서 False 반환


# df.head().isnull()에서 누락데이터의 총 개수:
# isnull() 메서드로 누락 데이터 개수 구하기
print(df.head().isnull().sum(axis=0)) # 행방향 합
# 각 열에서 누락된 데이터가 있는 행의 개수
print(df.isnull().sum(axis=0))
# age : 177 / deck : 688
'''
survived         0
pclass           0
sex              0
age            177
sibsp            0
parch            0
fare             0
embarked         2
class            0
who              0
adult_male       0
deck           688
embark_town      2
alive            0
alone            0
dtype: int64
'''

# 각 행에서 누락된 데이터가 있는 컬럼의 갯수
print(df.isnull().sum(axis=1))
'''
0      1
1      0
2      1
3      0
4      1
      ..
886    1
887    0
888    2
889    0
890    1
Length: 891, dtype: int64
'''









