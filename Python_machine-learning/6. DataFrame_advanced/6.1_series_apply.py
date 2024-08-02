# -*- coding: utf-8 -*-

# 라이브러리 불러오기
import seaborn as sns

# titanic 데이터셋에서 age, fare 2개 열을 선택하여 데이터프레임 만들기
titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age','fare']]
# 'ten'열 index이고 값이 10인 열생성
df['ten'] = 10
print(df.head())
'''
 age     fare  ten
0  22.0   7.2500   10
1  38.0  71.2833   10
2  26.0   7.9250   10
3  35.0  53.1000   10
4  35.0   8.0500   10
'''
#%%


# 사용자 함수 정의
def add_10(n):   # 10을 더하는 함수
    return n + 10

def add_two_obj(a, b):    # 두 객체의 합
    return a + b

print(add_10(10))
print(add_two_obj(10, 10))

#%%

# 시리즈 객체에 add_10 즉 나이에 10씩 더해주는 함수를 적용하여 출력.

sr1 = df['age'].apply(add_10)               # n = df['age']의 모든 원소
print('sr1.shape:', sr1.shape, type(sr1))
#sr1.shape: (891,) <class 'pandas.core.series.Series'>
print(sr1.head())
'''
0    32.0
1    48.0
2    36.0
3    45.0
4    45.0
Name: age, dtype: float64'''

sr0 = df['age'] + 10
# 단순한 계산을 할 수 없는 경우 apply 함수 적용 

#%%

# 시리즈 객체와 숫자에 적용 : 2개의 인수(시리즈 + 숫자)
sr2 = df['age'].apply(add_two_obj, b=10)    # a=df['age']의 모든 원소, b=10
print(sr2.head())

#%%
#sr1 값에 대해서는 더하는 값 10으로 고정
#sr2 값의 경우 b에 따라 달라짐 따라서 위처럼 b=10값 정의하여 df['age']에 적용
#%%

# 람다 함수 활용: 시리즈 객체에 적용 (1)
def add_10(n):   # 10을 더하는 함수
    return n + 10

sr3 = df['age'].apply(lambda x: add_10(x))  # x=df['age']
print(sr3.head())

#%%

# 람다 함수 활용: 시리즈 객체에 적용 (2)
sr4 = df['age'].apply(lambda n : n + 10)  # x=df['age']
print(sr4.head())

#%%

sr5 = df['age'].apply(add_10)               # n = df['age']의 모든 원소
print(sr5.head())

#%%
# 칼럼 : age, ten을 apply() 함수로 전달하여 age+ten의 결과를 새로운 칼럼 
# ageten 만들어서 넣어라ㅣ
# 람다 함수 활용 : 시리즈 객체에 적용
# n : Series


df = titanic.loc[:, ['age','fare']]
df['ten'] = 10

'''
def addten(n):
    print(f"> addten : type({n}), ({n})")
    return n['age'] + n['ten']
'''

def addten(n):
    print(f"> addten : type({n})")
    print(n)
    print('=' * 30)
    return n + 10
    # return n['age'] + n['ten']
# 열 단위 순회 : 3번 순회 
r1 = df.apply(addten, axis = 0)
print('=' * 30)
print(r1)

#%%



ageten2 = df.apply(addten, axis=1)
'''
> addten : type(age     32.00
fare     7.75
ten     10.00
Name: 890, dtype: float64), (age     32.00
fare     7.75
ten     10.00
Name: 890, dtype: float64)'''
    
#%% 
# axis = 0 -> 행단위지만 개별 탐색을 할 때는 열단위가 됨
# axis = 1 -> 열단위지만 개별 탐색을 할 때는 행단위가 됨
ageten = df.apply((lambda n : n['age'] + n['ten']), axis = 1)
# %%
# a는 개수만큼 b는 통으로 호출
#%%
df['ageten'] = ageten
print(df.head())






