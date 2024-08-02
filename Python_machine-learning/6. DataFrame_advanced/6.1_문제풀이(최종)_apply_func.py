# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 12:01:35 2024

@author: Shin
"""
import seaborn as sns
import pandas as pd 

# titanic 데이터셋에서 age, fare 2개 열을 선택하여 데이터프레임 만들기
titanic = sns.load_dataset('titanic')

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

#%%
# agent 역할 
def user_dataset_apply(df, cols, newcol, func, args):    
     
    ndf = df.loc[:, cols] # 새로운 df
    ndf[newcol] = 0.0 # 새로운 컬럼명
    
    for n in range(len(ndf)):
        lx = ndf.index[n]
        ncols = ndf.loc[lx,cols].astype(float)
        ndf.loc[lx,newcol] = func(ncols, args)
        
    return ndf 

#%%
# 핵심 적용함수 (가중치)
def user_func(ncols, args):
    tot = 0.0
    for col in ncols:
        if pd.isnull(col) != True and pd.isna(col) != True:
            tot += col 
            
    return tot + args
#%%
# 평균
def user_mean(ncols, args):
    tot = 0.0
    for col in ncols:
        if pd.isnull(col) != True and pd.isna(col) != True:
            tot += col 
            
    return tot / args


   
#%%

def user_fare_apply(df, age, fare, newcol, func):  
    cols = [age, fare]
    ndf = df.loc[:, cols] # 새로운 df 
    ndf[newcol] = 0.0 # 새로운 컬럼명
    
    for n in range(len(df)):
        lx = ndf.index[n]
        ncols = ndf.loc[lx,cols]
        ndf.loc[lx, newcol] = func(ncols.iloc[0], ncols.iloc[1])
        
    return ndf

    
#%%
# 나이별 요금
def user_fare(age, fare):
    if age <= 20:
        return fare + 0.7
    elif age <= 50: # 50세이상 0.5
        return fare
    else:
        return fare + 0.5 
    
#%%
# 연령별로 요금을 차등
df1 = user_fare_apply(titanic.iloc[0:10,:], 'age', 'fare', 'new_fare', user_fare)
print(df1)



#%%
# 대상의 모든 컬럼을 더하고 가중치를 더함
ndf1 = user_dataset_apply(titanic.iloc[0:10, :], ['age', 'fare'], 'weight', user_func, 10.0)
print(ndf1)
#%%
# 대상의 모든 컬럼으로 평균을 구함
ndf2 = user_dataset_apply(titanic.iloc[0:10, :], ['age', 'fare'], 'avg', user_mean, 2.0)
print(ndf2)   

   