# -*- coding: utf-8 -*-

# 라이브러리 불러오기
import seaborn as sns

# titanic 데이터셋 가져오기
df = sns.load_dataset('titanic')

# for 반복문으로 각 열의 NaN 개수 계산하기
# 유효하지 않는 데이터는 True, 유효한 데이터는 False
missing_df = df.isnull()


# 전체 열에 대해 nan값의 개수 출력하기
missing_counts = missing_df.sum()
print(missing_counts)

#age 열에 대해 boolean배열 출력
col = 'age'
missing_count = missing_df[col].value_counts()
print(missing_count)
'''
age
False    714
True     177
Name: count, dtype: int64
'''
#%%
missing_df = df.isnull()
# 모든 column에 대해서 nan 값 출력.
for col in missing_df.columns:
    missing_count = missing_df[col].value_counts()    # 각 열의 NaN 개수 파악

    try: 
        print(col, ': ', missing_count[True])   # NaN 값이 있으면 개수를 출력
    except:
        print(col, ': ', 0)                     # NaN 값이 없으면 0개 출력
        
        
#%%

print(len(df.columns)) # 15


#%%
# axis = 0 인경우? 
#thresh옵션 : nan값이 1개이 이상인 열을 삭제
#결측값이 1개 이상 있는 raw 삭제 

df_dropped_rows = df.dropna(axis=0)
print(df_dropped_rows)

df_dropped_row2s = df.dropna()
print(df_dropped_row2s)

#%%
# Parameter
# axis : {0 or 'index', 1 or 'columns'}, default 0
# axis : 0('index', 'rows'), 1(columns)
# 기본값 : 0
df_dropped_rows3 = df.dropna(axis='rows')
print(df_dropped_rows3)

help(df.dropna)


# 만약에 NAN값이 여러개 있다면 thresh 에 할당한 값은 nan값의 개수보다 
# 같거나 많아야 한다

#한 행에서 최소한 1개의 유효한 값이 있어야 해당 행이 
#삭제되지 않음을 의미합니다. 따라서, 이 코드는 최소 1개의 유효한 값이 있는 
#모든 행을 유지하고 나머지는 삭제 


  
# NaN 값이 500개 이상인 열을 모두 삭제 - deck 열(891개 중 688개의 NaN 값)
# 500개 이상 NAN을 가지고 있는 컬럼은 삭제

df_thresh = df.dropna(axis=1, thresh=500)  
print(df_thresh.columns)

# age 열에 나이 데이터가 없는 모든 행을 삭제 - age 열(891개 중 177개의 NaN 값)
df_age = df.dropna(subset=['age'], how='any', axis=0)  
print(len(df_age))
# 714
#%%
'''
 how : {'any', 'all'}, default 'any'
     Determine if row or column is removed from DataFrame, when we have
     at least one NA or all NA.
 
     * 'any' : If any NA values are present, drop that row or column.
     * 'all' : If all values are NA, drop that row or column.
'''
# any : 1개 이상의 값이 na면
# all : 모든 값이 NA면

df_drop_any = df.dropna(how = 'any')  
# df_dropped_row2s = df.dropna() 와 결과가 같다 
# 즉 default 값으로 how=any 내재됨
# 182

# 모든 컬럼이 na이면 해당하는 행을 삭제
df_drop_all = df.dropna(how = 'all')
print(df_drop_all)

#%%
# thresh(hold) : 임계치 값
# how = 'any' = drop.na()
# axis = 0 행삭제 / axis = 1 열삭제

#%%
# 지정한 컬럼에 결측값이 있는 행만 삭제 
# 지정할 컬럼에서 하나라도 결측값이 있는 행이 삭제
del_cols = ['age', 'embarked']
df_droped_subs = df.dropna(subset = del_cols)
print(df_droped_subs)
 # subset : 제거할 때 고려할 열 또는 행의 부분집합을 지정하는 데 사용됩니다.
 # 따라서 subset옵션에는 []안에 삭제하고 싶은 열을 지정해주면 됨.
 
#%%
# 특정한 컬럼의 범위를 지정



del_cols = df.columns[3:8]
df_drop_subs = df.dropna(subset = del_cols)
print(df_drop_subs) # 714

# age~embarked 열까지 
df_drop_subs = df.dropna(subset=df.loc[:, 'age':'embarked'])






