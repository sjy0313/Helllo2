# -*- coding: utf-8 -*-

import pandas as pd

# 리스트를 시리즈로 변환하여 변수 sr에 저장
list_data = ['2019-01-02', 3.14, 'ABC', 100, True]
sr = pd.Series(list_data)
print(sr)
print('\n')

# 인덱스 배열은 변수 idx에 저장. 데이터 값 배열은 변수 val에 저장
idx = sr.index # 인덱스 값 
val = sr.values # value 값
print(idx) # 인덱스 값의 범위
print('\n')
print(val) # numpy 배열

#%%
# 데이터 프레임과의 호환성을 위해 dtypes 사용
list_data = [0.123,  3.14]
sr = pd.Series(list_data)
print(sr.dtypes) # float64 실수형 자료형 



