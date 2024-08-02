# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:06:46 2024

@author: Shin
"""
# 난수 발생
import pandas as pd
import numpy as np

start = 0
stop = 10
num = 5
data = np.random.randint(start, stop, num) # 0~9까지 5개
# 시작과 끝의 범위를 지정 즉 0~9까지 10개의 수중에 5개 random으로 뽑기
index = list("abcde")
series = pd.Series(data=data,
                   index=index,
                   name="Seires from ndarray") # 배열로 부터 출력된 series값
print(series)
'''
a    3
b    7
c    5
d    7
e    5
Name: Seires from ndarray, dtype: int32
'''


import pandas as pd
import numpy as np
# 중복되지 않는 값을 출력하려면
# numpy.random.choice() 함수사용  + replace 매개변수를 사용하여 중복선택 금지
start = 0
stop = 10
num = 5
data = np.random.choice(np.arange(start, stop), num, replace=False)
index = list("abcde")
series1 = pd.Series(data=data,
                   index=index,
                   name="Seires from ndarray") # 배열로 부터 출력된 series값
print(series1)
#%%
help(np.random.randint)
#%%

print(series.name) # Seires from ndarray

print(series.values) # [3 7 5 7 5]

print(series.index) # Index(['a', 'b', 'c', 'd', 'e'], dtype='object')

print(type(series.values)) # <class 'numpy.ndarray'> 
#%%
# 리스트로 전환
series_list = series.tolist() 
print(type(series_list),series_list) # <class 'list'> [3, 7, 5, 7, 5]
