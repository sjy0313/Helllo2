# -*- coding: utf-8 -*-

import pandas as pd

df = pd.read_excel('./남북한발전전력량.xlsx', engine='openpyxl')  # 데이터프레임 변환 

df_ns = df.iloc[[0, 5], 3:]            # 남한, 북한 발전량 합계 데이터만 추출
df_ns.index = ['South','North']        # 행 인덱스 변경
df_ns.columns = df_ns.columns.map(int) # 열 이름의 자료형을 정수형으로 변경

print(df_ns.values)
print(df_ns.values.dtype) # object
df_ns.info() # -> 정수가 아닌 객체로 출력됨 
# 행, 열 전치하여 히스토그램 그리기
tdf_ns = df_ns.T


#자료형 변환 (object -> int)
tdf_ns = tdf_ns.astype(int)
print(tdf_ns.values.astype(int))
tdf_ns.info()
'''
<class 'pandas.core.frame.DataFrame'>
Index: 26 entries, 1991 to 2016
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   South   26 non-null     int32
 1   North   26 non-null     int32
dtypes: int32(2)
'''


tdf_ns.plot(kind='hist')