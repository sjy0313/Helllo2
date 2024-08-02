# -*- coding: utf-8 -*-

# 라이브러리 불러오기
import pandas as pd

# read_csv() 함수로 파일 읽어와서 df로 변환
df = pd.read_csv('stock-data.csv')

# 문자열인 날짜 데이터를 판다스 Timestamp로 변환
df['new_Date'] = pd.to_datetime(df['Date'])   # 새로운 열에 추가
df.set_index('new_Date', inplace=True)        # 행 인덱스로 지정
#%%

print(df.head())
print('\n')
print(df.index)
print('\n')

# 날짜 인덱스를 이용하여 데이터 선택하기
# 문제점 : 인덱스가 날짜인 경우 슬라이싱 방법?????


df_y = df.loc['2018']# 전체
df_ym = df.loc['2018-06']  # 18년 6월 데이터

df_ym_cols = df.loc['2018-07', 'Start':'High']    # 열 범위 슬라이싱
# (행인덱스 + 열인덱스의 범위) 
print(df_ym_cols)

# 데이터프레임으로 변환
df_ym = df.loc[['2018-06']]


df_ymd = df.loc[['2018-07-02']] 
print(df_ymd)

#%%
# 판다스에서 datetime으로 변경 후 참조 가능
df_y1 = pd.to_datetime('2018-06-25')
df_y2 = pd.to_datetime('2018-06-20')
print(df_y1, df_y2)
# 2018-06-25 00:00:00 2018-06-20 00:00:00

df_yy = df.loc[df_y1:df_y2, 'Close':'Volume']
print(df_yy)
#%%
# 인덱스를 정렬 / 오름차순 정렬
df_in = df.sort_index()

#  오름차순 정렬 시 빠른 날짜를 우선으로 슬라이싱 첫번 째 값으로 지정
ddf_ymf_range = df_in.loc[pd.to_datetime('2018-06-20'):pd.to_datetime('2018-06-25')]
print(ddf_ymf_range)


df_index = pd.date_range(df_y1,df_y2)
print(df_index)
'''
2018-06-25 00:00:00 2018-06-20 00:00:00
            Close  Start   High    Low  Volume
new_Date                                      
2018-06-25  11150  11400  11450  11000   55519
2018-06-22  11300  11250  11450  10750  134805
2018-06-21  11200  11350  11750  11200  133002
2018-06-20  11550  11200  11600  10900  308596
DatetimeIndex([], dtype='datetime64[ns]', freq='D')'''



sort_June = df_index.sort_index(ascending=False, inplace =True)


# 내림차순
help(df.sort_index)

df_in1 = df.sort_index(ascending=False, inplace= True)


# KeyError: 'Value based partial slicing on non-monotonic DatetimeIndexes
# with non-existing keys is not allowed.'
df_ymd_range = df['2018-06-25':'2018-06-20']

# KeyError: '2018'
df_y = df['2018']

# KeyError: 'new_Date'
print(type(df['new_Date'].dt))

# KeyError: 'new_Date'
values = df['new_Date'].dt
print(values)

 # KeyError: '2018-07-02'
df_ymd = df['2018-07-02'] 
print(df_ymd)

print('\n')
# KeyError: 'Value based partial slicing on non-monotonic 
# DatetimeIndexes with non-existing keys is not allowed.'
df_ymd_range = df['2018-06-25':'2018-06-20']    # 날짜 범위 지정
print(df_ymd_range)
print('\n')
#%%
# key error 발생방지 -> 정책이 바뀜에 따라
# 각각의 index 지정
df_y1 = pd.to_datetime('2018-06-25')
df_y2 = pd.to_datetime('2018-06-20')
print(df_y1, df_y2)
# 2018-06-25 00:00:00 2018-06-20 00:00:00

df_yy = df.loc[df_y1:df_y2, 'Close':'Volume']
print(df_yy)

#%%

# 시간 간격 계산. 최근 180일 ~ 189일 사이의 값들만 선택하기
today = pd.to_datetime('2018-12-25')            # 기준일 생성
df['time_delta'] = today - df.index             # 날짜 차이 계산
df.set_index('time_delta', inplace=True)        # 행 인덱스로 지정
df_180 = df['180 days':'189 days']
print(df_180)

#%%
dt_index = df.index
print(type(dt_index))
# <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
# 내림차순

ddf = df.sort_index(ascending=False)

ddf_ymf_range = ddf.loc[pd.to_datetime('2018-06-25'):pd.to_datetime('2018-06-20')]
print(ddf_ymf_range)


help(df.sort_index)

df.sort_index(ascending=False, inplace= True)
df_ymd_range = df['2018-06-25':'2018-06-20']





















