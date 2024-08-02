# -*- coding: utf-8 -*-

# 라이브러리 불러오기
import seaborn as sns

# titanic 데이터셋 가져오기
df = sns.load_dataset('titanic')

# embark_town 열의 829행의 NaN 데이터 출력
print(df['embark_town'][825:830])
print('\n')
'''
825     Queenstown
826    Southampton
827      Cherbourg
828     Queenstown
829            NaN
Name: embark_town, dtype: object


Southampton
'''
# embark_town 열의 NaN값을 승선도시 중에서 가장 많이 출현한 값으로 치환하기
most_freq = df['embark_town'].value_counts(dropna=True).idxmax()   
print(df['embark_town'].value_counts(dropna=False))
'''
embark_town
Southampton    646
Cherbourg      168
Queenstown      77
Name: count, dtype: int64
'''
print(most_freq) # Southampton
print('\n')
# NAN을 제외하면 Queenstown이 77건으로 가장 작은 열의 값


missing_value_embark = df['embark_town'].isnull()

import pandas as pd
missing_df = pd.DataFrame(missing_value_embark)

col ='embark_town'
missing_count = missing_df.value_counts() # 2개의 nan값 존재



df['embark_town'].fillna(most_freq, inplace=True)

# embark_town 열 829행의 NaN 데이터 출력 (NaN 값이 most_freq 값으로 대체)
print(df['embark_town'][825:830])
'''
825     Queenstown
826    Southampton
827      Cherbourg
828     Queenstown
829    Southampton
Name: embark_town, dtype: object
'''