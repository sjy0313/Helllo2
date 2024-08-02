# -*- coding: utf-8 -*-

import pandas as pd

# DataFrame() 함수로 데이터프레임 변환. 변수 df에 저장 
exam_data = {'이름' : [ '서준', '우현', '인아'],
             '수학' : [ 90, 80, 70],
             '영어' : [ 98, 89, 95],
             '음악' : [ 85, 95, 100],
             '체육' : [ 100, 90, 90]}
df = pd.DataFrame(exam_data)
print(df)
'''
이름  수학  영어   음악   체육
0  서준  90  98   85  100
1  우현  80  89   95   90
2  인아  70  95  100   90
'''
print(type(df)) # <class 'pandas.core.frame.DataFrame'>
#%%

# '수학' 점수 데이터만 선택. 변수 math1에 저장
# 1개의 칼럼 선택되면 모든 행의 series객체 리턴
math1 = df['수학']
print(math1)
'''
0    90
1    80
2    70
Name: 수학, dtype: int64
'''
print(type(math1)) # <class 'pandas.core.series.Series'>
print('\n')

# '영어' 점수 데이터만 선택. 변수 english에 저장
# ''안싸줘도 출력됨. 
english = df.영어
print(english)
'''
0    98
1    89
2    95
Name: 영어, dtype: int64'
'''
print(type(english))
print('\n')

# '음악', '체육' 점수 데이터를 선택. 변수 music_gym 에 저장
music_gym = df[['음악', '체육']]
print(music_gym)
'''
    음악   체육
0   85  100
1   95   90
2  100   90
'''
print(type(music_gym))
# <class 'pandas.core.frame.DataFrame'>
print('\n')

# '수학' 점수 데이터만 선택. 변수 math2에 저장
math2 = df[['수학']]
print(math2)
'''
  수학
0  90
1  80
2  70
'''
# 한번 더 싸줘서 df로 전환
print(type(math2)) # <class 'pandas.core.frame.DataFrame'>
