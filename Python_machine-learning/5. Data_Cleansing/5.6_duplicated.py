# -*- coding: utf-8 -*-

# 라이브러리 불러오기
import pandas as pd

# 중복 데이터를 갖는 데이터프레임 만들기
df = pd.DataFrame({'c1':['a', 'a', 'b', 'a', 'b'],
                  'c2':[1, 1, 1, 2, 2],
                  'c3':[1, 1, 2, 2, 2]})
print(df)
print('\n')
'''
 c1  c2  c3
0  a   1   1
1  a   1   1
2  b   1   2
3  a   2   2
4  b   2   2
'''
# 데이터프레임 전체 행 데이터 중에서 중복값 찾기
# 1행과 0행의 중복 -> 1 True 값 반환
df_dup = df.duplicated()
print(df_dup)
'''
0    False
1     True
2    False
3    False
4    False
dtype: bool
'''
print('\n')

# 데이터프레임의 특정 열 데이터에서 중복값 찾기
col_dup = df['c2'].duplicated()
print(col_dup)
'''
0    False
1     True
2     True
3    False
4     True
Name: c2, dtype: bool'''