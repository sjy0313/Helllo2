# -*- coding: utf-8 -*-

import pandas as pd

# 딕셔서리를 정의
dict_data = {'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9], 'c3':[10,11,12], 'c4':[13,14,15]}

# 딕셔서리를 데이터프레임으로 변환. 인덱스를 [r0, r1, r2]로 지정
df = pd.DataFrame(dict_data, index=['r0', 'r1', 'r2'])
print(df)
'''
   c0  c1  c2  c3  c4
r0   1   4   7  10  13
r1   2   5   8  11  14
r2   3   6   9  12  15
'''
#%%
ren_index = ['x0', 'x1', 'x2', 'x3']
# ValueError: Length mismatch: Expected axis has 3 elements, new values have 4 elements
df.index = ren_index
# 새로운 인덱스를 지정할 떄 개수가 일치하지 않으면 value error 발생
print(df.index) 
# 개수 맞춰서 지정 필요
ren_index = ['x0', 'x1', 'x2']
df.index = ren_index
print(df.index) 
#%%
# reindex() : dataframe의 행 인덱스를 새로운 배열로 재지정 가능. 
# 인덱스의 개수가 일피하지 않아도 됨
# 지정한 인덱스만 선택되는 효과'
# 기존 프레임을 변경
new_index = ['r0', 'r1']
ndf = df.reindex(new_index)
print(ndf)
'''
    c0  c1  c2  c3  c4
r0   1   4   7  10  13
r1   2   5   8  11  14
'''
#%%
# 인덱스를 [r0, r1, r2, r3, r4]로 재지정
new_index = ['r0', 'r1', 'r2', 'r3', 'r4']
ndf = df.reindex(new_index)
print(ndf)
'''   c0   c1   c2    c3    c4
r0  1.0  4.0  7.0  10.0  13.0
r1  2.0  5.0  8.0  11.0  14.0
r2  3.0  6.0  9.0  12.0  15.0
r3  NaN  NaN  NaN   NaN   NaN
r4  NaN  NaN  NaN   NaN   NaN
'''
print('\n')

# reindex로 발생한 NaN값을 숫자 0으로 채우기
new_index = ['r0', 'r1', 'r2', 'r3', 'r4']
ndf2 = df.reindex(new_index, fill_value=0)
print(ndf2)
'''
c0  c1  c2  c3  c4
r0   1   4   7  10  13
r1   2   5   8  11  14
r2   3   6   9  12  15
r3   0   0   0   0   0
r4   0   0   0   0   0'''
ndf2 = df.reindex(new_index, fill_value=1)
print(ndf2)
'''
    c0  c1  c2  c3  c4
r0   1   4   7  10  13
r1   2   5   8  11  14
r2   3   6   9  12  15
r3   1   1   1   1   1
r4   1   1   1   1   1
'''
