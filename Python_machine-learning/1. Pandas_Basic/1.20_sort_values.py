# -*- coding: utf-8 -*-

import pandas as pd

# 딕셔서리를 정의
dict_data = {'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9], 'c3':[10,11,12], 'c4':[13,14,15]}

# 딕셔서리를 데이터프레임으로 변환. 인덱스를 [r0, r1, r2]로 지정
df = pd.DataFrame(dict_data, index=['r0', 'r1', 'r2'])
print(df)
print('\n')
''' c0  c1  c2  c3  c4
r0   1   4   7  10  13
r1   2   5   8  11  14
r2   3   6   9  12  15
'''
# c1 열을 기준으로 내림차순 정렬 
ndf = df.sort_values(by='c1', ascending=False)
print(ndf)
#%%

''' c0  c1  c2  c3  c4
r2   3   6   9  12  15
r1   2   5   8  11  14
r0   1   4   7  10  13
'''
#%%
sdf = df.sort_values(by='c1', ascending=True)
print(sdf)
'''
    c0  c1  c2  c3  c4
r0   1   4   7  10  13
r1   2   5   8  11  14
r2   3   6   9  12  15
'''
#%% 
# 새로운 행 추가
df.loc['r3'] = [2,5,7,9,10]
print(df)
''' c0  c1  c2  c3  c4
r0   1   4   7  10  13
r1   2   5   8  11  14
r2   3   6   9  12  15
r3   2   5   7   9  10
'''
#%%
# 여러 컬럼을 정렬 기준으로 지정
# 컬럼을 리스트 목록으로 지정
# ascending : True, 기본값
mdf = df.sort_values(by=['c2', 'c4'])
print(mdf)
#  DataFrame df를 'c2'를 우선으로 오름차순으로 정렬하고, 
#그 후에 'c4'를 기준으로 같은 값들을 오름차순으로 정렬하여 mdf에 할당하는 것을 의미
'''
    c0  c1  c2  c3  c4
r0   1   4   7  10  13
r1   2   5   8  11  14
r2   3   6   9  12  15
r3   2   5   7   9  10
'''
