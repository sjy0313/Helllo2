# -*- coding: utf-8 -*-

import pandas as pd

# DataFrame() 함수로 데이터프레임 변환. 변수 df에 저장 
exam_data = {'이름' : ['서준', '우현', '인아'],
             '수학' : [ 90, 80, 70],
             '영어' : [ 98, 89, 95],
             '음악' : [ 85, 95, 100],
             '체육' : [ 100, 90, 90]}
df = pd.DataFrame(exam_data)
print(df)
print('\n')

# 새로운 행(row)을 추가 - 같은 원소 값을 입력
df.loc[3] = 0
print(df)
print('\n')
'''
 이름  수학  영어   음악   체육
0  서준  90  98   85  100
1  우현  80  89   95   90
2  인아  70  95  100   90
3   0   0   0    0    0
'''
# 새로운 행(row)을 추가 - 원소 값 여러 개의 배열 입력
df.loc[4] = ['동규', 90, 80, 70, 60]
print(df)
print('\n')
'''
 이름  수학  영어   음악   체육
0  서준  90  98   85  100
1  우현  80  89   95   90
2  인아  70  95  100   90
3   0   0   0    0    0
4  동규  90  80   70   60'''
# 새로운 행(row)을 추가 - 기존 행을 복사
df.loc['행5'] = df.loc[3]
print(df)
'''
    이름  수학  영어   음악   체육
0   서준  90  98   85  100
1   우현  80  89   95   90
2   인아  70  95  100   90
3    0   0   0    0    0
4   동규  90  80   70   60
행5   0   0   0    0    0'''
#%%
# iloc : 기존 행은 변경
df.iloc[5] = ['길동', 90, 80, 70, 60]
df
'''
이름  수학  영어   음악   체육
0   서준  90  98   85  100
1   우현  80  89   95   90
2   인아  70  95  100   90
3    0   0   0    0    0
4   동규  90  80   70   60
행5  길동  90  80   70   60
'''
#%%
#df.iloc[6] = ['우치', 90, 80, 70, 60]
print(df) # IndexError: iloc cannot enlarge its target object


# 만약 행을 추가 할 떄 loc 를 쓰지 않으면, 아래와 같은 step을 밟아야함.
s1 = pd.Series(['우치', 90, 80, 70, 60])
# dict로 변환
s3 = s1.to_dict()
# value 값들로 list 만들어주고,
s4 = list(s3.values())
# column 이름 동일하게 해준 뒤 concat
s2 = pd.DataFrame([s4], columns=['이름', '수학', '영어', '음악', '체육'])
df = pd.concat([df,s2], axis=0, ignore_index =True)




