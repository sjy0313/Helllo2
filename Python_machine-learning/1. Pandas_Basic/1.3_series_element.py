# -*- coding: utf-8 -*-

import pandas as pd

# 투플을 시리즈로 변환(index 옵션에 인덱스 이름을 지정)
tup_data = ('영인', '2010-05-01', '여', True)
sr = pd.Series(tup_data, index=['이름', '생년월일', '성별', '학생여부'])
# 포지션 index를 쓰고 싶다면 print(sr['이름']) or sr.iloc[0] 활용
'''FutureWarning: Series.__getitem__ treating keys as positions is deprecated. 
In a future version, integer keys will always be treated as labels
(consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
print(sr[[1, 2]])'''
  



print(sr)
'''이름              영인
생년월일    2010-05-01
성별               여
학생여부          True
dtype: object
'''
print('\n')
#%%%
# 원소를 1개 선택
print(sr[0]) # 영인 (바람직하지 않음) # sr의 1 번째 원소를 선택 (정수형 위치 인덱스를 활용)

#%%%
print(sr['이름']) # 영인 (labeling 활용) # '이름' 라벨을 가진 원소를 선택 (인덱스 이름을 활용)
print('\n')
#%%
# 적절한 방법
# loc : location / iloc : integer location
# dataframe에서 loc 활용 -> loc[행, 칼럼명] / iloc[row_index, column_index] 숫자기반으로 데이터 찾을 수 있음
#[deprecated]  :앞으로 지원되지 않을것이므로 사용을 자제 해달라는 의미
#[obsolete] : 더 이상 쓸모가 없는, 한물간, 

print(sr.loc['이름']) # 영인
# 여러 개의 원소를 선택 (인덱스 리스트 활용)
print(sr[[1, 2]]) 
'''
생년월일    2010-05-01
성별               여
'''
print('\n')           
print(sr[['생년월일', '성별']])
'''
생년월일    2010-05-01
성별               여
'''
print('\n')

# 여러 개의 원소를 선택 (인덱스 범위 지정)
print(sr[1 : 2]) # (X)권고 하지않음.
sr.iloc[0] # '영인'
# 아래와 같이 만약 정수 값 1개를 지정하면 해당 series의 첫번 쨰 값인 '영인' 출력
# 2개를 선택하면 
print(sr)
'''이름              영인
생년월일    2010-05-01
성별               여
학생여부          True
dtype: object
'''
df = pd.DataFrame(sr)
df1 = df.iloc[1:2] # dataframe으로 출력
'''
     0
생년월일  2010-05-01
'''
df2 = sr.iloc[1:2] # 2번쨰 행의 값 출력 -> series 출력
'''
생년월일    2010-05-01
dtype: object
'''
# 생년월일    2010-05-01


# sr.iloc[1:2] (O)권고 [iloc : dataframe뿐만 아니라 series 에서도 사용가능]
print('\n')              
print(sr['생년월일' : '성별'])
'''
생년월일    2010-05-01
성별               여
dtype: object
'''
