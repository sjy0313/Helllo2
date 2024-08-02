# -*- coding: utf-8 -*-

import pandas as pd

# read_csv() 함수로 df 생성
df = pd.read_csv('./auto-mpg.csv', header=None)

# 열 이름을 지정
df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

# 평균값 
print(df.mean())
 # TypeError: Could not convert (horsepower에 대한 평균 결측치떄매 못구함)
 
#%%
# horsepower를 제외한 나머지 요소들에 대한 평균
# integer가 아닌 name객체도 제외
ncols = ['mpg','cylinders','displacement','weight',
              'acceleration','model year','origin']
print(df[ncols].mean())
'''
mpg               23.514573
cylinders          5.454774
displacement     193.425879
weight          2970.424623
acceleration      15.568090
model year        76.010050
origin             1.572864
dtype: float64
'''
# 평균값 : 'weight' 
print(df['weight'].mean())
# 2970.424623115578

print('\n')
print(df['mpg'].mean())
print(df.mpg.mean())
print('\n')
print(df[['mpg','weight']].mean())
'''
mpg         23.514573
weight    2970.424623
dtype: float64'''
# 중간값 
print(df.median())
print('\n')
print(df['mpg'].median()) # 23.0

# 최대값 
print(df.max())
print('\n')
print(df['mpg'].max()) # 46.6

# 최소값 
print(df.min())
print('\n')
print(df['mpg'].min())

# 표준편차 
print(df.std())
print('\n')
print(df['mpg'].std()) # 7.815984312565782 

# 상관계수 correlation coefficient
# 절댓값 1에 가까울 수록 상관관계를 갖는다.
# -1 :음의 선형 관계
# +1 :음의 선형 관계
# 0 : 선형 관계가 없다


print(df[ncols].corr())
'''
                   mpg  cylinders  ...  model year    origin
mpg           1.000000  -0.775396  ...    0.579267  0.563450
cylinders    -0.775396   1.000000  ...   -0.348746 -0.562543
displacement -0.804203   0.950721  ...   -0.370164 -0.609409
weight       -0.831741   0.896017  ...   -0.306564 -0.581024
acceleration  0.420289  -0.505419  ...    0.288137  0.205873
model year    0.579267  -0.348746  ...    1.000000  0.180662
origin        0.563450  -0.562543  ...    0.180662  1.000000

[7 rows x 7 columns]'''

# 연비와 무게는 상관관계가 높다
print(df.corr())
print('\n')
print(df[['mpg','weight']].corr())
'''
            mpg    weight
mpg     1.000000 -0.831741
weight -0.831741  1.000000'''


