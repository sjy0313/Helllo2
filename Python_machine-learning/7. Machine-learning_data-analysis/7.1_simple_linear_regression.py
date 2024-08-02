# -*- coding: utf-8 -*-

### 기본 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
[Step 1] 데이터 준비 - read_csv() 함수로 자동차 연비 데이터셋 가져오기
'''
# CSV 파일을 데이터프레임으로 변환
df = pd.read_csv('./auto-mpg.csv', header=None)

# 열 이름 지정
df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name'] 

# 데이터 살펴보기
print(df.head())   
print('\n')

#  IPython 디스플레이 설정 - 출력할 열의 개수 한도 늘리기
pd.set_option('display.max_columns', 10)
print(df.head())   
print('\n')

#%%
'''
[Step 2] 데이터 탐색
'''

# 데이터 자료형 확인
print(df.info())  
print('\n')

# 데이터 통계 요약정보 확인
print(round(df.describe(),2)) # 소숫점 2자리로 데이터 축소(round()활용)
print('\n')
# object인 horsepower 없음. 

# horsepower 열의 자료형 변경 (문자열 ->숫자)
horsepower = (df['horsepower'].unique())          # horsepower 열의 고유값 확인
print('\n')
print(len(horsepower),'건')  # 94 건
print('마력:' , sorted(horsepower)) # 오름차순으로 정렬

#%%

# df['horsepower'].replace('?', np.nan, inplace=True)      # '?'을 np.nan으로 변경
# 권고 코드 
df['horsepower'] = df['horsepower'].replace('?', np.nan)

df.dropna(subset=['horsepower'], axis=0, inplace=True)   # 누락데이터 행을 삭제


df['horsepower'] = df['horsepower'].astype('float')      # 문자열을 실수형으로 변환

print(df.describe())                                     # 데이터 통계 요약정보 확인
print('\n')


'''
[Step 3] 속성(feature 또는 variable) 선택
'''

# 분석에 활용할 열(속성)을 선택 (연비, 실린더, 출력, 중량)
ndf = df[['mpg', 'cylinders', 'horsepower', 'weight']]
print(ndf.head())   
print('\n')

### 종속 변수 Y인 "연비(mpg)"와 다른 변수 간의 선형관계를 그래프(산점도)로 확인
# Matplotlib으로 산점도 그리기
ndf.plot(kind='scatter', x='weight', y='mpg',  c='coral', s=10, figsize=(10, 5))
plt.show()
plt.close()

# seaborn으로 산점도 그리기
fig = plt.figure(figsize=(10, 5))   
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
sns.regplot(x='weight', y='mpg', data=ndf, ax=ax1)   #fit_reg=True가 default  # 회귀선 표시
sns.regplot(x='weight', y='mpg', data=ndf, ax=ax2, fit_reg=False)  #회귀선 미표시
plt.show()
plt.close()

# seaborn 조인트 그래프 - 산점도, 히스토그램
sns.jointplot(x='weight', y='mpg', data=ndf)              # 회귀선 없음
sns.jointplot(x='weight', y='mpg', kind='reg', data=ndf)  # 회귀선 표시
plt.show()
plt.close()

# seaborn pariplot으로 두 변수 간의 모든 경우의 수 그리기
sns.pairplot(ndf)  
plt.show()
plt.close()


'''
Step 4: 데이터셋 구분 - 훈련용(train data)/ 검증용(test data)
'''

# 속성(변수) 선택
X=ndf[['weight']]  #독립 변수 X
y=ndf['mpg']       #종속 변수 Y

# train data 와 test data로 구분(7:3 비율) 
# 훈련 데이터 70% / 검증용 데이터 30%  사용 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,               #독립 변수 
                                                    y,               #종속 변수
                                                    test_size=0.3,   #검증 30%
                                                    random_state=10) #랜덤 추출 값 

print('전체 data 개수: ', len(X))         # 392 
print('train data 개수: ', len(X_train))  # 274
print('test data 개수: ', len(X_test))    # 118


'''
Step 5: 단순회귀분석 모형 - sklearn 사용
'''

# sklearn 라이브러리에서 선형회귀분석 모듈 가져오기
from sklearn.linear_model import LinearRegression

# 단순회귀분석 모형 객체 생성
lr = LinearRegression()    # 모델 생성

# train data를 가지고 모형 학습(지도학습)
lr.fit(X_train, y_train) # 훈련 시킴
#%%
# 평가하기 : 
#모형의 적합도를 평가하는 다른 평가 지표인 결정계수(Coefficient of Determination)


# 학습을 마친 모형에 test data를 적용하여 결정계수(R-제곱) 계산
# R^2 = 1 - ( 타깃 - 예측)^2 / (타깃 - 평균)^2) 
# 0부터 1사이의 값
# 1에 가까워질수록 관계가 깊다  
# (설명력)R^2 = SSR / SST = 1 - SSE/SST
# SSR(Sum Of Regression) : 타깃 - 예측
# SST(Sum Of Squares Total) : SSE + SSR
# SSE(Sum of Squares Error) : 평균 - 예측
# 최소한 0.5보다 커야 의미있는 데이터임. 
#%%
# 학습을 마친 모형에 test data를 적용하여 결정계수(R-제곱) 계산
r_square = lr.score(X_test, y_test)
print(r_square) # 0.6822458558299325
print('\n')
# y의 분산을 68.2% 설명함. 
# 회귀식의 기울기
print('기울기 a: ', lr.coef_) # 기울기 a:  [-0.00775343]
print('\n')

# 회귀식의 y절편
print('y절편 b', lr.intercept_) # y절편 b 46.7103662572801
print('\n')
#%%
# 결과 :
    
# 모형에 전체 X 데이터를 입력하여 예측한 값 y_hat을 실제 값 y와 비교 
y_hat = lr.predict(X) # 예측데이터

plt.figure(figsize=(10, 5))
ax1 = sns.kdeplot(y, label="y")
ax2 = sns.kdeplot(y_hat, label="y_hat", ax=ax1)
plt.legend()
plt.show()
#%%

# MAE : 평균 절대 오차(mean absolute error)
# 실제값과 예측값 사이의 오차에 절대값을 씌움
# 값이 작을 수록 좋음, 0에 가까울수록 좋음
from sklearn.metrics import mean_absolute_error
MAE = mean_absolute_error(y, y_hat)
# 0에 가까울 수록 좋은 지표 
print(MAE) # 18.716383172421985

#%%
#  추측값에 대한 정확성을 측정하는 방법
#MSE : 평균 제곱 오차
# 실제값과 예측값 사이의 오차를 제곱한 후 평균으로 나눈 값
# 예측값과 실제값 차이의 면적의 평균과 같다
# 값이 적을수록 좋음, 0에 가까울 수록 좋음
# MSE = (test_value - predicted_value)^2 / len(test_value)
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y, y_hat)
print(MSE) # 18.716383172421985

#%%

# RMSE : 루트평균제곱오차
# MSE에 루트를 씌운 값
# 값이 작을수록 좋음, 0에 가까울 수록 좋음
RMSE = mean_squared_error(y, y_hat, squared=False)
print(RMSE) # 4.326243540581365

 # FutureWarning: 'squared' is deprecated in version 1.4 and will be removed i
 #n 1.6. To calculate the root mean squared error, use the function'root_mean_squared_error'.
  # warnings.warn(
#%%

from sklearn.metrics import root_mean_squared_error 
RMSE2 = root_mean_squared_error(y, y_hat)
print(RMSE2) #  4.326243540581365
       

















