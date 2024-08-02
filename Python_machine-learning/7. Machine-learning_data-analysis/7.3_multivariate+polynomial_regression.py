# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 16:28:26 2024

@author: Shin
"""
# 다중+다항회귀 분석
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 다중회귀 분석(Multiv
df = pd.read_csv('./auto-mpg.csv', header=None)

# 열 이름 지정
df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name'] 

# horsepower 열의 자료형 변경 (문자열 ->숫자)
df['horsepower'] = df['horsepower'].replace('?', np.nan)   # '?'을 np.nan으로 변경
df.dropna(subset=['horsepower'], axis=0, inplace=True)   # 누락데이터 행을 삭제
df['horsepower'] = df['horsepower'].astype('float')      # 문자열을 실수형으로 변환

# 분석에 활용할 열(속성)을 선택 (연비, 실린더, 출력, 중량)
ndf = df[['mpg', 'cylinders', 'horsepower', 'weight']]


'''
Step 4: 데이터셋 구분 - 훈련용(train data)/ 검증용(test data)
'''

# 속성(변수) 선택
X=ndf[['cylinders', 'horsepower', 'weight']]  #독립 변수 X1, X2, X3
y=ndf['mpg']     #종속 변수 Y


# train data 와 test data로 구분(7:3 비율)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10) 

print('훈련 데이터: ', X_train.shape)
print('검증 데이터: ', X_test.shape)   
print('\n')
'''
훈련 데이터: (274, 3)
검증 데이터:  (118, 3)
'''
#%%
# Step 5: 다중회귀분석 모형 - sklearn 사용

# sklearn 라이브러리에서 필요한 모듈 가져오기 
from sklearn.linear_model import LinearRegression      #선형회귀분석
from sklearn.preprocessing import PolynomialFeatures   #다항식 변환

# 다항식 변환 
poly = PolynomialFeatures(degree=3)               #3차항 적용
X_train_poly=poly.fit_transform(X_train)     #X_train 데이터를 3차항으로 변형
X_test_poly = poly.fit_transform(X_test)  

print('원 데이터: ', X_train.shape) # 원 데이터:  (274, 3)
print('3차항 변환 데이터: ', X_train_poly.shape)   # 3차항 변환 데이터:  (274, 20) 

#%%
# train data를 가지고 모형 학습
lr = LinearRegression()   
lr.fit(X_train_poly, y_train) # 훈련

# fit 과 transform 분리
# fit : 데이터를 학습
# transform : 학습에서 얻은 정보로 계산 
X_test_poly = poly.fit_transform(X_test)       #X_test 데이터를 3차항으로 변형
r_square = lr.score(X_test_poly,y_test)
print(r_square) # 0.7744507210485588
print('\n')

#%%

# train data의 산점도와 test data로 예측한 회귀선을 그래프로 출력 
y_hat_test = lr.predict(X_test_poly)

#%%
# 회귀식의 기울기 (머신러닝에서 기울기를 coefficient or weight(가중치))
print('X 변수의 계수 a: ', lr.coef_)  

# 회귀식의 y절편
print('상수항 b', lr.intercept_) 
#%%
# train data의 산점도와 test data로 예측한 회귀선을 그래프로 출력 
# y_hat = lr.predict(X_test)
y_hat = lr.predict(X_test_poly)

plt.figure(figsize=(10, 5))
ax1 = sns.kdeplot(y_test, label="y_test")
ax2 = sns.kdeplot(y_hat, label="y_hat", ax=ax1)
plt.legend()
plt.show()
'''
회귀분석에서 "기울기"는 독립 변수와 종속 변수 간의 관계를 설명하는 매개 변수입니다. 
이를 "가중치(weight)"라고도 합니다. 기울기는 독립 변수의 변화량이 종속 변수에 미치는
 영향을 나타내는데, 이 영향은 해당 독립 변수의 가중치에 따라 결정됩니다. 즉,
 독립 변수가 종속 변수에 미치는 영향의 크기와 방향을 결정하는 것이 가중치입니다. 
 따라서 회귀분석에서 기울기와 가중치는 동일한 의미로 사용됩니다.
'''








