#!/usr/bin/env python
# coding: utf-8

# pip install scikit-learn===1.1.3

#%%

# 회귀(Regression) - 보스턴 주택 가격 예측
# 적용모델
#   - 베이스라인 모델 - 선형회귀
#   - 과대적합 회피(L2/L1 규제)
#   - Ridge(L2 규제) 모델
#   - Lasso(L1 규제) 모델
#   - ElasticNet(L1, L2 규제) 모델
#   - 의사결정나무(DecisionTreeRegressor)
#   - 랜덤포레스트(RandomForestRegressor)
#   - 부스팅(XGBRegressor)

#%%
# # 라이브러리 설정

# # 데이터셋 불러오기

# In[1]:


# 기본 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# from sklearn import datasets

#%%

# skleran 데이터셋에서 보스턴 주택 데이터셋 로딩
# housing = datasets.load_boston()

# CSV 파일에서 데이터셋 로딩
data = pd.read_csv("./boston_housing.csv")
target = pd.read_csv("./boston_housing_target.csv")

# 딕셔너리 형태이므로, key 값을 확인
# housing.keys()


# In[2]:

# 판다스 데이터프레임으로 변환
# data = pd.DataFrame(housing['data'], columns=housing['feature_names'])
# target = pd.DataFrame(housing['target'], columns=['Target'])
# 데이터셋 크기
print(data.shape)
print(target.shape)
'''
(506, 13)
(506, 1)
'''

#%%
# CSV로 저장
# 인덱스는 제외
"""
data.to_csv("./boston_housing.csv", index=False)
target.to_csv("./boston_housing_target.csv", index=False)
"""

# 다시 읽기
"""
data1 = pd.read_csv("./boston_housing.csv")
target1 = pd.read_csv("./boston_housing_target.csv")
"""

# In[3]:


# 데이터프레임 결합하기
df = pd.concat([data, target], axis=1)
df.head(2)


# # 데이터 탐색 (EDA)

# In[4]:


# 데이터프레임의 기본정보
df.info()


# In[5]:


# 결측값 확인
df.isnull().sum()


# In[6]:


# 상관계수 행렬
df_corr = df.corr()

# 히트맵 그리기
plt.figure(figsize=(10, 10))
sns.set(font_scale=0.8)
sns.heatmap(df_corr, annot=True, cbar=False);
plt.show()


# In[7]:


# 변수 간의 상관관계 분석 - Target 변수와 상관관계가 높은 순서대로 정리
# Target의 절대값을 기준으로 내림차순 정렬
corr_order = df_corr.loc[:'LSTAT', 'Target'].abs().sort_values(ascending=False)
corr_order


# In[8]:


# Target 변수와 상관관계가 높은 4개 변수를 추출
plot_cols = ['Target', 'LSTAT', 'RM', 'PTRATIO', 'INDUS']
plot_df = df.loc[:, plot_cols]
plot_df.head()


# In[9]:

# regplot으로 선형회귀선 표시
plt.figure(figsize=(10,10))
for idx, col in enumerate(plot_cols[1:]):
    ax1 = plt.subplot(2, 2, idx+1) # 2행 2열의 그래프
    sns.regplot(x=col, y=plot_cols[0], data=plot_df, ax=ax1)    
plt.show()


# In[10]:


# Target 데이터의 분포
sns.displot( x='Target', kind='hist', data=df)
plt.show()


# # 데이터 전처리

# ### 피처 스케일링

# In[11]:


# 사이킷런 MinMaxScaler 적용 
# 정규화(Normalization) : 값의 크기를 비슷한 수준으로 조정
# 0~1사이 값으로 변환
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df_scaled = df.iloc[:, :-1] # Target 데이터 제외
#%%
scaler.fit(df_scaled)
df_scaled = scaler.transform(df_scaled)

#%%
# 스케일링 변환된 값을 데이터프레임에 반영
df.iloc[:, :-1] = df_scaled[:, :]
df.head()


# ### 학습용-테스트 데이터셋 분리하기

# In[12]:


# 학습 - 테스트 데이터셋 분할
from sklearn.model_selection import train_test_split
X_data = df.loc[:, ['LSTAT', 'RM']] # 저소득층비율, 거주목적의 방의 갯수
y_data = df.loc[:, 'Target'] # 정답
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                    test_size=0.2, # 테스트 데이터 20%
                                                    shuffle=True,  # 데이터를 섞음
                                                    random_state=12)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
'''
(404, 2) (404,)
(102, 2) (102,)
'''
## Baseline 모델 - 선형 회귀

# 선형 회귀 모형
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

print ("회귀계수(기울기): ", np.round(lr.coef_, 1))
print ("상수항(절편): ", np.round(lr.intercept_, 1))


# In[14]:


# 예측
y_test_pred = lr.predict(X_test)

# 예측값, 실제값의 분포
plt.figure(figsize=(10, 5))
plt.scatter(X_test['LSTAT'], y_test, label='y_test')  
plt.scatter(X_test['LSTAT'], y_test_pred, c='r', label='y_pred')
plt.legend(loc='best')
plt.show()


# In[15]:


# 평가
from sklearn.metrics import mean_squared_error
y_train_pred = lr.predict(X_train) # 훈련 데이터로 예측

# 훈련데이터의 정답과 훈련데이터로 예측한 결과 비교
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE: %.4f" % train_mse) # Train MSE: 30.8042

# 테스트데이터의 정답과 테스트데이터로 예측한 결과 비교
test_mse = mean_squared_error(y_test, y_test_pred) # Test MSE: 29.5065
print("Test MSE: %.4f" % test_mse)


# ## 교차 검증

# In[16]:


# cross_val_score 함수
from sklearn.model_selection import cross_val_score
lr = LinearRegression()
print("mse_scores:", cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
print("mse_scores:", cross_val_score(lr, X_train, y_train, cv=5))

# MSE를 음수로 계산하여 -1을 곱하여 양수로 변환
mse_scores = -1*cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("개별 Fold의 MSE: ", np.round(mse_scores, 4))
print("평균 MSE: %.4f" % np.mean(mse_scores))    


#%%

# 과대적합 회피
# # L1/L2 규제

# In[17]:


# 2차 다항식 변환
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2) # 2차
X_train_poly = pf.fit_transform(X_train)
print("원본 학습 데이터셋: ", X_train.shape)
print("2차 다항식 변환 데이터셋: ", X_train_poly.shape)
'''원본 학습 데이터셋:  (404, 2)
2차 다항식 변환 데이터셋:  (404, 6)'''

# In[18]:


# 2차 다항식 변환 데이터셋으로 선형 회귀 모형 학습
lr = LinearRegression()
lr.fit(X_train_poly, y_train)

#%%
# MSE 낮을 수록 예측력이 높아짐
#MSE는 실제 값과 예측 값 간의 오차를 제곱한 후,
# 이를 모든 데이터 포인트에 대해 평균화하여 계산됩니다
# 테스트 데이터에 대한 예측 및 평가
y_train_pred = lr.predict(X_train_poly)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE: %.4f" % train_mse) # Train MSE: 21.5463

X_test_poly = pf.fit_transform(X_test)
y_test_pred = lr.predict(X_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE: %.4f" % test_mse) # Test MSE: 16.7954
'''
Train MSE: 21.5463
Test MSE: 16.7954''' 

# In[19]:


# 15차 다항식 변환 데이터셋으로 선형 회귀 모형 학습
pf = PolynomialFeatures(degree=15)
X_train_poly = pf.fit_transform(X_train)

lr = LinearRegression()
lr.fit(X_train_poly, y_train)

# 테스트 데이터에 대한 예측 및 평가
y_train_pred = lr.predict(X_train_poly)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE: %.4f" % train_mse) # Train MSE: 11.2109

X_test_poly = pf.fit_transform(X_test)
y_test_pred = lr.predict(X_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE: %.4f" % test_mse) # Test MSE: 95441494600181.1250


# In[27]:


# 다항식 차수에 따른 모델 적합도 변화
plt.figure(figsize=(15,5))
for n, deg in enumerate([1, 2, 15]):
    ax1 = plt.subplot(1, 3, n+1)
    # degree별 다항 회귀 모형 적용
    pf = PolynomialFeatures(degree=deg)
    X_train_poly = pf.fit_transform(X_train.loc[:, ['LSTAT']])
    X_test_poly = pf.fit_transform(X_test.loc[:, ['LSTAT']])
    lr = LinearRegression()
    lr.fit(X_train_poly, y_train)
    y_test_pred = lr.predict(X_test_poly)
    # 실제값 분포
    plt.scatter(X_test.loc[:, ['LSTAT']], y_test, label='Targets') 
    # 예측값 분포
    plt.scatter(X_test.loc[:, ['LSTAT']], y_test_pred, label='Predictions') 
    # 제목 표시
    plt.title("Degree %d" % deg)
    # 범례 표시
    plt.legend()  
plt.show()


# In[21]:
#L2 규제는 모든 가중치를 조금씩 감소시키는 효과를 가지며, 
#각 가중치의 크기를 줄여서 모델의 복잡성을 감소시킵니다.

#어느 한 가중치 벡터(파라미터)를 크게 하지 않으면서, 
#전체적으로 잘 흩어지게 하는 효과가 있습니다.
# Ridge (L2 규제)
from sklearn.linear_model import Ridge
rdg = Ridge(alpha=2.5)
rdg.fit(X_train_poly, y_train)

y_train_pred = rdg.predict(X_train_poly)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE: %.4f" % train_mse)
y_test_pred = rdg.predict(X_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE: %.4f" % test_mse)
'''Train MSE: 35.9484
Test MSE: 42.0011
'''

# In[22]:
#L1 규제는 가중치의 절대값의 합을 제한하는 방식으로 작동합니다.
#L1 규제는 일부 가중치를 0으로 만들어 모델에서 특성 선택(feature selection)을 수행하므로
#특성의 희소성을 증가시키는 효과가 있습니다.
#lasso > ridge  (규제정도)
# Lasso (L1 규제)
from sklearn.linear_model import Lasso
las = Lasso(alpha=0.05)
las.fit(X_train_poly, y_train)

y_train_pred = las.predict(X_train_poly)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE: %.4f" % train_mse)
y_test_pred = las.predict(X_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE: %.4f" % test_mse)
'''Train MSE: 32.3204
Test MSE: 37.7103'''

# In[23]:


# ElasticNet (L2/L1 규제) 엘라스틱넷 회귀는 L2 규제와 L1 규제를 결합한 회귀
#엘라스틱넷은 Lasso 회귀가 서로 상관관계가 높은 feature들의 경우에 
#이들 중에서 중요 feature만을 선택하고 다른 feature들은 회귀 계수를 0으로 만드는 성향이 강하다

from sklearn.linear_model import ElasticNet
ela = ElasticNet(alpha=0.01, l1_ratio=0.7)
ela.fit(X_train_poly, y_train)

y_train_pred = ela.predict(X_train_poly)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE: %.4f" % train_mse)
y_test_pred = ela.predict(X_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE: %.4f" % test_mse)
'''
Train MSE: 33.7551
Test MSE: 39.4968'''

# # 트리 기반 모델 - 비선형 회귀

# In[24]:


# 의사결정 나무
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(max_depth=3, random_state=12)
dtr.fit(X_train, y_train)

y_train_pred = dtr.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE: %.4f" % train_mse)   # Train MSE: 18.8029
 
y_test_pred = dtr.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE: %.4f" % test_mse) # Test MSE: 17.9065


# In[25]:


# 랜덤 포레스트
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(max_depth=3, random_state=12)
rfr.fit(X_train, y_train)

y_train_pred = rfr.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE: %.4f" % train_mse)   # Train MSE: 16.0201

y_test_pred = rfr.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE: %.4f" % test_mse)# Test MSE: 17.7751


# In[26]:


# XGBoost
from xgboost import XGBRegressor
xgbr = XGBRegressor(objective='reg:squarederror', max_depth=3, random_state=12)
xgbr.fit(X_train, y_train)

y_train_pred = xgbr.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE: %.4f" % train_mse)   # Train MSE: 3.9261

y_test_pred = xgbr.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE: %.4f" % test_mse) # Test MSE: 19.9509

