# -*- coding: utf-8 -*-

### 기본 라이브러리 불러오기
import pandas as pd
import seaborn as sns



# 분류(classification)
# 목표변수가 갖고 있는 카테고리(범주형) 값 중에서 분류 예측
# 고객분류/질병진단/스펨매일 필터링/음성인식 등에 쓰임
# KNN, SVM, Decision, Tree, Logistic Regression

# 타이타닉 승객들의 생존여부(survived) 예측


'''
[Step 1] 데이터 준비 - Seaborn에서 제공하는 titanic 데이터셋 가져오기
'''

# load_dataset 함수를 사용하여 데이터프레임으로 변환
df = sns.load_dataset('titanic')

# 데이터 살펴보기
print(df.head())   
print('\n')

#  IPython 디스플레이 설정 - 출력할 열의 개수 한도 늘리기
pd.set_option('display.max_columns', 15)
print(df.head())   
print('\n')


'''
[Step 2] 데이터 탐색
'''

# 데이터 자료형 확인
print(df.info())  
print('\n')
# drop() 지정된 행/열 삭제 / dropna() nan값을 포함한 행/열 삭제
# NaN값이 많은 deck 열을 삭제, embarked와 내용이 겹치는 embark_town 열을 삭제
rdf = df.drop(['deck', 'embark_town'], axis=1)  
print(rdf.columns.values) #  열 index값 확인
'''['survived' 'pclass' 'sex' 'age' 'sibsp' 'parch' 'fare' 'embarked' 'class'
 'who' 'adult_male' 'alive' 'alone']'''
print('\n')

# age 열에 나이 데이터가 없는 모든 행을 삭제 - age 열(891개 중 177개의 NaN 값)
rdf = rdf.dropna(subset=['age'], how='any', axis=0)  
print(len(rdf))  # 714
# how='any'는 한 행 또는 열에 하나 이상의 NaN 값이 있을 경우 해당 행 또는 열을 삭제합니다. 
# 따라서 'age' 열에서 NaN 값이 하나라도 있는 행은 모두 삭제됩니다.
print('\n')

# embarked 열의 NaN값을 승선도시 중에서 가장 많이 출현한 값으로 치환하기
most_freq = rdf['embarked'].value_counts(dropna=True).idxmax()   
print(most_freq) # S (S와 C중 S가 가장 많이 출현)
print('\n')

print(rdf.describe(include='all'))
print('\n')
#컬럼('embarked')의 누락데이터는 가장 많은 빈도수를 가진 'S'로 채움. 
#rdf['embarked'].fillna(most_freq, inplace=True)
rdf['embarked'] = rdf['embarked'].fillna(most_freq)

rdf['embarked'].value_counts()
'''
embarked
S    556
C    130
Q     28
Name: count, dtype: int64
'''


'''
[Step 3] 분석에 사용할 속성을 선택
'''

# 분석에 활용할 열(속성)을 선택 
ndf = rdf[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked']]
print(ndf.head())   
print('\n')

# 원핫인코딩 - 범주형 데이터를 모형이 인식할 수 있도록 숫자형으로 변환
onehot_sex = pd.get_dummies(ndf['sex']) # male/female
#  이 함수는 주어진 데이터프레임의 범주형 변수를 새로운 이진(dummy) 변수(0 또는 1)로 변환

ndf = pd.concat([ndf, onehot_sex], axis=1) # 열추가

onehot_embarked = pd.get_dummies(ndf['embarked'], prefix='town') # 접두사 town이 추가 
ndf = pd.concat([ndf, onehot_embarked], axis=1)

ndf.drop(['sex', 'embarked'], axis=1, inplace=True)
print(ndf.head())   
print('\n')


'''
[Step 4] 데이터셋 구분 - 훈련용(train data)/ 검증용(test data)
'''

# 속성(변수) 선택
X=ndf[['pclass', 'age', 'sibsp', 'parch', 'female', 'male', 
       'town_C', 'town_Q', 'town_S']]  #독립 변수 X
y=ndf['survived']                      #종속 변수 Y

#%%
# 설명 변수 데이터를 정규화(normalization) [표준정규분포]로 만들어주는 과정
# 표준정규분포 = {X- 평균(x)}  /  표준편차(x)
from sklearn import preprocessing

X0 = preprocessing.StandardScaler().fit(X)
# preprocessing.StandardScaler()는 데이터의 평균을 0으로, 표준편차를 1로 만들어 
# 데이터의 스케일을 조정하는 데 사용되는 스케일러

X = preprocessing.StandardScaler().fit(X).transform(X)
#이 코드는 데이터를 표준화(Standardization)하는 과정을 수행합니다. 
#표준화는 데이터의 평균을 0으로, 표준편차를 1로 만들어 데이터의 분포를 조정하는 작업

#preprocessing.StandardScaler(): StandardScaler() 객체를 생성합니다.
# 이 객체는 데이터를 표준화

#fit(X): 주어진 데이터 X에 대해 평균과 표준편차를 계산하여 스케일러를 적합시킵니다.
# train data 와 test data로 구분(7:3 비율)

#transform(X): 적합된 스케일러를 사용하여 입력 데이터 X를 변환합니다. 
#이렇게 하면 데이터가 표준화됩니다. 즉, 데이터의 각 특성에 대해 평균을 빼고 
#표준편차로 나누어 데이터를 조정
#%%

from sklearn.model_selection import train_test_split
#  전체 데이터셋 중 30%를 테스트용 데이터로 사용하겠다는 의미입니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10) 

print('train data 개수: ', X_train.shape)
print('test data 개수: ', X_test.shape)
'''
train data 개수:  (499, 9)
test data 개수:  (215, 9)'''

#%%

#[Step 5] KNN 분류 모형 - sklearn 사용

#KNN은 지도학습 알고리즘 중 하나로, 새로운 데이터 포인트를 분류할 때,
# 그 주변에 가장 가까운 이웃들의 레이블을 기반으로 결정
# sklearn 라이브러리에서 KNN 분류 모형 가져오기
from sklearn.neighbors import KNeighborsClassifier

#n_neighbors: 이웃의 수를 지정하는 매개변수입니다. 새로운 데이터 포인트를 분류할 때
#고려할 이웃의 개수를 지정
# 모형 객체 생성 (k=5로 설정)
#n_neighbors : 5 기본값
# k의 개수를 홀수로 지정하는게 좋은게 짝수로 지정시 동점 상황이 만들어지면서
# 데이터를 분류할 수 없는 상황이 생길 수 있다 
knn = KNeighborsClassifier(n_neighbors=5)

# train data를 가지고 모형 학습
knn.fit(X_train, y_train)   

# test data를 가지고 y_hat을 예측 (분류) 
y_hat = knn.predict(X_test)

print('prediction :', y_hat[0:10])
print('정답 :', y_test.values[0:10])
'''
[0 0 1 0 0 1 1 1 0 0]
[0 0 1 0 0 1 1 1 0 0]'''


# 학습을 마친 모형에 test data를 적용하여 결정계수(R-제곱) 계산
r_square = knn.score(X_test, y_test)
print(r_square) # 0.8093023255813954


# 모형 성능 평가 - Confusion Matrix 계산
from sklearn import metrics 
knn_matrix = metrics.confusion_matrix(y_test, y_hat)  
print(knn_matrix)
# Confusion_matrix
"""
                 예측값
                 False   True
-------------|----------------
예측값 False |    TN      FP
       True  |    FN      TP
       
"""
# Confusion matrix
'''
[[110  15]
 [ 26  64]]
'''
# 모형 성능 평가 - 평가지표 계산
knn_report = metrics.classification_report(y_test, y_hat)            
print(knn_report)
"""
                정확도     재현율   F1 지표
              precision    recall  f1-score   support

           0       0.81      0.88      0.84       125 # 미생존자(0)의 정확도 0.84
           1       0.81      0.71      0.76        90 # 생존자(1)의 정확도 0.76

    accuracy                           0.81       215
   macro avg       0.81      0.80      0.80       215
weighted avg       0.81      0.81      0.81       215
"""
#%%
# 교재 p316 
TP = knn_matrix[1,1] # 정확함(양성을 양성으로 판단)  
FP = knn_matrix[0,1] # 음성을 양성으로 판단 1종 오류
FN = knn_matrix[1,0] # 양성을 음성으로 판단 2종 오류
TN = knn_matrix[0,0] # 정확함(음성을 음성으로 판단)
















