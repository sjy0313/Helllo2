#!/usr/bin/env python
# coding: utf-8

# # 데이터셋 불러오기

# In[1]:


# 라이브러리 환경
import pandas as pd
import numpy as np


# In[2]:
# 붓꽃의 품종 판별 

# skleran 데이터셋에서 iris 데이터셋 로딩
from sklearn import datasets
iris = datasets.load_iris()

# iris 데이터셋은 딕셔너리 형태이므로, key 값을 확인
iris.keys()
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names',
# 'filename', 'data_module'])

# In[3]:


# DESCR 키를 이용하여 데이터셋 설명(Description) 출력
print(iris['DESCR'])
'''
 sepal length in cm # 꽃받침(길이)
- sepal width in cm      #너비
- petal length in cm     #꽃잎 
- petal width in cm
- class:
        - Iris-Setosa
        - Iris-Versicolour
        - Iris-Virginica
'''
# In[4]:


# target 속성의 데이터셋 크기
print("데이터셋 크기:", iris['target'].shape)  # 0 ,1 ,2 
 # 데이터셋 크기: (150,)

# target 속성의 데이터셋 내용
print("데이터셋 내용: \n", iris['target'])
'''데이터셋 내용: 
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]'''

# In[5]:


# data 속성의 데이터셋 크기
print("데이터셋 크기:", iris['data'].shape) # 데이터셋 크기: (150, 4)
#%%
# data 속성의 데이터셋 내용 (첫 7개 행을 추출)
print("데이터셋 내용: \n", iris['data'][:7, :])


# In[6]:


# data 속성을 판다스 데이터프레임으로 변환
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
print("데이터프레임의 형태:", df.shape)
df.head()
''' sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
'''

# In[7]:


# 열(column) 이름을 간결하게 변경
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df.head(2)
# (cm) 삭제

# In[8]:


# Target 열 추가
df['Target'] = iris['target']
print('데이터셋의 크기: ', df.shape)
df.head()


# # 데이터 탐색(EDA)

# In[9]:


# 데이터프레임의 기본정보
df.info()


# In[10]:


# 통계정보 요약
df.describe()


# In[11]:


# 결측값 확인
df.isnull().sum() # NON


# In[12]:


# 중복 데이터 확인
df.duplicated().sum() # 1 


# In[13]:


# 중복 데이터 출력 df.loc[중복된 행, 열 전체]
df.loc[df.duplicated(), :] # 중복된 데이터가 142행에 존재
'''
   sepal_length  sepal_width  petal_length  petal_width
142           5.8          2.7           5.1          1.9
'''
#%%
help(df.duplicated)
'''
   By setting ``keep`` on False, all duplicates are True.
    
df.duplicated(keep=False)
    0     True
    1     True
    2    False
    3    False
    4    False
    dtype: bool
    '''

# df_dup 이라는 변수 지정을 통해 중복 데이터를 따로 담아놔야함.(실무에선)
df.loc[df.duplicated(keep=False), :]
'''
 sepal_length  sepal_width  petal_length  petal_width
101           5.8          2.7           5.1          1.9
142           5.8          2.7           5.1          1.9'''


#%%

# 중복 데이터 모두 출력
df_dup = df.loc[(df.sepal_length==5.8)&(df.petal_width==1.9), :]
'''
    sepal_length  sepal_width  petal_length  petal_width
101           5.8          2.7           5.1          1.9
142           5.8          2.7           5.1          1.9
'''

# In[15]:


# 중복 데이터 제거
df = df.drop_duplicates()
df.loc[(df.sepal_length==5.8)&(df.petal_width==1.9), :]
'''
sepal_length  sepal_width  petal_length  petal_width
101           5.8          2.7           5.1          1.9'''


# In[16]:
df_dup.index
# Index([101, 142], dtype='int64'
# 142 행이 지워짐.
#%%
# 중복된 데이터에서 삭제처리 된 행을 출력. 
for n in df_dup.index:
    try:
        print(n, df.loc[n,:]) # 여기서 끝내면, KeyError: 142 (애러 발생)
    except KeyError as e: # 삭제된 행까지 확인가능 (예외 처리)
        print(e)
'''
101 
sepal_length    5.8
sepal_width     2.7
petal_length    5.1
petal_width     1.9
Name: 101, dtype: float64
142'''

df.loc[df_dup.index[0],:]
'''
101 sepal_length    5.8
sepal_width     2.7
petal_length    5.1
petal_width     1.9
Name: 101, dtype: float64
'''
# df.loc[df_dup.index, :] # KeyError: '[142] not in index'


#%%

# 변수 간의 상관관계 분석
#값이 1에 가까울수록 강한 양의 상관관계, -1에 가까울수록 강한 음의 상관관계,
# 0에 가까울수록 상관관계가 없음
 # ​-1에 가까운 값: 두 변수가 상반되는 강한 음의 상관관계 
 #(어느 한 쪽이 증가하면 다른 한 쪽이 반대로 감소)
df.corr()
'''
              sepal_length  sepal_width  petal_length  petal_width
sepal_length      1.000000    -0.118129      0.873738     0.820620
sepal_width      -0.118129     1.000000     -0.426028    -0.362894
petal_length      0.873738    -0.426028      1.000000     0.962772
petal_width       0.820620    -0.362894      0.962772     1.000000
'''

# In[17]:


# 시각화 라이브러리 설정
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)


# In[18]:


# 상관계수 히트맵
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()
# 꽃받침과 꽃잎의 상관관계가 상당히 높음을 알 수 있음.

# In[19]:


# Target 값의 분포 - value_counts 함수
df['Target'].value_counts()
'''
Target
0    50
1    50
2    49
Name: count, dtype: int64
'''

# In[20]:


# sepal_length 값의 분포 - hist 함수
plt.hist(x='sepal_length', data=df)
plt.show()


# In[21]:


# sepal_widgth 값의 분포 - displot 함수 (histogram)
sns.displot(x='sepal_width', kind='hist', data=df)
plt.show()


# In[22]:


# petal_length 값의 분포 - displot 함수 (kde 밀도 함수 그래프)
sns.displot(x='petal_width', kind='kde', data=df)
plt.show()


# In[23]:


# 품종별 sepal_length 값의 분포 비교
sns.displot( x='sepal_length', hue='Target', kind='kde', data=df)
plt.show()


# In[24]:


# 나머지 3개 피처 데이터를 한번에 그래프로 출력
for col in ['sepal_width', 'petal_length', 'petal_width']:
    sns.displot(x=col, hue='Target', kind='kde', data=df)
plt.show()


# In[25]:


# 두 변수 간의 관계
sns.pairplot(df, hue = 'Target', height = 2.5, diag_kind = 'kde')
plt.show()


# # Baseline 모델 학습

# #### 학습용-테스트 데이터셋 분리하기

# In[26]:


from sklearn.model_selection import train_test_split
 
X_data = df.loc[:, 'sepal_length':'petal_width']
y_data = df.loc[:, 'Target']
# 검증용 : 20% 
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                    test_size=0.2, 
                                                    shuffle=True, 
                                                    random_state=20)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
'''
(119, 4) (119,) # 훈련 데이터 
(30, 4) (30,) # 검증 데이터
'''
# 데이터가 적으므로 검증 데이터의 개수를 늘리면 정확도가 높아질 수 있겠지만
# 검증 데이터 수를 줄이게 되면 일반화를 하는데 사용할 수 있는 정보가 그만큼
# 작아져 일반화되는 정도를 제대로 평가하는데 한계가 있음. 

#데이터의 특성에 따라 적절한 훈련 및 검증 데이터의 비율이 달라질 수 있습니다. 
#특히 데이터가 매우 크거나 특이한 패턴을 가지고 있는 경우, 더 많은 데이터를
#검증에 사용하는 것이 도움이 될 수 있습니다.

# ### KNN

# In[27]:


# 모델 학습 : kNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)


# In[28]:

# 예측
y_knn_pred = knn.predict(X_test) 
print("예측값: ", y_knn_pred[:5]) #예측값:  [0 1 1 2 1]

#%%
# 정답이 틀린 위치
yn = y_knn_pred != y_test
print(yn)
 # 72      True
 
# In[29]:

# 성능 평가
from sklearn.metrics import accuracy_score
knn_acc = accuracy_score(y_test, y_knn_pred)
print("Accuracy: %.4f" % knn_acc)
 # Accuracy: 0.9667


##############################################################################

# ### SVM (support vector machine)
# 모델 학습
from sklearn.svm import SVC
# RBF(Radial Basic Function) 가우시안 함수에 기반한 비선형 대응
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)

# SVC (support vector classifier)
# 예측
y_svc_pred = svc.predict(X_test)
print("예측값: ", y_svc_pred[:5])


y_svc_pred 
# 성능 평가
svc_acc = accuracy_score(y_test, y_svc_pred)
print("Accuracy: %.4f" % svc_acc)

'''
예측값:  [0 1 1 2 1]
Accuracy: 1.0000
'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ###분류알고리즘 -  로지스틱 회귀(logisticRegression)
# 이진 분류(Binary Classification)
# 다중 분류(Multi-Class Classification)
# 선형회귀 분석에 근간을 두고 있는 분류 알고리즘
# 장점 : 
    # 선형회귀처럼 쉽다
    # 계수(기울기)를 사용해서 각 변수의 중요성을 파악

# In[32]:


# 모델 학습
from sklearn.linear_model import LogisticRegression
lrc = LogisticRegression()
lrc.fit(X_train, y_train)


# In[33]:


# 예측
y_lrc_pred = lrc.predict(X_test)
print("예측값: ", y_lrc_pred[:5])
# 성능 평가
lrc_acc = accuracy_score(y_test, y_lrc_pred)
print("Accuracy: %.4f" % lrc_acc)
'''
예측값:  [0 1 1 2 1]
Accuracy: 1.0000
'''

# In[34]:
# 모든 클래스(정답)에 대한 확률값
# 확률값 예측 : Probability estimates
y_lrc_prob = lrc.predict_proba(X_test)
y_lrc_prob
# (30,3) numpy 배열 출력 ->
# 첫 째열 : 클래스 0의 예측 확률
# 둘 째열 : 클래스 1의 예측 확률
# 셋 째열 : 클래스 2의 예측 확률
#%%
# 분활시켜 DB or 파일로 보관해놔야 후속처리 가능 : 
len(y_test) # 30

# 5.74319e-08 = "0.0000000574319"
'''
"e-08"은 지수 표기법(과학적 표기법)에서 사용되는 부분으로,
 수를 더 간결하게 표현하는 방법입니다.
여기서 "e"는 "exponent"의 약자로, 10의 거듭제곱을 나타냅니다. 
"e-08"은 10의 음의 8승을 의미합니다. 즉, 소수점을 왼쪽으로 8번 이동하라는 뜻이며,
 숫자를 더 작은 형태로 표현합니다.
 '''
 
for n in range(len(y_test)):
    print( "정답(%d), 인덱스(%3d):" % (y_test.iloc[n], y_test.index[n]), end='')
    for p in y_lrc_prob[n]:
        print("[%.4f]" % (p), end = '')
    print() # 줄바꿈 실행(다음 행의 출력을 준비)
    


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ### 의사결정나무
# sklearn 알고리즘모형 사용순서 : 객체생성 -> 모형학습 -> 훈련데이터의 기울기 구함 -> 모형평가
# sklearn 라이브러리에서 Decision Tree 분류 모형 가져오기

#거시적 상태와 미시적 상태 사이의 관계는 시스템의 모델링과 분석에 중요한 역할을 합니다.
#거시적 관측만으로는 시스템의 작동 원리를 완전히 이해하기 어려울 수 있으며, 
#미시적 상태를 고려하여 시스템을 더 깊이 이해할 수 있습니다.

# 확률이 낮을수록, 어떤 정보일지는 불확실하게 되고, 우리는 이때 '정보가 많다', '엔트로피가 높다'고 표현한다.

#%%

# 모형 객체 생성 (criterion='entropy' 적용)
# 각 분기점에서 최적의 속성을 찾기 위해 분류 정도를 평가하는 기준으로 entropy값을 사용
# 트리레벨을 5레벨로 지정(=5단계까지 가지확장) 레벨이 많아질수록 모형 학습에 사용하는 훈련
# 데이터에 대한 예측 정확해진다.(하지만 train_set에 대한 모형이 최적화 되면 상대적으로 실제
# 데이터  예측 능력은 떨어지는 문제발생) 따라서 적정한 레벨값을 찾는 것이 중요. 
# 복잡성에 따라 깊이설정
# criterion = 'gini' 기본값 (불순도) 노드의 순도를 평가하는 방법
# 결정트리가 최적의 질문을 찾기 위한 기준
# criterion='entropy' : 'entropy'는 엔트로피를 사용하여 정보이득을 계산합니다. 

#엔트로피는 정보의 불확실성을 나타내며, 불순도가 낮을수록 더 좋은 분할
# 엔트로피는 '어떤 상태에서의 불확실성', 또는 이와 동등한 의미로 '평균 정보량'을 의미

#정보 엔트로피가 커지는것은 역시 변수(불확실성)가 증가하는 것을 의미하므로, 
#변수를 제어함으로써 불확실성이 줄어드는 것은 결국 정보 획득을 의미하게 된다

# 노드의 순도를 평가하는 방법 : 노드의 순도가 높을수록 지니나 엔트로피 값은 낮아진다


# max_depth = None
# None 이면 leaf가 불순(entropy가 0에 가까워질떄까지)(데이터의 불확실함?) 하지 않을 때 까지 node(분기점) 확장  or
# 모든 leaf가 min_samples_split 노드를 분활하는데 필요한 최소한의 데이터 수 보다 적어질 떄까지 확장

#min_samples_split 매개변수는 의사결정트리에서 노드를 분할하기 위한 최소한의 샘플 수를 지정하는 요소입니다.
# 즉, 노드를 분할하기 위해 필요한 최소한의 데이터 포인트 수


# max_depth=5: 이 매개변수는 의사결정트리의 최대 깊이를 지정합니다. 
# 의사결정트리가 훈련 데이터에 너무 깊이 들어가면 과적합(overfitting)이 발생할 수 있습니다.

# In[35]:


# 모델 학습 및 예측
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=3, random_state=20)
dtc.fit(X_train, y_train)


# In[36]:


# 예측
y_dtc_pred = dtc.predict(X_test)
print("예측값: ", y_dtc_pred[:5])
# 성능 평가
dtc_acc = accuracy_score(y_test, y_dtc_pred)
print("Accuracy: %.4f" % dtc_acc)
'''
예측값:  [0 1 1 2 1]
Accuracy: 0.9333
'''
#%%
print("###보팅 모델 성능 평가###")
#%%

####################################################################################
#앙상블 모델 (Ensemble Learning -> 여러 모델을 만들어 각예측 값을 투표 or 평균으로 통합예측)
### 보팅 
# 랜덤포레스트(random forest)는 결정트리(decision tree)모델을 여러 개 사용하여 각 모델의
# 예측값을 voting하여 결정
# 같은 종류의 알고리즘 모델을 여러 개 결합하여 예측하는 방법을 배깅(baggin)이라 한다
# 각 트리는 전체 학습 데이터 중에서 서로 다른 데이터를 샘플링하여 학습

# 작동방식 : 
    # decisiontree 모델을 여러 개 생성
    # 전체 학습 데이터 중에서 서로 다른 데이터를 샘플링
    # 각 모델의 예측값들을 더한 후, 평균값을 구함. 평균값이 곳 최종결과

# Pros (random forest)
#  결정트리(decision tree)모델(단점: 트리가 무한정 깊어지면 오버피팅 문제)
# 오버 피팅 문제 완화
# 선형이나 비선형 데이터에 상관없이 잘 동작
# 랜덤하게 생성된 많은 트리를 이용하여 예측

# Cons (random forest)
# 학습속도 상대적으로 느림
# 수 많은 트리를 동원하기 때문에 모델에 대한 해석이 어렵다

# 유용성 : 
    #  - 종속변수(y)가 연속된/범주형 데이터인 경우 모두 사용가능(선형/ 비선형)
    # outlier(이상치)가 문제가 되는 경우 선형 모델보다 좋은 대안이 될 수 있음
    # (overfitting)오버피팅 문제로 결정 트리를 사용하기 어려울 떄 대안이 됨. 

# In[37]:


# Hard Voting 모델 학습 및 예측  : 다수결을 통해 최종 예측을 결정
# Soft Voting : 각 모델의 예측에 대한 확률을 고려하(class별 확률을 예측) 
#이 확률들의 평균을 내어 최종 예측을 만듭니다. 
#보통은 각 모델의 예측 확률에 가중치를 부여하여 계산
# voting = 'hard' : 다수결로 최종 분류 결정 (soft-voting) 은 평균값임.
# knn, svc, dtc (모델(객체)은 만들어줘야 하지만 훈련시킬 필요는 없다)

#from sklearn.ensemble import VotingClassifier
#hvc = VotingClassifier(estimators=[('KNN', knn), ('SVM', svc), ('DT', dtc)], 
#                     voting='soft')


# 튜플에 들어가 있음(전달되는 인자(parameter)는 모델 객체aks 필요 -> 훈련x )
from sklearn.ensemble import VotingClassifier
hvc = VotingClassifier(estimators=[('KNN', knn), ('SVM', svc), ('DT', dtc)], 
                       voting='hard')
hvc.fit(X_train, y_train)
 
# 예측
y_hvc_pred = hvc.predict(X_test)
print("예측값: ", y_hvc_pred[:5])

# 성능 평가
hvc_acc = accuracy_score(y_test, y_hvc_pred)
print("Accuracy(Voting Classifier) SVM : %.4f" % hvc_acc)


'''
Ensemble learning에서 soft voting과 hard voting은 다른 투표 방법을 나타냅니다.
 여러 다른 모델들의 예측을 결합하여 최종 예측을 만들 때 사용됩니다.

Hard Voting:
Hard voting은 각각의 모델이 클래스 레이블 하나를 예측하고, 
이 예측들 중에서 다수결을 통해 최종 예측을 결정합니다. 
즉, 각 모델의 예측을 단순히 과반수를 차지하는 클래스로 선택합니다.
Soft Voting:
Soft voting은 각 모델의 예측에 대한 확률을 고려하여,
 이 확률들의 평균을 내어 최종 예측을 만듭니다.
 보통은 각 모델의 예측 확률에 가중치를 부여하여 계산합니다.
 확률을 고려하기 때문에, 예측이 불확실한 경우에도 유연하게 처리할 수 있습니다.
 
일반적으로는 soft voting이 hard voting보다 더 나은 결과를 제공할 수 있습니다.
 왜냐하면 soft voting은 각 모델의 확신 정도를 고려하여 최종 예측을 결정하므로 
 더 많은 정보를 활용할 수 있기 때문입니다.

 하지만 모델 간의 상관 관계가 크거나,
 특정 모델이 다른 모델보다 더 신뢰할만한 경우에는 hard voting이 더 나을 수도 있습니다.
 '''
####################################################################################
# ### 배깅 (랜덤포레스트)
# Decision tree 모델을 여러 개 결합
# 서로 다른 데이터를 샘플링
# n_estimators : 트리 모델의 갯수
# max_depth(개별 트리의 깊이)
# In[38]:


# 모델 학습 및 검증
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=20)
rfc.fit(X_train, y_train)
# 예측
y_rfc_pred = rfc.predict(X_test)
print("예측값: ", y_rfc_pred[:5])
# 모델 성능 평가
rfc_acc = accuracy_score(y_test, y_rfc_pred)
print("Accuracy: %.4f" % rfc_acc)
# randomforest :
'''
예측값:  [0 1 1 2 1]
Accuracy: 0.9667
'''
# decision tree :
'''
예측값:  [0 1 1 2 1]
Accuracy: 0.9333
'''
# ### 앙상블 모델 - 부스팅 (XGBoost) eXtreme Gradient Boosting
import sklearn as sk
print(sk.__version__) # 1.4.2
# pip install xgboost
#%%
# 부스팅 모델은 순차적으로 트리를 만들어 이전 트리로부터 더 나은 트리를 만들어 내는 알고리즘
# 점진적 상승모델 (랜덤포레스트보다 빠르며 성능이 우수함.)
# 종류 : XGBoost , LightGBM, CatBoost

# 장점: 
    # -예측 속도 빠름 / 예측력 좋음 / 변수 종류가 많고 데이터가 클수록 상대적으로 좋은 성능을 낸다
# 단점 : 
    # - 해석이 어려움
    # - 하이퍼 파라미터(hyper parameter) 튜닝이 어렵다.
# 활용 :
    # - 종속변수가 연속형 데이터나 범주형으로 모두 사용가능 
    # - 이미지나 자연어가 아닌 표 형태로 정리된 데이터인 경우 활용
    
# 모듈 : from xgboost import XGBClassifier
# 예시 : XGBClassifier(n_estimators=50, max_depth=3, random_state=20)

# 설명 : 
# 잘못 예측한 데이터에 대해 오차를 줄일 수 있는 방향으로 진행
# In[39]:


# 모델 학습 및 예측
from xgboost import XGBClassifier
xgbc = XGBClassifier(n_estimators=50, max_depth=3, random_state=20)
xgbc.fit(X_train, y_train)
# 예측
y_xgbc_pred = xgbc.predict(X_test)
print("예측값: ", y_xgbc_pred[:5])
# 모델 성능 평가
xgbc_acc = accuracy_score(y_test, y_xgbc_pred)
print("Accuracy: %.4f" % xgbc_acc)
'''
예측값:  [0 1 1 2 1]
Accuracy: 0.9333
'''
###############################################################################
# # 교차 검증 (Cross-Validation)
# 데이터 일부를 검증 데이터로 사용하는 방법
# ### Hold out 교차 검증
# 검증 데이터는 모델학습에 사용하지 않는 데이터로 모델의 일반화 성능평가에 사용 
# 훈련 / 검증 (기존) -> 훈련데이터(훈련/검증) 한번 더 쪼갬)
# 테스트 데이터에 대한 예측력을 높임
# In[40]:


# 검증용 데이터셋 분리
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, 
                                            test_size=0.3, 
                                            shuffle=True, 
                                            random_state=20)
print(X_tr.shape, y_tr.shape)
print(X_val.shape, y_val.shape)
# 맨위 훈련 데이터셋 119개에서 훈련/검증 데이터 개수 나눔 (3:7)
'''
(83, 4) (83,)
(36, 4) (36,)
'''

# In[41]:


# 학습
rfc = RandomForestClassifier(max_depth=3, random_state=20)
rfc.fit(X_tr, y_tr)
# 예측
y_tr_pred = rfc.predict(X_tr)
y_val_pred = rfc.predict(X_val)
# 검증
tr_acc = accuracy_score(y_tr, y_tr_pred)
val_acc = accuracy_score(y_val, y_val_pred)
print("Train Accuracy: %.4f" % tr_acc)
print("Validation Accuracy: %.4f" % val_acc)
'''
Train Accuracy: 0.9880
Validation Accuracy: 0.9167
'''
# In[42]:


# 테스트 데이터 예측 및 평가
y_test_pred = rfc.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print("Test Accuracy: %.4f" % test_acc)
# Test Accuracy: 0.9000

# ### K-Fold 교차 검증

# In[43]:

# n_splits=5 5개로 분활했기 때문에  8:2 비율을 가짐
# 데이터셋을 5개의 Fold로 분할하는 KFold 클래스 객체 생성
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=20)
# 훈련용 데이터와 검증용 데이터의 행 인덱스를 각 Fold별로 구분하여 생성
num_fold = 1
for tr_idx, val_idx in kfold.split(X_train): 
    print("%s Fold----------------------------------" % num_fold)
    print("훈련: ", len(tr_idx), tr_idx[:10])
    print("검증: ", len(val_idx), val_idx[:10])
    num_fold = num_fold + 1


# In[44]:


# 훈련용 데이터와 검증용 데이터의 행 인덱스를 각 Fold별로 구분하여 생성
val_scores = []
num_fold = 1
for tr_idx, val_idx in kfold.split(X_train, y_train):
    # 훈련용 데이터와 검증용 데이터를 행 인덱스 기준으로 추출
    X_tr, X_val = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
    # 학습
    rfc = RandomForestClassifier(max_depth=5, random_state=20)
    rfc.fit(X_tr, y_tr)
    # 검증
    y_val_pred = rfc.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)  
    print("%d Fold Accuracy: %.4f" % (num_fold, val_acc))
    val_scores.append(val_acc)   
    num_fold += 1  
'''
1 Fold Accuracy: 0.8750
2 Fold Accuracy: 1.0000
3 Fold Accuracy: 0.9167
4 Fold Accuracy: 0.9583
5 Fold Accuracy: 0.9565
'''

# In[45]:


# 평균 Accuracy 계산
import numpy as np
mean_score = np.mean(val_scores)
print("평균 검증 Accuraccy: ", np.round(mean_score, 4))

