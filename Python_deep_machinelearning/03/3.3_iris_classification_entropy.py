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
df.loc[df_dup.index, :] # KeyError: '[142] not in index'


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

#%%
# MSE(Mean Squared Error)를 구해서 가장 낮은 MSE값이 나오도록 노드(node)를 분리한다.
# In[35]:


# 모델 학습 및 예측

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # 정규화 위한 모듈 가져오기 

 
X_data = df.loc[:, 'sepal_length':'petal_width']
y_data = df.loc[:, 'Target']

# 설명 변수 데이터를 정규화
X = StandardScaler().fit(X_data).transform(X_data)  

# 검증용 : 20% 
X_train, X_test, y_train, y_test = train_test_split(X, y_data, 
                                                    test_size=0.2, # 테스트 데이터 20%
                                                    shuffle=True,  # 데이터 섞음
                                                    random_state=20) # 난수 

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy', max_depth= 5, random_state=20)
dtc.fit(X_train, y_train)



# In[36]:


# 예측
y_dtc_pred = dtc.predict(X_test)
print("예측값: ", y_dtc_pred[:5])

# 성능 평가
from sklearn.metrics import accuracy_score
dtc_acc = accuracy_score(y_test, y_dtc_pred)
print("Accuracy: %.4f" % dtc_acc)

'''
(119, 4) (119,)
(30, 4) (30,)
예측값:  [0 1 1 2 1]
Accuracy: 0.9333

'''
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# 모형 성능 평가 - Confusion Matrix 계산
tree_matrix = metrics.confusion_matrix(y_test, y_dtc_pred)
print(tree_matrix)

xcols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

plt.figure(figsize=(40, 15))
plot_tree(dtc, fontsize=20, feature_names=xcols, filled=True)
plt.show()

# samples = 478
# yes -> 왼쪽 / no -> 오른쪽
# filled=True (색상) 색상의 구분으로 class분류
# 노드(node) : max_deapth=5
# 평가 : 과소적합(모델이 충분히 학습x 훈련데이터에 대해서 예측력이 떨어짐)
