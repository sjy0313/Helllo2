# -*- coding: utf-8 -*-


# 결정트리(Decision Tree)
# 의사결정나무
# 관측값과 목표값을 연결시켜주는 예측모델
# 나무구조(tree), 분기점(node-> git commit과 유사)
# 선형모델과는 다른 특징을 가짐 - 선형은 기울기를 찾음. 
# 특정지점을 기준으로 분류
# 예측력 + 성능은 뛰어나지 않음 하지만 시각화에 특화된 모델.(설명력이 좋음)
# 설명력 : 중요한 요인을 밝히는데 

# 장점 : 데이터에 대한 자정이 업는 모델
# 단점 : 
#   - 트리가 무한정 깊어 지면 오버피팅 문제(과접합)
#   - 예측력이 떨어짐
#%%
# 기본 라이브러리 불러오기
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
'''
[Step 1] 데이터 준비/ 기본 설정
'''

# Breast Cancer 데이터셋 가져오기 (출처: UCI ML Repository)
uci_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/\
breast-cancer-wisconsin/breast-cancer-wisconsin.data'
df = pd.read_csv(uci_path, header=None)
# 샘플 ID, 암세포 조직의 크기와 모양 등 종양 특성을 나타내는 열 9개, (2: 양성 / 4: 악성)을 나타내는 열
# 열 이름 지정
df.columns = ['id', 'clump', 'cell_size', 'cell_shape', 'adhesion', 'epithlial',
              'bare_nuclei', 'chromatin', 'normal_nucleoli', 'mitoses', 'class']
# bare_nuclei : 세포핵
#  IPython 디스플레이 설정 - 출력할 열의 개수 한도 늘리기
pd.set_option('display.max_columns', 15)


'''
[Step 2] 데이터 탐색
'''

# 데이터 살펴보기
print(df.head())
print('\n')

# 데이터 자료형 확인
print(df.info())
print('\n')

# 데이터 통계 요약정보 확인
print(df.describe())
print('\n')

# bare_nuclei 열의 자료형 변경 (문자열 ->숫자)
# bare_nuclei 열의 고유값 확인
print(df['bare_nuclei'].unique())
print('\n')

# df['bare_nuclei'].replace('?', np.nan, inplace=True)      # '?'을 np.nan으로 변경
df['bare_nuclei'] = df['bare_nuclei'].replace({'?': np.nan}) #  # '?'을 np.nan으로 변경


df.dropna(subset=['bare_nuclei'], axis=0, inplace=True)   # 누락데이터 행을 삭제
df['bare_nuclei'] = df['bare_nuclei'].astype('int')       # 문자열을 정수형으로 변환

print(df.describe())                                      # 데이터 통계 요약정보 확인

#%%
df.dtypes
print('\n')


'''
[Step 3] 데이터셋 구분 - 훈련용(train data)/ 검증용(test data)
'''

# 속성(변수) 선택
'''
X = df[['clump', 'cell_size', 'cell_shape', 'adhesion', 'epithlial',
        'bare_nuclei', 'chromatin', 'normal_nucleoli', 'mitoses']]  # 설명 변수 X
'''
xcols = ['clump', 'cell_size', 'cell_shape', 'adhesion', 'epithlial',
        'bare_nuclei', 'chromatin', 'normal_nucleoli', 'mitoses']
X = df[xcols] # 설명 변수 X

y = df['class']  # 예측 변수 Y

#%%
class_unique = df['class'].unique()
print("class:", class_unique) # class: [2 4]



# 설명 변수 데이터를 정규화
X = preprocessing.StandardScaler().fit(X).transform(X)

# train data 와 test data로 구분(7:3 비율)
# shuffle : True / test_size = 30% 검증데이터 비율 / random_state=10 난수 고정 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=10) 

print('train data 개수: ', X_train.shape) # train data 개수:  (478, 9)
print('test data 개수: ', X_test.shape) # test data 개수:  (205, 9)
print('\n')


'''
[Step 4] Decision Tree 분류 모형 - sklearn 사용
'''
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
tree_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
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


# train data를 가지고 모형 학습
tree_model.fit(X_train, y_train)

# test data를 가지고 y_hat을 예측 (분류)
y_hat = tree_model.predict(X_test)      # 2: benign(양성), 4: malignant(악성)

print(y_hat[0:10]) # [4 4 4 4 4 4 2 2 4 4] 예측 
print(y_test.values[0:10]) # [4 4 4 4 4 4 2 2 4 4] 정답
print('\n')


# 모형 성능 평가 - Confusion Matrix 계산
tree_matrix = metrics.confusion_matrix(y_test, y_hat)
print(tree_matrix)
# Confusion_matrix
"""
                    실제값
                  Positive(1)    Negative(0)
   -------------|---------------------------
예측값 Pos(1)   |    TP              FP
       Neg(0)   |    FN              TN
       
"""
# 출력 형태
'''
[[TN  FP] 
 [FN  TP]]
'''
'''
[[127   4]
 [  2  72]]
'''
print('\n')

# 모형 성능 평가 - 평가지표 계산
tree_report = metrics.classification_report(y_test, y_hat)
print(tree_report)
'''

              precision    recall  f1-score   support

           2       0.98      0.97      0.98       131
           4       0.95      0.97      0.96        74

    accuracy                           0.97       205
   macro avg       0.97      0.97      0.97       205
weighted avg       0.97      0.97      0.97       205
'''
#%%
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(30, 15))
plot_tree(tree_model) # 출력할 모델지정
plt.show()

#%%

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(40, 15))
plot_tree(tree_model, fontsize=20, feature_names=xcols, filled=True)
plt.show()

# samples = 478
# yes -> 왼쪽 / no -> 오른쪽
# filled=True (색상) 색상의 구분으로 class분류
# 노드(node) : max_deapth=5
# 평가 : 과소적합(모델이 충분히 학습x 훈련데이터에 대해서 예측력이 떨어짐)

