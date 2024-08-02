# -*- coding: utf-8 -*-

### 기본 라이브러리 불러오기

import pandas as pd
import matplotlib.pyplot as plt

# 귀납적_연역적 접근방식
#https://www.enago.co.kr/academy/inductive-and-deductive-reasoning/

# 비지도학습
# 군집(clustering) : k-Means, DBSCAN
# 분류와 차이는 정답이 없다.
# 유사성만을 기준으로 판단
# 신용카드 부정 사용 탐지, 구매 패턴 분석, 소비자 행동 특성
# 탐색적 자료 분석 / 피처 엔지니어링 용도로 사용가능.
# 최적의 k값을 사용자가 직접 선택

# 데이터셋의 관측값이 갖고 있는 여러 속성을 분석하여 서로 비슷한 특징을 갖는 관측값
# 끼리 클러스터로 묶는 알고리즘임. 

# 거리기반알고리즘으로 변수의 스케일에 따라 다른 결과를 나타남. 

# 적용분야 : 
    # 마케팅 , 제품기획 등을 목적으로 하는 고객분류 

# 어떤 소비자의 구매 패턴이나 행동 등을 예측하는데 활용가능
# K-Means 알고리즘
# 데이터 간의 유사성을 측정하는 기준으로 각 클러스터의 중심까지의 거리를 이용
# 어떤 데이터에 대하여 k개의 클러스터가 주어졌을 떄
# 클러스터 간에는 서로 완전하게 구분하기 위하여 일정한 거리 이상 떨어져야 한다. 
# 일반적으로 k값이 클수록 모델의 정확도가 개선되지만, k값이 너무 커지면 선택지가 많아지므로
# 분석의 효과가 사라진다. 

'''
[Step 1] 데이터 준비
'''

# Wholesale customers 데이터셋 가져오기 (출처: UCI ML Repository)

# https://archive.ics.uci.edu/dataset/00292/Wholesale+20customers


uci_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/\
00292/Wholesale%20customers%20data.csv'
df = pd.read_csv(uci_path, header=0)


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


'''
[Step 3] 데이터 전처리
'''

# 분석에 사용할 속성을 선택
X = df.iloc[:, :]
print(X[:5])
print('\n')

# 설명 변수 데이터를 정규화
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

print(X[:5])
print('\n')


'''
[Step 4] k-means 군집 모형 - sklearn 사용
'''

# sklearn 라이브러리에서 cluster 군집 모형 가져오기
from sklearn import cluster

# 모형 객체 생성 
kmeans = cluster.KMeans(init='k-means++', n_clusters=5, n_init=10)
# n_clusters=5 (default = 8) 클러스터 개수 지정
# 모형 학습
# init : 초기 클러스터 중심점을 어떻게 설정할지 지정
#   - random : 무작위로 초기화
#   - k-means++ : 더 나은 초기화 방법을 사용
# 
kmeans.fit(X)   

# 예측 (군집) 
cluster_label = kmeans.labels_   
print(cluster_label)
print('\n')

#%%
# 예측 결과를 데이터프레임에 추가
df['Cluster'] = cluster_label
print(df.head())   
print('\n')
#%%

# 그래프로 표현 - 시각화
# grocery, frozen
df.plot(kind='scatter', x='Grocery', y='Frozen', c='Cluster', cmap='Set1', 
        colorbar=False, figsize=(10, 10))
# milk, delicassen 
df.plot(kind='scatter', x='Milk', y='Delicassen', c='Cluster', cmap='Set1', 
        colorbar=True, figsize=(10, 10))
plt.show()
plt.close()

# 큰 값으로 구성된 클러스터(0, 4)를 제외 - 값이 몰려 있는 구간을 자세하게 분석
mask = (df['Cluster'] == 0) | (df['Cluster'] == 4)
ndf = df[~mask]

ndf.plot(kind='scatter', x='Grocery', y='Frozen', c='Cluster', cmap='Set1', 
        colorbar=False, figsize=(10, 10))
ndf.plot(kind='scatter', x='Milk', y='Delicassen', c='Cluster', cmap='Set1', 
        colorbar=True, figsize=(10, 10))
plt.show()
plt.close()
#%%
import seaborn as sns
sns.scatterplot(x="Milk", y="Delicassen", data=ndf, hue="Cluster", palette="rain")
