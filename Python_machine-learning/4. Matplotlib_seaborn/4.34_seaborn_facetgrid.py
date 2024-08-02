# -*- coding: utf-8 -*-

# 라이브러리 불러오기
import matplotlib.pyplot as plt
import seaborn as sns
 
# Seaborn 제공 데이터셋 가져오기
titanic = sns.load_dataset('titanic')
 
# 스타일 테마 설정 (5가지: darkgrid, whitegrid, dark, white, ticks)
sns.set_style('whitegrid')

# 조건에 따라 그리드 나누기

# FacetGrid는 데이터를 여러 부분집합으로 나누어 각각을 따로 시각화할 수 있도록 도와줍니다.
# 주로 범주형 변수에 대한 분석에서 사용되며, 각 부분집합에 해당하는 그래프를 격자 형태로
# 배열하여 한눈에 비교할 수 있게 합니다. FacetGrid를 사용하면 데이터의 패턴이나 관계를 
# 서로 다른 부분집합에 대해 쉽게 확인할 수 있습니다.

#  데이터를 세분화 시켜 man/woman/child 에 대해 적용
# titanic data에 대해 who는 man/woman/child 세분화 -> 열의 구조여부
# 구조 = survived = 1
# 구조실패 = survived = 0 
g = sns.FacetGrid(data=titanic, col='who', row='survived') 

# 그래프 적용하기
g = g.map(plt.hist, 'age')
