#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt

sample1 = 200 + np.random.randn(10)
# np.random.randn(10)
#이 함수는 평균이 0이고 표준 편차가 1인 표준 정규 분포에서 10개의 랜덤 숫자를 생성합니다.
print(sample1)

sample2 = 200 + np.random.randn(10)
# 생성된 랜덤 숫자 각각에 200을 더하여 평균을 0에서 200으로 이동시킵니다.
print(sample2)

print(abs(sample1-sample2))

x = [x for x in range(len(sample2))]
plt.plot(x, sample1, '-')
plt.plot(x, sample2, '-')
plt.fill_between(x, sample1, np.maximum(sample1, sample2),
                 where=(abs(sample1-sample2) > 1), facecolor='g', alpha=0.6)
#  where=(abs(sample1-sample2) > 1)  조건임. 
plt.title("Sample Visualization")
plt.show()

