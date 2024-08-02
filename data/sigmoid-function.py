# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 12:01:23 2024

@author: Shin
"""

# 시그모이드 함수(sigmoid function)
# 0~1(0~100%) 사이의 값
# z가 무한하게 큰 음수일 떄 0에 가까워 짐
# z가 무한하게 큰 양수일 떄 1에 가까워 짐
# 음성 : p가 0.5이하
# 양성 : p가 0.5보다 크면

import numpy as np 
import matplotlib.pyplot as plt 
z= np.arange(-10, 10 ,0.1)
p = 1 / (1 + np.exp(-z)) # 지수함수 계산(e**z / e**z + 1) # 0~1사이값으로 만들어주기 위함

plt.plot(z, p)
plt.xlabel('z')
plt.ylabel('p')
plt.show()

#%%


