#!/usr/bin/env python
# coding: utf-8

# In[ ]:
# p344 사진 그림 4-39
# relu에 대한 실행 예

import tensorflow as tf
# 정규분포에서 임의로 추출된 입력에 대한 relu 동작 예시 
# 0보다 작은 값은 0으로
# 0보다 큰 값은 해당하는 값
# shape : [1,5]
# mean : 정규분포의 평균(기본값 0)
# stddev : 정규분포의 표준편차(기본값 1)

norm = tf.random.normal([1, 5], mean=-1, stddev=10, dtype="half")
print("input : ", norm)

tf.keras.activations.relu(norm).numpy()
'''
input :  tf.Tensor([[  2.762  -1.838   5.168   9.98  -28.56 ]], shape=(1, 5), dtype=float16)
Out[5]: array([[2.762, 0.   , 5.168, 9.98 , 0.   ]], dtype=float16)
'''
# In[ ]:


al = tf.keras.activations.relu(norm, alpha=0.5)
max_v = tf.keras.activations.relu(norm, max_value=1)
thres = tf.keras.activations.relu(norm, threshold=norm[0,0])
print("alpha : ", al)
print("\nmax_value : ", max_v)
print("\nthreshold : ", thres)


# In[ ]:
# 시그모이드 
# 실수 입력을 0과 1사이의 값으로 변환

# 쓰임: 
# 일반적으로 이진 분류 문제에서 출력층의 활성화 함수로 사용됩니다.
# 입력 값이 극단적으로 클 때나 작을 때 기울기가 매우 작아지는 기울기
# 소실 문제(Vanishing Gradient Problem)가 발생할 수 있습니다.
from matplotlib import pyplot as plt

in1 = tf.constant([-10, -5.0, -2.0, 0.0, 2.0, 5.0, 10], dtype = tf.float32)
out1 = tf.keras.activations.sigmoid(in1)
print("sigmoid : ", out1)

plt.plot(in1 ,out1)
plt.title("Sigmoid Visualization")


# In[ ]:

# 소프트맥스 
# 소프트맥스 함수는 벡터 입력을 받아 각 요소를 0과 1 사이의 값으로 변환
# 하며, 출력 값들의 총합이 1이 되도록 만듭니다.
# 쓰임: 
# 다중 클래스 분류 문제에서 출력층 전체의 활성화 함수.
# 범주형 확률의 백터 : 0과 1사이의 확률
# 백터의 합이 1
# 출력값이 1

in2 = tf.constant([[0.0, 1.0, 5.0, -2.0, 4.0, 8.0, 3.0]], dtype = tf.float32)
plt.plot(in2[0,:])
plt.title("Input Visualization")
plt.show()

out2 = tf.keras.activations.softmax(in2, axis=-1).numpy()
print("softmax : ", out2)
print("\nsum of softmax : ", out2.sum())

x = [x for x in range(7)]
plt.plot(x, out2[0,x], '-')

plt.title("Softmax Visualization")
plt.show()

