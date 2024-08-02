# -*- coding: utf-8 -*-

# 라이브러리 불러오기 
import pandas as pd
import numpy as np

# 딕셔너리 데이터로 판다스 시리즈 만들기

student1 = pd.Series({'국어':np.nan, '영어':80, '수학':90})
student2 = pd.Series({'수학':80, '국어':90})

print(student1)
국어     NaN
영어    80.0
수학    90.0
dtype: float64
print('\n')
print(student2)
수학    80
국어    90
dtype: int64
print('\n')

# 두 학생의 과목별 점수로 사칙연산 수행 (연산 메소드 사용)
sr_add = student1.add(student2, fill_value=0)    #덧셈
sr_sub = student1.sub(student2, fill_value=0)    #뺄셈
sr_mul = student1.mul(student2, fill_value=0)    #곱셈
sr_div = student1.div(student2, fill_value=0)    #나눗셈

# 사칙연산 결과를 데이터프레임으로 합치기 (시리즈 -> 데이터프레임)
result = pd.DataFrame([sr_add, sr_sub, sr_mul, sr_div], 
                      index=['덧셈', '뺄셈', '곱셈', '나눗셈'])
print(result)
'''
       국어       수학  영어
덧셈   90.0   170.000   80.0
뺄셈  -90.0    10.000   80.0
곱셈    0.0  7200.000   0.0
나눗셈   0.0     1.125   inf 
'''
#  inf(inity)  #  student2['영어']를 0으로 나눔.