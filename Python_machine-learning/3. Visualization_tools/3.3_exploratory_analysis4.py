# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:02:31 2024

@author: Shin
"""

import pandas as pd

# read_csv() 함수로 df 생성
df = pd.read_csv('./auto-mpg.csv', header=None)
df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

df.drop('name', axis= 1, inplace=True)

print(df.iloc[:, 0:3])
'''
  mpg  cylinders  displacement
0    18.0          8         307.0
1    15.0          8         350.0
2    18.0          8         318.0
3    16.0          8         304.0
4    17.0          8         302.0
..    ...        ...           ...
393  27.0          4         140.0
394  44.0          4          97.0
395  32.0          4         135.0
396  28.0          4         120.0
397  31.0          4         119.0

[398 rows x 3 columns]'''