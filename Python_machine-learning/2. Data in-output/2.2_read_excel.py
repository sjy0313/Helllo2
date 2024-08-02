# -*- coding: utf-8 -*-

import pandas as pd

# read_excel() 함수로 데이터프레임 변환 
# excel : 병합되어 있는 셀은 데이터 프레임으로 전환하면 nan
df1 = pd.read_excel('./남북한발전전력량.xlsx', engine='openpyxl')            # header=0 (default 옵션)
df2 = pd.read_excel('./남북한발전전력량.xlsx', engine='openpyxl', 
                    header=None)  # header=None 옵션

#

# 데이터프레임 출력
print(df1)
print('\n')
print(df2)