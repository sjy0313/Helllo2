# -*- coding: utf-8 -*-

# 라이브러리 불러오기
import pandas as pd

# 파일경로를 찾고, 변수 file_path에 저장
file_path = './read_csv_sample.csv'

# read_csv() 함수로 데이터프레임 변환. 변수 df1에 저장 default 값 : header = 0, 'infer'
# 
df1 = pd.read_csv(file_path)
print(df1)
print('\n')
''' c0  c1  c2  c3
0   0   1   4   7
1   1   2   5   8
2   2   3   6   9'''
# read_csv() 함수로 데이터프레임 변환. 변수 df2에 저장. header=None 옵션
# csv 파일에 header가 존재 시 column index 가 아닌 데이터로 처리되기 떄문에 주의가 필요
df2 = pd.read_csv(file_path, header=None) 

print(df2)
''' 0   1   2   3
0  c0  c1  c2  c3
1   0   1   4   7
2   1   2   5   8
3   2   3   6   9
'''
print('\n')
#%%
df2nh= pd.read_csv(file_path, header=0) 
print(df2nh)
'''
   c0  c1  c2  c3
0   0   1   4   7
1   1   2   5   8
2   2   3   6   9
'''
#%%
df2in= pd.read_csv(file_path, header='infer') 
print(df2in)
'''
c0  c1  c2  c3
0   0   1   4   7
1   1   2   5   8
2   2   3   6   9
'''

#%%
# read_csv() 함수로 데이터프레임 변환. 변수 df3에 저장. index_col=None 옵션
df3 = pd.read_csv(file_path, index_col=None)
print(df3)
'''
   c0  c1  c2  c3
0   0   1   4   7
1   1   2   5   8
2   2   3   6   9'''
print('\n')

# read_csv() 함수로 데이터프레임 변환. 변수 df4에 저장. index_col='c0' 옵션
# index column의 이름지정할 떄는 파일을 읽어올 떄 index_col = name 옵션 필욧
df4 = pd.read_csv(file_path, index_col='c0')
print(df4)
'''
    c1  c2  c3
c0            
0    1   4   7
1    2   5   8
2    3   6   9
'''
#%%
# skiprows=[1] : 스킵하고 싶은 row 지정 
# skiprows=[0] : header가 다음행으로 승격됨. (header를 0으로 인식)
# skiprows=[1,3] : 1,3행 삭제  

df6 = pd.read_csv(file_path, sep=',', header = None , skiprows=[1])
print(df6)

''' 0   1   2   3
0  c0  c1  c2  c3
1   1   2   5   8
2   2   3   6   9
'''
#%%
#If, multiple header exists
# header(None), skiprows(0)
file_path = './read_csv_x.csv'
'''
h0,h1,h2,h3
c0,c1,c2,c3
0,1,4,7
1,2,5,8
2,3,6,9
'''
df7 = pd.read_csv(file_path, header = None, skiprows=[0,1], encoding="UTF-8")#header 2개 무시 
print(df7)
'''0  1  2  3
0  0  1  4  7
1  1  2  5  8
2  2  3  6  9
'''
#컬럼구분 : ,
#헤더 : 무시
#스킵 " 처음부터 첫쨰 header 무시
#인코딩 : UTF8
df8 = pd.read_csv(file_path, sep=',', skiprows=[0])#header 2개 무시 
print(df8)


