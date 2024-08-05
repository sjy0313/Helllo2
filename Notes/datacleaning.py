# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:36:27 2024

@author: Shin
"""

import pandas as pd


df_features = pd.read_excel('./project/book_info.xlsx')
df_genre = pd.read_excel('./Project/Genrelist_of_bestseller2023.xlsx')

# 읽어오기 

df1 = pd.read_excel('./Project/Genrelist_of_bestseller2020.xlsx')
df2 = pd.read_excel('./Project/Genrelist_of_bestseller2021.xlsx')
df3 = pd.read_excel('./Project/Genrelist_of_bestseller2022.xlsx')
df4 = pd.read_excel('./Project/Genrelist_of_bestseller2023.xlsx')

# 읽어오기 
import pandas as pd

excel_f = [] 
for year in range(2020, 2024):
    excel_f.append(pd.read_excel(f"./Project/Genrelist_of_bestseller{year}.xlsx"))
    print(excel_f)
   
df_dict = {f"df{i}": df for i, df in enumerate(excel_f, 1)}
print(df_dict)

genre_lst = excel_f[0]
genre_dict = genre_lst['장르'].value_counts().to_dict() 
genre_df = pd.DataFrame([genre_dict])
  
for genre_lst in excel_f:
    genre_dict = excel_f[0,3]
    genre_dict = genre_lst['장르'].value_counts().to_dict() 
    genre_df = pd.DataFrame([genre_dict])
    
    
import pandas as pd

for genre_lst in excel_f:  
    genre_dict = genre_lst['장르'].value_counts().to_dict()  # 각 데이터프레임에 대해 '장르'열의 빈도수를 딕셔너리로 변환
    genre_dfs = pd.DataFrame([genre_dict])  
    print(genre_dfs)

genre_dfs
