# -*- coding: utf-8 -*-

import pandas as pd

# 판다스 DataFrame() 함수로 데이터프레임 변환. 변수 df1, df2에 저장 
data1 = {'name' : [ 'Jerry', 'Riah', 'Paul'],
         'algol' : [ "A", "A+", "B"],
         'basic' : [ "C", "B", "B+"],
          'c++' : [ "B+", "C", "C+"]}

data2 = {'c0':[1,2,3], 
         'c1':[4,5,6], 
         'c2':[7,8,9], 
         'c3':[10,11,12], 
         'c4':[13,14,15]}

df1 = pd.DataFrame(data1)
df1.set_index('name', inplace=True)      #name 열을 인덱스로 지정
print(df1)
print('\n')

df2 = pd.DataFrame(data2)
df2.set_index('c0', inplace=True)        #c0 열을 인덱스로 지정
print(df2)

# df1을 'sheet1'으로, df2를 'sheet2'로 저장 (엑셀파일명은 "df_excelwriter.xlsx")
# pandas 의 writer객체로 전달
# 2번쨰 시트까지 생성하여 읽기
with pd.ExcelWriter("./df_excelwriter.xlsx") as writer: 
    df1.to_excel(writer, sheet_name="시트1")
    df2.to_excel(writer, sheet_name="시트2")

#%%
#To specify the sheet name:
df1.to_excel("output.xlsx",sheet_name='Sheet_name_1')  

#%%
# If you wish to write to more than one sheet in the workbook, it is necessary to specify an ExcelWriter object:

>>> df2 = df1.copy()
>>> with pd.ExcelWriter('output.xlsx') as writer:  
...     df1.to_excel(writer, sheet_name='Sheet_name_1')
...     df2.to_excel(writer, sheet_name='Sheet_name_2')






  
# AttributeError: 'XlsxWriter' object has no attribute 'save'
# 생략

#writer.close()