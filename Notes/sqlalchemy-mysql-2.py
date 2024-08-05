# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:19:22 2024

@author: Solero
"""

#%%

# SQLAlchemy를 사용한 예제

# pip install sqlalchemy
# pip install pymysql

#%%

import pandas as pd
from sqlalchemy import create_engine

#%%
    
# 데이터베이스 연결 URL 형식
# DATABASE_URL = 'mysql+pymysql://username:password@localhost:3306/database'
DATABASE_URL = 'mysql+pymysql://root:solsql@localhost:3306/hellodb'
engine = create_engine(DATABASE_URL)

#%%

# SQL 쿼리 정의
sql = "SELECT * FROM hello"

hello_df = pd.read_sql_query(sql, engine)
print(hello_df)

#%%

#  THE END