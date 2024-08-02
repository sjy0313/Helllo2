# -*- coding: utf-8 -*-

# 라이브러리 불러오기
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd

# 위키피디아 미국 ETF 웹 페이지에서 필요한 정보를 스크래핑하여 딕셔너리 형태로 변수 etfs에 저장
url = "https://en.wikipedia.org/wiki/List_of_American_exchange-traded_funds"
resp = requests.get(url)

#%%
html = resp.text
print(resp.text)


soup = BeautifulSoup(resp.text, 'lxml') 
# CSS 선택자를 사용하여 필요한 요소를 선택합니다. 여기서는 'div > ul > li'를 선택하여 각 ETF 항목을 가져옵니다.  
#rows = soup.select('div > ul > li')
rows = soup.select('li')
text_elements = soup.find_all(text=True)
etfs = []
etfs = row.text 
for row in rows:
    # re.findall(pattern, string) 함수는 주어진 문자열(string)에서 패턴(pattern)과 일치하는
    #모든 부분을 찾아 리스트로 반환합니다.
    print(row.text)

    print("-" * 100)
    try:
        # ^임의의 문자열의 시작 .는 임의의 한 문자를 의미하고, *는 바로 앞의 패턴이 0개 이상 나타날 수 있음을 나타냅니다
        etf_name = re.findall('^(.*) \(NYSE', row.text)
        etf_market = re.findall('\((.*)\|', row.text)
        etf_ticker = re.findall('NYSE Arca\|(.*)\)', row.text)
        
        if (len(etf_ticker) > 0) & (len(etf_market) > 0) & (len(etf_name) > 0):
            etfs[etf_ticker[0]] = [etf_market[0], etf_name[0]]

    except AttributeError as err:
        pass    



# etfs 딕셔너리 출력
print(etfs)
print('\n')

# etfs 딕셔너리를 데이터프레임으로 변환
df = pd.DataFrame(etfs)
print(df)