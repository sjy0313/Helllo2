# -*- coding: utf-8 -*-

# 라이브러리 불러오기
import pandas as pd
import matplotlib.pyplot as plt

# matplotlib 한글 폰트 오류 문제 해결
from matplotlib import font_manager, rc
font_path = "./malgun.ttf"   #폰트파일의 위치
font_name = font_manager.FontProperties(fname=font_path,size=8).get_name()
rc('font', family=font_name)

plt.style.use('ggplot')   # 스타일 서식 지정
plt.rcParams['axes.unicode_minus']=False   # 마이너스 부호 출력 설정

# Excel 데이터를 데이터프레임 변환 
df = pd.read_excel('./남북한발전전력량.xlsx', engine= 'openpyxl')
df = df.loc[5:9] # indexer 즉 index 5~8에 해당하는 데이터만 출력
# i.e., vertically -> i.e라틴어로 idest=다시말하면=즉=in other word
# 전력량의 세로열 삭제
df.drop('전력량 (억㎾h)', axis='columns', inplace=True)
# index 열로 이동
df.set_index('발전 전력별', inplace=True)
# Transpose -> 열과 행의 전치
df = df.T 

# 증감율(변동률) 계산
df = df.rename(columns={'합계':'총발전량'})
# df['총발전량']의 열의 값을 한칸 아래로 이동
# 이로서 첫번쨰 값과 마지막 값에 누락이 생김 (첫번쨰 값: none 출력 / 마지막 값: 자리x)
df['총발전량 - 1년'] = df['총발전량'].shift(1) 
# 전년대비 증감율을 알려면 나눈 후 -1을 해주어 -의 값이 나오면 
# 전년대비 감소되었다고 판단/ +값이 나오면 전년대비 증가
df['증감율'] = ((df['총발전량'] / df['총발전량 - 1년']) - 1) * 100      

# 2축 그래프 그리기
ax1 = df[['수력','화력']].plot(kind='bar', figsize=(20, 10), width=0.7, stacked=True)  
ax2 = ax1.twinx()
ax2.plot(df.index, df.증감율, ls='--', marker='o', markersize=20, 
         color='green', label='전년대비 증감율(%)')  

ax1.set_ylim(0, 500)
ax2.set_ylim(-50, 50)

ax1.set_xlabel('연도', size=20)
ax1.set_ylabel('발전량(억 KWh)')
ax2.set_ylabel('전년 대비 증감율(%)')

plt.title('북한 전력 발전량 (1990 ~ 2016)', size=30)
ax1.legend(loc='upper left')

plt.show()