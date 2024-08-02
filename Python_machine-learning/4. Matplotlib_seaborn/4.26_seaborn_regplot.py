# -*- coding: utf-8 -*-

# 라이브러리 불러오기
import matplotlib.pyplot as plt
import seaborn as sns
 
# Seaborn 제공 데이터셋 가져오기
titanic = sns.load_dataset('titanic')
 
# seaborn website : 
    # https://seaborn.pydata.org/examples/faceted_histogram.html
# 스타일 테마 설정 (5가지: darkgrid, whitegrid, dark, white, ticks)
sns.set_style('darkgrid')



#%%
# 보통 그래프를 출력할 환경(context)을 지정해주지 않았을 때 발생합니다.
# plt.rc_context() 메서드를 사용하여 환경을 설정해주면 해결할 수 있습니다.
# 이때 'paper', 'notebook', 'talk', 'poster' 중 하나를 선택하여 설정
# ValueError: context must be in paper, notebook, talk, poster
#sns.set_theme('darkgrid') 

# 그래프 출력 환경 설정 : paper, notebook, talk, poster

# "notebook"이란 용어는 Matplotlib에서 플로팅을 할 때 사용되는 환경 중 하나
# Notebook 환경은 대개 작은 그래프를 인라인으로 표시하므로 그림의 크기가 작습니다.
# 보통 기본 설정
#with plt.rc_context({'figure.figsize': (6, 4)}):  # 예시로 'notebook' 선택


# Paper 환경은 출판을 위한 그래프로, 일반적으로 크기가 더 크고 해상도가 높습니다.
#with plt.rc_context({'figure.figsize': (8, 6)}): 
    
    
#Talk 환경은 발표용 그래프로, 크기가 크고 주요 요소가 잘 보이도록 설정됩니다.
#with plt.rc_context({'figure.figsize': (10, 8)}): 
    
    
# Poster 환경은 대형 포스터나 큰 화면에 표시할 그래프로, 매우 큰 크기로 설정됩니다.
#with plt.rc_context({'figure.figsize': (12, 10)}): 
#%%
# 그래프 객체 생성 (figure에 2개의 서브 플롯을 생성)
fig = plt.figure(figsize=(15, 5))   
ax1 = fig.add_subplot(1, 2, 1) # 1행 2열의 첫번 쨰 
ax2 = fig.add_subplot(1, 2, 2) # 1행 2열의 두번 쨰 
 
# 그래프 그리기 - 선형회귀선 표시(fit_reg=True)
sns.regplot(x='age',        #x축 변수
            y='fare',       #y축 변수
            data=titanic,   #데이터
            ax=ax1)         #axe 객체 - 1번째 그래프 

# 그래프 그리기 - 선형회귀선 미표시(fit_reg=False)
sns.regplot(x='age',        #x축 변수
            y='fare',       #y축 변수
            data=titanic,   #데이터
            ax=ax2,         #axe 객체 - 2번째 그래프        
            fit_reg=False)  #회귀선 미표시

plt.show()