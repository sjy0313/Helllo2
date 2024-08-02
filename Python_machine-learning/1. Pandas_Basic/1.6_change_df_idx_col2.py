# -*- coding: utf-8 -*-

import pandas as pd

# 행 인덱스/열 이름 지정하여, 데이터프레임 만들기
df = pd.DataFrame([[15, '남', '덕영중'], [17, '여', '수리중']], 
                   index=['준서', '예은'],
                   columns=['나이', '성별', '학교'])

# 데이터프레임 df 출력
print(df)
'''
  나이 성별   학교
준서  15  남  덕영중
예은  17  여  수리중
'''
print("\n")

# 열 이름 중, '나이'를 '연령'으로, '성별'을 '남녀'로, '학교'를 '소속'으로 바꾸기
# dict의 key:value 형태와 같이 변경하고자하는 {열이름 지정 : 변경 후 이름지정}
# 만약에 []리스트로 싸주면 문법상 애러 발생 -> SyntaxError: invalid syntax 
# inplace=True 원본 변경
df.rename(columns={'나이':'연령', '성별':'남녀', '학교':'소속'}, inplace=True)

# df의 행 인덱스 중에서, '준서'를 '학생1'로, '예은'을 '학생2'로 바꾸기
df.rename(index={'준서':'학생1', '예은':'학생2' }, inplace=True)

# df 출력(변경 후)
print(df)
'''
 연령 남녀   소속
학생1  15  남  덕영중
학생2  17  여  수리중
'''
# inplace : False 를 주거나 지정x 원본이 변경되지 않음
# 리턴 사본
df2 = df.rename(columns={'나이':'연령', '성별':'남녀', '학교':'소속'})
df2 
'''
  연령 남녀   소속
준서  15  남  덕영중
예은  17  여  수리중
'''

df3 = df2.rename(index={'준서':'학생1', '예은':'학생2' }, inplace=False)
df3
df2
'''  수학  영어   음악   체육
서준  90  98   85  100
인아  70  95  100   90
''' 
# 새로운 바뀐 df3 생성 df2는 변경x
# 즉 변경된 사본에 새로운 변수 할당

# inplace=False는 rename() 메서드가 호출된 DataFrame을 직접 변경하지 않고
# 새로운 DataFrame을 반환하도록 지시하는 매개변수
'''
연령 남녀   소속
학생1  15  남  덕영중
학생2  17  여  수리중
'''