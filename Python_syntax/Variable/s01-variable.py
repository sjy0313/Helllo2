# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:22:12 2024

@author: Shin
"""

# 변수(variable)
# 메모리 저장 공간 
# 1 바이트 단위 
# 변수 이름 명명 규칙 
# 영문자(대/소문자)나 언더스코어(_)로 시작해야 한다
# 문자와 숫자 조합 가능
# 대소문자 구분(다르다)
# 예약어(reserved word, key word)는 사용할 수 없다 
# 변수명으로 한글사용가능 권고x 

# 변수 선언은 값을 할당(assing)해야 한다 
# 할당 연산자 (=) (==와 다르다)
# 할당은 우측의 값을 좌측 변수에 넣는다
# 기존의 좌측의 값은 지워지고 지정된 새로운 값이 들어간다.

# 변수 타입
# 지정되는 값에 의해 지정
# 이미 선언된 변수에 다른 타입으로 변환 가능
# 이미 선언된 변수에 다른 타입의 값을 넣으면 해당하는 타입으로 변환

# 변수 확인 
# 자료형 확인 : type()
# 메모리 확인 : id()


#%%
#id()
# 객체를 식별할 수 있는 고유의 값
# 메모리 주소와 맵핑된 형태
# id가 같으면 동일한 메모리 참조
#%%
h1 = "Hello"
h2 = "Hello"
print(id(h1))
print(id(h2))
# 동일한 메모리 주소를 참조하고 있다.
#%%
n1 = 100
n2 = 200
print(id(n1))
print(id(n2))
#140736765157256
#140736765160456

h2 = "World"
print('h1:', id(h1))
print('h2:', id(h2))
#h1: 2113185625264
#h2: 2113185813296 # id가 바뀜


#%%





_abc = "abc"
한글 = "한글"

print(_abc) #abc

# 특수문자($)는 지원하지 않는다.
# SyntaxError: invalid syntax

#%%
# 대소문자 구분 : 다른 변수로 취급한다.
abc = "abc"
ABC = "ABC"
print(abc, ABC) # abc ABC <class 'int'>
#%%
# 이미 선언된 변수에 다른 타입의 값을 넣으면 해당하는 타입으로 변환
abc = 999
print(abc, type(abc)) # 999 <class 'int'>

