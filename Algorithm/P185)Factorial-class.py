# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:51:33 2024

@author: Shin
"""
# 데이터를 찾아가서 처리해야 할 떄 재귀 도출 사용 
# 5! = 1*2*3*4*5 =120 
# 양의 정수인 팩토리얼 구하기
# 인자,리턴 = 정수
def factorial(n:int) -> int:
    t = 1
    
    print("{}".format(1))
    
    while(n>1): # 5,4,3,2
        t *= n
        print("* {} = {}".format(n,t))
        n -= 1
        print("* {}".format(1))
    return t
    
if __name__== '__main__': 
    n = int(input('출력할 팩토리얼 값을 입력하세요.:'))
    print(f'{n}의 팩토리얼은 {factorial(n)}입니다.')
    
#%%   
class Factoriral:
    def __init__(self, n:int):
        self.__n = n # 속성: 계산할 팩토리얼 값
    # 내부 메서드     
    def __factorial(self, x:int) -> int:
        if x <= 1:
            return 1
       
        return x * self.__factorial(x-1)
    # 공개 메서드 (변수와 메서드를 분리)
    def compute(self):
        return self.__factorial(self.__n)
    
n=5    
factobj = Factoriral(n)

result = factobj.compute()


#AttributeError: 'Factoriral' object has no attribute '__n'    
# print(f'{n},{factobj.__n}의 팩토리얼은 {result}입니다.')       
# 객체 밖에서 객체 내부 속성에 접근 못함 __를 지정. (비공개 = __)

print(f'{n}의 팩토리얼은 {factorial(n)}입니다.')

    
    