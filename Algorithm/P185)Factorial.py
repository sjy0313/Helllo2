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
def factorial(n:int) -> int:
    
    print("[factorial] n: {}".format(n))
    # 5,4,3,2
    if n <= 1: # n:1 
        return 1
    
    return n * factorial(n - 1)

if __name__== '__main__': 
    n = int(input('출력할 팩토리얼 값을 입력하세요.:'))
    print(f'{n}의 팩토리얼은 {factorial(n)}입니다.')
'''
[factorial] n: 5
[factorial] n: 4
[factorial] n: 3
[factorial] n: 2
[factorial] n: 1
5의 팩토리얼은 120입니다.
'''

    
    