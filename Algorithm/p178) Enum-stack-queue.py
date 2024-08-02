# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:00:52 2024

@author: Shin
"""
#스택과 큐 p 178
from enum import Enum
# Menu라는 type(class)를 정의했다고 보면 됨.

Menu = Enum('Menu', ['인큐', '디큐', '피크', '검색', '덤프', '종료'])

print(Menu.인큐.name) # 인큐
print(Menu.덤프.name) # 덤프

print(Menu.인큐.value) # 1
print(Menu.덤프.value) # 5
# 클라스를 활용하지 않아도 도출가능 p56enum.py 참고 
# Menu 가 정의되지 않아도(Enum을 import해왔기 떄문) 출력됨.
for menu in Menu:
    print("{} : {}".format(menu.name, menu.value))
'''
인큐 : 1
디큐 : 2
피크 : 3
검색 : 4
덤프 : 5
종료 : 6
'''
#%%
import random

def select_menu() -> Menu: # -> 위 함수가 리턴형은 Menu임을 명시논 것으로 
# 
# 별다른 의미는 없음.
    s = [f'({m.value}){m.name}' for m in Menu]
    print(*s, sep='   ') 
    n = random.randint(1,6)
    return Menu(n)
# 인자 : Menu / 리턴 : None 
def print_menu(menu: Menu) -> None: # 소문자 menu형의 인자는 Menu
    print("[print_menu]{} : {}".format(menu.name, menu.value))
    

#%%
menu = select_menu()
print(menu)
print_menu(menu) # [print_menu]피크 : 3
# ['(1)인큐', '(2)디큐', '(3)피크', '(4)검색', '(5)덤프', '(6)종료']

