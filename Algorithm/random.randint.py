# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:16:18 2024

@author: Shin
"""
#알고리즘

import random 
n = int(input('난수 개수입력: '))

for _ in range(n): # _ 는 횟수만 0 ~ n-1 처리하겠다는 의미
    r =random.randint(10,99)
    print(r, end = ' ')
    if r == 20:
        print('\n당첨을 축하합니다')
        break
else: #for문에서 break로 종료하지 않으면 처리한다. 
    print('\n난수의 생성을 종료')
    

# itertools.permutation(iterable, r= None) 반복가능한 객체 46개중
# r개를 선택한 순열을 반환하는 함수
import itertools
list(itertools.permutations(range(1,46), 6))
#경우의 수가 너무 많아 출력불가
# for 문을 이용해 permutation함수를 구현할 수 있지만 코드가 길어지기 
# 때문에 permutation()활용

import itertools
lotto = itertools.combinations(range(1,46), 6)
for num in lotto:
    print(num) 
# 반면에 combination(조합)을 활용하면 중복된 6개의 조합이 제외되지만
# itertools.combinations_with_replacement()
# 위는 숫자 중복을 허용하여 조합한 것이다(중복조합)
len(list(itertools.combinations_with_replacement(range(1,46), 6)))
# 15890700 가지의 경우의 수.



     