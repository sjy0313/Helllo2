# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:39:58 2024

@author: Shin
"""
# TrackBar

import cv2 
import numpy as np 
def nothing(x):
    pass
# TypeError: 'tuple' object is not callable
#TypeError가 발생한 이유는 함수 nothing()가 정의되었지만 호출되지 않았기 때문입니다.
# 이 함수는 단순히 아무 작업도 수행하지 않기 때문에 이름대로 아무 것도 하지 않습니다.
#오류를 수정하려면 nothing 함수를 변수로 정의하고 사용해야 합니다. 
#함수 이름에 괄호를 사용하지 않고 그냥 변수로 사용하면 됩니다. 
cv2.namedWindow("RGB track bar")
cv2.createTrackbar('Red Color', 'RGB track bar', 0, 255, nothing) # 0 : value / 255 : count
cv2.createTrackbar('Green Color', 'RGB track bar', 0, 255, nothing)
cv2.createTrackbar('Blue Color', 'RGB track bar', 0, 255, nothing)

cv2.setTrackbarPos('Red Color', 'RGB track bar', 125)
cv2.setTrackbarPos('Green Color', 'RGB track bar', 125)
cv2.setTrackbarPos('Blue Color', 'RGB track bar', 125)

img = np.zeros((512,512,3), np.uint8)

while(1): 
    redval = cv2.getTrackbarPos('Red Color', 'RGB track bar')
    greenval = cv2.getTrackbarPos('Green Color', 'RGB track bar')
    blueval = cv2.getTrackbarPos('Blue Color', 'RGB track bar')
    
    cv2.rectangle(img, (0,0), (512,512), (blueval, greenval, redval), -1)
    cv2.imshow('RGB track bar', img)
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키 누르면 종료
        break

cv2.destroyAllWindows()
    