# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:05:08 2024

@author: Shin
"""

# 호도법 (반지름r = 호의 길이ㅣ) 비례할 때 (호(호 길이 = 반지름개수)의 길이를 각도로 나타내는 방법)
# 원의둘레 2파이 [1rad = 180 // 파이 = 57.29] 1도 = 파이 // 180  1회전 = 2파이Rad
# 180도 = 파이(radian) 



import cv2
import time
from math import *
import numpy as np

cv2.namedWindow('Clock')

img = np.zeros((512,512,3), np.uint8)
while(True): 
    cv2.circle(img, (256,256), 250, (125,125,125), -1)
    
    now = time.localtime()
    hour = now.tm_hour
    min = now.tm_min
    if hour >12: 
        hour -= 12
        
        # 각도
        # 분 60분 * 6 -> 360도
        # 시 : 12시간 * 30 -> 360도 
    
    Ang_Min = min * 6 # 분 당 6도 씩 회전  # 분 단위 각도
    Ang_Hour = hour*30+min*0.5 # 시간당 30도 씩 회전하며 분이
    # 이동할 떄 마다 시침이 약간 씩 이동하므로(0.5)  # 시간 단위 각도 
    # if Ang_hour = 30 
    if(hour == 12 or 1<=hour<=2):
        x_pos = int(150.0*cos((90.0-Ang_Hour)*3.141592/180)) # 시침의 x좌표(cos)
        y_pos = int(150.0*sin((90.0-Ang_Hour)*3.141592/180)) # 시침의 y좌표(sin)
        # 이미지에 선을 그리는 함수
        cv2.line(img,(256,256), (256+x_pos,256-y_pos), (0,255,0), 6)
        #(256,256)이미지 중앙으로 선의 시작점
        #(256+x_pos,256-y_pos) 선의 끝 점 
        #(0,255,0) rgb값중 green 최대 
        #6 선의 두께
        
        
      
        cv2.imshow('Clock', img)
        
        if cv2.waitKey(10) >=0:  
            break

cv2.destroyAllWindows()    



#%%

import cv2
import numpy as np
import math
import time

def draw_clock(img, center, radius, color, thickness):
    # 시계판 그리기
    cv2.circle(img, center, radius, color, thickness)

    # 시, 분, 초 바늘 그리기
    current_time = time.localtime(time.time())
    second_angle = math.radians((current_time.tm_sec / 60.0) * 360.0 - 90) # 바늘이 12시방향을 기준으로 각도 계산 
    minute_angle = math.radians((current_time.tm_min / 60.0) * 360.0 - 90)
    hour_angle = math.radians(((current_time.tm_hour % 12) / 12.0 * 360.0) - 90)

    # 시침
    hour_length = radius * 0.5 # 시침의 길이 
    hour_x = int(center[0] + hour_length * math.cos(hour_angle))
    hour_y = int(center[1] + hour_length * math.sin(hour_angle))
    cv2.line(img, center, (hour_x, hour_y), color, thickness * 2)

    # 분침
    minute_length = radius * 0.7
    minute_x = int(center[0] + minute_length * math.cos(minute_angle))
    minute_y = int(center[1] + minute_length * math.sin(minute_angle))
    cv2.line(img, center, (minute_x, minute_y), color, thickness)

    # 초침
    second_length = radius * 0.9
    second_x = int(center[0] + second_length * math.cos(second_angle))
    second_y = int(center[1] + second_length * math.sin(second_angle))
    cv2.line(img, center, (second_x, second_y), color, thickness // 2)

# 창 생성
img = np.zeros((400, 400, 3), dtype=np.uint8)
center = (200, 200)
radius = 150
color = (255, 255, 255)
thickness = 2

while True:
    img.fill(0)  # 화면 지우기
    draw_clock(img, center, radius, color, thickness)
    cv2.imshow('Analog Clock', img)
    if cv2.waitKey(1000) & 0xFF == ord('q'):  # 'q'를 누르면 종료
        break

cv2.destroyAllWindows()
