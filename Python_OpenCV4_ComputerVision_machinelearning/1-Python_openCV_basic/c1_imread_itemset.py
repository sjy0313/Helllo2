# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:09:57 2024

@author: Shin
"""

import cv2 as cv
import numpy as np 
image = cv.imread("./images/img_6_0.png")
height = image.shape[0] # 열(행 방향 동작)
width = image.shape[1] # 행(열 방향 동작) 책옆으로 정열한다고 생각



# 세로선 (bgr)
for y in range(0, height): 
    image.itemset(y, int(width/2), 0, 0) # blue
    image.itemset(y, int(width/2), 1, 0) # green
    image.itemset(y, int(width/2), 2, 255) # red 
    
# 가로선
for x in range(0, height): 
    image.itemset(int(height/2), x, 0, 255) # blue
    image.itemset(int(height/2), x, 1, 0)
    image.itemset(int(height/2), x, 2, 0)
    

cv.imshow('win', image)

cv.waitKey(0)
    
    