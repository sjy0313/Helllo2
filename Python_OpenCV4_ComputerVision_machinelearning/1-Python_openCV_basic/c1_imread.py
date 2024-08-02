# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:26:53 2024

@author: Shin
"""
import cv2

print(cv2.__version__) # 4.9.0

#%%

img = cv2.imread("./images/img_6_3.png") # 이미지를 파일에서 읽어라 
cv2.namedWindow("image") # 이름(image) 가진 윈도우 생성 
cv2.imshow("image", img) # 윈도우(image)에 이름에 이미지를 그려라
cv2.waitKey() # 아무키가 입력될 떄까지 기다려라

cv2.destroyAllWindows() # 모든 열려있는 파일 닫기
