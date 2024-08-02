# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:26:12 2024

@author: Shin
"""

import cv2

# cap = cv2.VideoCapture("vtest.avi")
# cap = cv2.VideoCapture(0) # Default Camera

cap = cv2.VideoCapture("./images/video2.mp4")

while cap.isOpened(): 
    success, frame = cap.read()
    
    if success:
        cv2.imshow('image', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if (key == 27): # Escape Keyboard
            break
    else:
        break

cap.release() # 카메라 해제

cv2.destroyAllWindows() # 창 닫기