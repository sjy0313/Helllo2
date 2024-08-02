# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:05:24 2024

@author: Shin
"""
#마우스 이벤트 활용 P99
import cv2 
import numpy as np

# GUI(graphic user interface) 특징
def draw_rectangle(event, x, y, flags, parm):
    print(f"x({x}),y({y})")
    if event == cv2.EVENT_LBUTTONDBLCLK: # 왼쪽 마우스 더블 클릭
    # 사각형 도형 출력
        cv2.rectangle(img, (x,y), (x+50, y+50), (0,0,255), -1) # BGR : RED 
        print(f"사각형: x({x}),y({y})")
        
img = np.zeros((512,512,3), np.uint8) # 검정색 바탕 
img = np.ones((512, 512, 3), np.uint8) * 255 # 흰색 바탕 # 모든 채널을 255로 채워야 합니다
cv2.namedWindow('image')

# 마우스 이벤트 리스너 등록
# 이 함수는 사용자가 마우스 이벤트를 발생시킬 때 호출
cv2.setMouseCallback('image', draw_rectangle)

while(1):
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC(버튼눌르면 창 종료) 
    # waitKey() 함수는 사용자의 키 입력을 기다리고 해당 키의 ASCII 값을 반환
    # 이는 8비트 이하의 값을 나타내는 마스크로 사용 0xFF = 255
    # ESC 키의 ASCII 코드는 27
    # 그려지는 이미지 입장에서 좌표의 위치는 왼쪽상단 꼭짓점에서 (0,0) 시작된다.
        break
    
cv2.destroyAllWindows()

#%%
import cv2
import numpy as np

# 마우스 이벤트 콜백 함수
def draw_triangle(event, x, y, flags, param):
    global points, img
    
    # 왼쪽 버튼 클릭 시
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        
        # 세 점을 선택하면 삼각형 그리기
        if len(points) == 3:
            cv2.line(img, points[0], points[1], (255, 0, 0), 2)
            cv2.line(img, points[1], points[2], (255, 0, 0), 2)
            cv2.line(img, points[2], points[0], (255, 0, 0), 2)
           
            
            points = []  # 세 점 초기화

# 이미지 로드
img = np.zeros((512, 512, 3), np.uint8) # unsigned integer(8비트 부호 없는 정수)

# 창 생성 및 마우스 콜백 함수 등록
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_triangle)

points = []  # 선택된 점들을 저장할 리스트

while(1):
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키 누르면 종료
        break

cv2.destroyAllWindows()
#%%
# 오각형
import cv2
import numpy as np

# 마우스 이벤트 콜백 함수
def draw_pentagon(event, x, y, flags, param):
    global points, img
    
    # 왼쪽 버튼 클릭 시
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        
        # 다섯 점을 선택하면 오각형 그리기
        if len(points) == 5:
            for i in range(5):
                cv2.line(img, points[i], points[(i + 1) % 5], (255, 0, 0), 2)
            points = []  # 다섯 점 초기화

# 이미지 로드
img = np.zeros((512, 512, 3), np.uint8)
#img = np.ones((512, 512, 3), np.uint8) * 255 흰색
# 창 생성 및 마우스 콜백 함수 등록
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_pentagon)

points = []  # 선택된 점들을 저장할 리스트

while(1):
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키 누르면 종료
        break

cv2.destroyAllWindows()
