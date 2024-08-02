# 산술 및 논리 연산 예제 (c2_arithmeticLogical.py)

# image augmentation
# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("./images/img1.jpg", cv2.IMREAD_UNCHANGED, 320, 240)
img2 = imgRead("./images/img2.jpg", cv2.IMREAD_UNCHANGED, 320, 240)
img3 = imgRead("./images/img3.jpg", cv2.IMREAD_UNCHANGED, 320, 240)
img4 = imgRead("./images/img4.jpg", cv2.IMREAD_UNCHANGED, 320, 240)
img5 = imgRead("./images/img5.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)

#%%
# 마스크 선언 및 초기화
#%%
print("img5.shape: ", img5.shape) # img5.shape:  (240, 320) 세로,가로

#%%
# Numpy로 full을 통해 0 = 검정색으로 채움 (=검정색으로 채원진 배열)
mask = np.full(shape=img5.shape, fill_value=0, dtype=np.uint8)
h, w = img5.shape

# 흰색으로 채운 이미지 # 100,60 -> 픽셀값 255 ->흰색배경으채워줌 그 외에 바탕 0으로 검정색
x = (int)(w/2) - 60 # 100 # 320/2 - 60
y = (int)(h/2) - 60 # 60 # 240/2 - 60 

cv2.rectangle(mask, (x,y), (x+120, y+120), (255,255,255), -1)  
# mask 이미지 x,y 에서 (x+120, y+120)까지의 사각형을 그립니다
# 사각형은 흰색(255, 255, 255)으로 채워집니다.

# 산술 및 논리 연산 수행
ress = []
ress.append(cv2.add(img1, img2))
ress.append(cv2.addWeighted(img1, 0.5, img2, 0.5, 0)) # 가중치  결과에 0.5씩 더해줌. 
 # cv2.addWeighted() 함수는 두 이미지를 선형 조합하여 새로운 이미지를 생성
 # img1과 img2를 각각 50%의 가중치로 섞은 후, 추가적으로 더해지는 값은 0으로 
 #설정하여 새로운 이미지를 생성하고, 이를 ress 리스트에 추가하는 것

ress.append(cv2.subtract(img3, img4)) 
ress.append(cv2.absdiff(img3, img4)) # 뺸 차의 절대값
ress.append(cv2.bitwise_not(img5))
ress.append(cv2.bitwise_and(img5, mask)) # 반전 + 축소(mask)

# 결과 영상 출력
displays = [("input1", img1),
            ("input2", img2),
            ("input3", img3),
            ("input4", img4),
            ("input5", img5),
            ("res1", ress[0]), # 더함
            ("res2", ress[1]), # 가중치 더함 
            ("res3", ress[2]), # 차
            ("res4", ress[3]), # 차의 절대값 [즉 두 이미지의 차이를 확연하게 식별가능]
            # 원본과 사본 비교할 떄. 
            ("res5", ress[4]), # 반전
            ("res6", ress[5]),] # 반전 +축소
for (name, out) in displays:
    cv2.imshow(name, out)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()

# 영상 저장
save_dir = './code_res_imgs/c2_arithmeticLogical'
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir+"/"+name+".jpg", out)

# cv2.rectangle(img1, (10,10), (100,100), (255,255,255), 1, cv2.LINE_AA, cv2.LINE)

