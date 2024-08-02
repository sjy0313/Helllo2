#영상 명암비 조절 예제 (c2_contrast.py)
# 극단적으로 명암을 조절하여 밝거나 어둡게만 표현한다면 이진영상과 동일하게 보일 수 있다
# 따라서 명암비 조절의 목적은 밝은 영역과 어두운 영역 간의 밝기 차이를 증가시키면서 동시에 
# 영상에 다양한 밝기 값이 분포하도록 하는 것. 
# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("./images/img5.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)

# 영상 명암비 조절 변수 선언 및 초기화
# fill_value=0 배열의 모든 요소를 초기화 
# dtype=np.uint8 : 배열의 데이터 유형을 8비트 부호 없는 정수
multi_lut = np.full(shape=[256], fill_value=0, dtype=np.uint8) # 1차원배열
log_lut = np.full(shape=[256], fill_value=0, dtype=np.uint8)
invol_lut = np.full(shape=[256], fill_value=0, dtype=np.uint8)
sel_lut = np.full(shape=[256], fill_value=0, dtype=np.uint8)  
#%%
# (자연로그로 밑이 e(2.71828)인 로그)
print(np.log(1+255)) # 5.545177444479562
# GAMMA보정
multi_v = 2; gamma1 = 0.1; gamma2= 0.6
thres1 = 5; thres2 = 100
max_v_log = 255 / np.log(1 + 255) # 로그
max_v_invol = 255 / np.power(255, gamma1) #제곱
max_v_sel = 100 / np.power(thres2, gamma2) # 제곱
#%%
for i in range(256): # 각각의 256개의 테이블에 배열에 값을 넣어줌. 
    val = i * multi_v
    if val > 255 : val = 255
    multi_lut[i] = val
    log_lut[i] = np.round(max_v_log * np.log(1+i))
    invol_lut[i] = np.round(max_v_invol * np.power(i, gamma1))
    if i < thres1 : sel_lut[i] = i
    elif i > thres2 : sel_lut[i] = i
    else: sel_lut[i] =  np.round(max_v_sel * np.power(i, gamma2))

# 명암비 조절
# LUT(Look-up Table)
# Perform a look-up table transform of an array
# 특정 값에 해당하는 결과 값이 저장되어 있는 테이블
# 각 픽셀에 대한 변환연산을 위함. 

ress = []
ress.append(cv2.LUT(img1, multi_lut)) # 상수곱
ress.append(cv2.LUT(img1, log_lut)) # 로그변환
ress.append(cv2.LUT(img1, invol_lut)) # 거듭제곱 변환
ress.append(cv2.LUT(img1, sel_lut)) # 구간변환

# 결과 영상 출력
displays = [("input1", img1),
            ("res1", ress[0]),
            ("res2", ress[1]),
            ("res3", ress[2]),
            ("res4", ress[3])]
for (name, out) in displays:
    cv2.imshow(name, out)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()

# 영상 저장
save_dir = './code_res_imgs/c2_contrast'
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir + "/" + name + ".jpg", out)