# 블러링 예제 (c2_blur.py)
# bluring은 필터링을 이용하여 영상을 부드럽게 만드는 기법( 딥러닝에서 잡음제거 및 전처리/ 시간적인 효과를 위해 사용)
# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("./images/img8.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)

# 필터 정의 및 블러링 수행
# size of filter : W * H 

ksize1 = 3; ksize2 = 5; ksize3 = 7; ksize4 = 9

# kernel = np.full(shape=[ksize4,ksize4], fill_value=1, dtype=np.float32) / (ksize4*ksize4)
kernel = np.full(shape=[ksize4,ksize4], fill_value=1, dtype=np.float32) / (ksize4*ksize4*4)
# ksize4 = 9 일떄 위처럼 9*9의 81개의 픽셀공간의 값을 1로 채운 뒤 9*9*4 = 324로 나누어준 값임 [1/324] 
# np.full: NumPy function that creates a new array of a given shape and
# fills it with a specified value.
# shape=[ , ] : shape of the array to be a 2D square matrix with dimensions ksize4 by ksize4
# fill_value=1 : every element in the array should be filled with the value 1.
#dtype=np.float32 : array elements to 32-bit floating-point.


#%%
#blur(src: UMat, ksize: cv2.typing.Size, dst: UMat | None=..., 
#anchor: cv2.typing.Point=..., borderType: int=...)
# src 비트의 입력영상
# dst 입력 영상과 동일한 크기의 자료형을 가지는 출력 영상
# ksize 필터의 크기(size)
# anchor 좌표의 원점을 지정 -1,-1이면 필터의 중심을 원점으로 사용
# boarderType : 영상의 가장 자리를 처리하는 방식 지정(boarder )
# blur() 는 평균값을 구하는 연산을 수행하기에 평균필터(average filter)라고 할 수 있음.
#%%
# res 값은 orignial 원본 픽셀값에 filter 값을 각각 위치에 맞게 곱한 후 더하여 적용됨. 
# p155 참고 ( org* filter를 kernal size에 맞게 실행 후 다 더해준 값이 res임.) 
res1 = cv2.blur(img1, (ksize1,ksize1))
res2 = cv2.blur(img1, (ksize2,ksize2))

res3 = cv2.boxFilter(img1, -1, (ksize3,ksize3))

# 정규화 작업을 실행하지 않을 때 
#res3 = cv2.boxFilter(img1, -1, (ksize3,ksize3), normalize=False)
res4 = cv2.filter2D(img1, -1, kernel)
# blur() 와 동일한 역할을 수행하는 boxFilter() 에서 정규화를 수행한다면 동일한 값을 얻음 .
# normalize 
res5 = cv2.boxFilter(img1, -1, (1,21))

# 결과 영상 출력
displays = [("input1", img1),
            ("res1", res1),
            ("res2", res2),
            ("res3", res3),
            ("res4", res4),
            ("res5", res5)]
for (name, out) in displays:
    cv2.imshow(name, out)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()

# 영상 저장
save_dir = './code_res_imgs/c2_blur'
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir+"/"+name+".jpg", out)