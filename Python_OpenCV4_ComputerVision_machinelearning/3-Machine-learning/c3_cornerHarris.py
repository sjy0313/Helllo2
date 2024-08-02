# 코너 검출 예제 (c3_cornerHarris.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("./images/img_6_4.png", cv2.IMREAD_GRAYSCALE, 320, 240)

# 코너 검지
# 코너점은 두 모서리의 교차점(영상에서 모든 방향으로 밝기 값의 차이가 큰 영역을 의미)

#cornerHarris(src: UMat(8비트 단일채널 영상), blockSize(코너를 찾기 위해 비교하는 이웃픽셀크기): int,
# ksize: int, k(해리스 코너를 계산하기 위해 사용하는 값): float, dst=None,borderType = None)
#dst = cv2.cornerHarris(img1, 2, 3, 0.06) # 코너검출
dst = cv2.cornerHarris(img1, 5, 3, 0.06)
# blocksize를 크게 할 수록 비교하는 이웃 픽셀이 증가하며 이로 인해 밝기 값 변화가 큰 영역이 포함될 
# 가능성이 커짐 -> 그 결과 더 많은 코너를 찾게된다.  
# 위에는 blocksize를 2로 설정하였지만 5로 설정하게 되면 더 많은 코너를 찾을 수 있음. 
dst = cv2.dilate(dst,None) # 팽창(밝은 색 부각, 관심대상 확대)

#%%
res1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
#print(dst.max()) 0.088641316
#resx = res1[dst>0.1*dst.max()]

#%%
res1[dst>0.1*dst.max()] = [0,0,255] # BGR : 빨강색(openCV의 내부구현방식은 BGR순임 not RGB)

# 결과 영상 출력
displays = [("input1", img1),
            ("res1", res1)]
for (name, out) in displays:
    cv2.imshow(name, out)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()

# 영상 저장
save_dir = "./code_res_imgs/c3_cornerHarris"
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir + "/" + name + ".jpg", out)