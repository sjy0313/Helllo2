# 샤프닝 예제 (c2_Sharpening.py)
# 샤프닝을 통해 영상을 선명하게 만드는 기법으로 블러링과는 반대의 결과를 가진다
# 블러링은 주변 픽셀과의 픽셀값 차이를 작게 하여 영상을 부드럽게 만든다면 샤프닝은 반대로
# 주변 픽셀값과의 차이를 크게 하여 윤곽을 명확하게 하는 것. 

# 언샤프(Unsharp)는 입력 영상과 블러링 영상간의 차이를 이용
# 블러링을 이용하여 강조해야할 픽셀들의 위치를 계산한 후에는 이들의 픽셀값을 조정하면 언샤프닝이 수행 
# 관련 라이브러리 선언#
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기
img1 = imgRead("./images/img9.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)

# 샤프닝 수행
kernel = np.array(
    [[-1,-1,-1],
     [-1, 9,-1],
     [-1,-1,-1]])
# 중심 픽셀에 대한 가중치를 높이고, 주변 픽셀에 대한 가중치를 줄입니다. 
#이를 통해 중심 픽셀의 영향력을 높이고 주변 픽셀의 영향력을 감소시켜 선명도를 증가시킵니다.

res1 = cv2.filter2D(img1, -1, kernel)

# 언샤프 기법 수행
ksize1 = 3; ksize2 = 15
img1_blur1 = cv2.blur(img1, (ksize1,ksize1))
img1_blur2 = cv2.blur(img1, (ksize2,ksize2))
res2 = cv2.subtract(img1.astype(np.uint16) * 2, img1_blur1.astype(np.uint16))
res3 = cv2.subtract(img1.astype(np.uint16) * 2, img1_blur2.astype(np.uint16))

# res2 = cv2.subtract(img1.astype(np.uint16) , img1_blur1.astype(np.uint16))
# res3 = cv2.subtract(img1.astype(np.uint16) , img1_blur2.astype(np.uint16))

res2 = res2.astype(np.uint8) 
res3 = res3.astype(np.uint8)

# 결과 영상 출력
dif_img1 = cv2.absdiff(img1, img1_blur1)
dif_img2 = cv2.absdiff(img1, img1_blur2)
displays = [("input1", img1),
            ("res1", res1),
            ("res2", res2),
            ("res3", res3),
            ("dif_img1", dif_img1),
            ("dif_img2", dif_img2)]
for (name, out) in displays:
    cv2.imshow(name, out)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()

# 영상 저장
save_dir = './code_res_imgs/c2_Sharpening'
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir+"/"+name+".jpg", out)