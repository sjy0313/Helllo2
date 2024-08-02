# 이동 변환, 크기 변환 예제 (c2_geometric.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder
#예로는 사이즈 변경(Scaling), 위치변경(Translation), 회전(Rotaion) 등이 있습니다.
# 변환의 종류에는 몇가지 분류가 있습니다.

#강체변환(Ridid-Body) : 크기 및 각도가 보존(ex; Translation, Rotation)
#유사변환(Similarity) : 크기는 변하고 각도는 보존(ex; Scaling)
#선형변환(Linear) : Vector 공간에서의 이동. 이동변환은 제외.
#Affine : 선형변환과 이동변환까지 포함. 선의 수평성은 유지.(ex;사각형->평행사변형)
#Perspective : Affine변환에 수평성도 유지되지 않음. 원근변환

# 영상 읽기
img1 = imgRead("./images/img11.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)

# 이동 변환
h, w, = img1.shape
tlans_x = 10; tlans_y = 25
point1_src = np.float32([[15,20], [50,70], [130,140]]) # 소스위치 
point1_dst = np.float32(point1_src + [tlans_x, tlans_y]) # 타겟 위치 
# 행열연산(matrix operations)
affine_mat1 = cv2.getAffineTransform(point1_src, point1_dst) # affinetransform()활용 
user_mat1 = np.float32([[1,0,tlans_x], [0,1,tlans_y]]) # 행열연산 수행 

# 동일한 결과값 출력함. 
res1 = cv2.warpAffine(img1, affine_mat1, (w,h)) 
res2 = cv2.warpAffine(img1, user_mat1, (w,h))

# 크기 변환
scale_x = 0.8; scale_y = 0.6
background = np.full(shape=[h,w], fill_value=0, dtype=np.uint8)

# cv2.warpAffine(src, M, dsize)
# src – Image
# M – 변환 행렬 [[1,0,x축이동],[0,1,y축이동]] 형태의 float32 type의 numpy array
# dsize (tuple) – output image size(ex; (width=columns, height=rows)
user_mat2 = np.float32([[scale_x,0,0], [0,scale_y,0]])
res3 = cv2.warpAffine(img1, user_mat2, (w,h))
# if w,h = 0 이면 src와 같은 크기로 설정하겠다는 의미
# fx, fy는 영상의 가로 세로 크기를 조정하는 역할을 하는데 

#cv2.resize(img, dsize, fx, fy, interpolation)
# dsize = 출력 영상의 크기 
res4 = cv2.resize(img1, (0,0), None, scale_x, scale_y)
#Scaling은 이미지의 사이즈가 변하는 것 입니다. OpenCV에서는 cv2.resize() 함수를
# 사용하여 적용할 수 있습니다. 사이즈가 변하면 pixel사이의 값을 결정을 해야 하는데
#, 이때 사용하는 것을 보간법(Interpolation method)입니다. 많이 사용되는 보간법은
# 사이즈를 줄일 때는 cv2.INTER_AREA , 사이즈를 크게할 때는 cv2.INTER_CUBIC ,
# cv2.INTER_LINEAR 을 사용합니다. 
# 선형보간법(선형에서 끝과 끝좌표를 알때 사이의 어떤 좌표의 x 좌표만 알떄 
# 끝좌표와의 사이 거리에대한 비율을 통해 y값좌표를 구할 수 있다. 


background[:(int)(h*scale_y), :(int)(w*scale_x)] = res4; res4 = background
                    
# 이동 및 크기 변환
user_mat3 = np.float32([[0.4, 0, 100], [0, 0.6, 50]])
res5 = cv2.warpAffine(img1, user_mat3, (w,h))

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
save_dir = './code_res_imgs/c2_geometric'
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir + "/" + name + ".jpg", out)