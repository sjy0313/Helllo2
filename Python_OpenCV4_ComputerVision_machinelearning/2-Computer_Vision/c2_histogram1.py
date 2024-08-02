#히스토그램 계산 예제 (c2_histogram1.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from matplotlib import pyplot as plt
from imgRead import imgRead
from createFolder import createFolder
#%%
# 영상 읽기 [회색으로 (GRAYSCALE) 읽어옴]
img1 = imgRead("./images/img5.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)
# img1 보기
cv2.imshow("img5", img1)  
cv2.waitKey(0)
cv2.destroyAllWindows()    
#%%        

# 히스토그램 계산 [이미지를 0~255 봤을 떄 색상의 분포]
# 0~20 즉 어두운 부분의 비중이 600픽셀이상 드러남. 
# GRAYSCALE로 읽어옴에 따라 데이터크기 감소 + 처리속도 향상(컬러정보 필요성 부재)
# 모든 채널을 단일 채널로 설정
ch1 = [0]; ch2 = [0]; ch3 = [0]
#이 코드에서는 cv2.calcHist() 함수를 사용하여 이미지의 
#각 채널을 개별적으로 처리하기보다는 그레이스케일 이미지로 가정하고 
#모든 픽셀 값이 단일 채널에 있도록 하여 히스토그램을 계산합니다.
#단일 채널에 모든 픽셀 값이 존재하기 떄문에 연산속도가 빨라짐.
ranges1 = [0, 256]; 
ranges2 = [0, 128]; # 빨강
ranges3 = [128, 256] # 초록
histSize1 = [256]; histSize2 = [128]; histSize3 = [128]
# 채널에서 특정 값 범위에 대하여 총 빈 개수 계산
hist1 = cv2.calcHist([img1], ch1, None, histSize1, ranges1) 
hist2 = cv2.calcHist([img1], ch2, None, histSize2, ranges2)
hist3 = cv2.calcHist([img1], ch3, None, histSize3, ranges3)
#%%
# 전체 빈도수 = 총 픽셀 수 
tot1 = hist1.sum()
tot2 = hist2.sum() + hist3.sum()
print('shape:', img1.shape[0] * img1.shape[1]) # shape: 76800

#%%

# 히스토그램 출력 및 저장
bin_x1 = np.arange(256)
bin_x2 = np.arange(128)
bin_x3 = np.arange(128) + 128
# 그래프 기법 
plt.title("Histogram")
plt.xlabel("Bin")
plt.ylabel("Frequency")
plt.plot(bin_x1, hist1, color='b') # 선차트 
plt.bar(bin_x2, hist2[:,0], width=6,color='r') # 바차트 : 초록
plt.bar(bin_x3, hist3[:,0], width=6,color='g') # 바차트 : 빨강
plt.grid(True, lw = 1, ls = '--', c= '.75') 
plt.xlim([0,255])

# 영상 저장
save_dir = './code_res_imgs/c2_histogram1'
createFolder(save_dir)
plt.savefig(save_dir + "/" + "hist.png")

plt.show()