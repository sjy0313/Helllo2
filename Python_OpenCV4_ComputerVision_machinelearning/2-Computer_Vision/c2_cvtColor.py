# 색상 공간 변환 예제 (c2.cvtColor.py)

# 관련 라이브러리 선언
import numpy as np
import cv2
from imgRead import imgRead
from createFolder import createFolder

# 영상 읽기 : Width = 320 / height = 240 
img1 = imgRead("./images/img7.jpg", cv2.IMREAD_UNCHANGED, 320, 240)
# "./images/img7.jpg" 경로의 이미지를 원본 해상도로 읽습니다.
#%%
# 색상 공간 변환
# 색상(Hue), 체도(Saturation), 명도(Value)의 좌표를 써서 특정한 색을 지정한다
# : BGR 형식의 이미지를 HSV(Hue, Saturation, Value) 형식으로 변환합니다.
res1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

#%%
# 색상 공간 분할 및 병합
#색상(Hue), 채도(Saturation), 명도(Value)는 HSV(Hue, Saturation, Value)
#색상 공간에서 사용되는 세 가지 주요 속성
'''
색상(Hue): 색상은 빛의 파장에 따라 인간이 인지하는 색의 질을 나타냅니다. 
일반적으로 0°에서 360°까지의 범위로 표현되며, 빨강, 노랑, 파랑, 초록 등과 같은 색조를 나타냅니다.

채도(Saturation): 채도는 색의 선명도나 순수도를 나타냅니다. 0%는 회색에 가까운 무채색이며,
 100%는 가장 선명한 색을 나타냅니다. 채도가 높을수록 색상이 더 선명하고 순수하게 보입니다.

명도(Value): 명도는 색의 밝기를 나타냅니다. 0%는 흑색에 가까우며, 100%는 가장 밝은 상태를 나타냅니다.
 명도가 높을수록 색상이 더 밝게 나타납니다.

이러한 세 가지 속성은 색상을 더 정확하게 표현하고, 이미지 처리에서 색상을 조절하거나 분석할 때
유용하게 사용됩니다.'''
 
res1_split = cv2.split(res1) # HSV 이미지를 채널별로 분할 (tuple)

res1_split = list(res1_split) # 튜플을 리스트로 변환 (list)
# TypeError: 'tuple' object does not support item assignment

res1_split[2] = cv2.add(res1_split[2], 100) # 명도 채널에 100을 더합니다.
res1_merge = cv2.merge(res1_split)
res1_merge = cv2.cvtColor(res1_merge, cv2.COLOR_HSV2BGR) # 다시 BGR 형식으로 변환

# 결과 영상 출력
displays = [("input1", img1), # 원본
            ("res1", res1), #HSV
                ("res2", res1_split[0]),  # H  
            ("res3", res1_split[1]), # s 
            ("res4", res1_split[2]), # v
            ("res5", res1_merge)] # 밝기가 조정된 RGB 영상
for (name, out) in displays:
    cv2.imshow(name, out)

# 키보드 입력을 기다린 후 모든 영상창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()

# 영상 저장
save_dir = './code_res_imgs/c2_cvtColor'
createFolder(save_dir)
for (name, out) in displays:
    cv2.imwrite(save_dir+"/"+name+".jpg", out)