# https://www.796t.com/content/1547209506.html

import numpy as np
import cv2                #影象處理庫OpenCV
import dlib               #人臉識別庫dlib

#dlib預測器
detector = dlib.get_frontal_face_detector()    #使用dlib庫提供的人臉提取器
predictor = dlib.shape_predictor('D:/github/Image-Processing/Final-Project/shape_predictor_68_face_landmarks.dat')   #構建特徵提取器

# cv2讀取img
img = cv2.imread("face11.png")

# 取灰度
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 人臉數rects
rects = detector(img_gray, 0)
for i in range(len(rects)):
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])  #人臉關鍵點識別
    for idx, point in enumerate(landmarks):        #enumerate函式遍歷序列中的元素及它們的下標
        # 68點的座標
        pos = (point[0, 0], point[0, 1])
        print(idx,pos)

        # 利用cv2.circle給每個特徵點畫一個圈，共68個
        cv2.circle(img, pos, 5, color=(0, 255, 0))
        # 利用cv2.putText輸出1-68
        font = cv2.FONT_HERSHEY_SIMPLEX
        #各引數依次是：圖片，新增的文字，座標，字型，字型大小，顏色，字型粗細
        cv2.putText(img, str(idx+1), pos, font, 0.8, (0, 0, 255), 1,cv2.LINE_AA)

#     #鼻子
#     pos1 = (landmarks[27,0], landmarks[27,1])
#     pos2 = (landmarks[30,0], landmarks[30,1])
#     cv2.line(img,pos1, pos2, (0, 0, 0), 4)
    
#     #眼睛
#     pos3 = (landmarks[36,0], landmarks[36,1])
#     pos4 = (landmarks[39,0], landmarks[39,1])
#     cv2.line(img,pos3, pos4, (0, 0, 0), 4)
    
#     #嘴巴
#     pos5 = (landmarks[42,0], landmarks[42,1])
#     pos6 = (landmarks[45,0], landmarks[45,1])
#     cv2.line(img,pos5, pos6, (0, 0, 0), 4)
    
#     #
#     pos7 = (landmarks[48,0], landmarks[48,1])
#     pos8 = (landmarks[54,0], landmarks[54,1])
#     cv2.line(img,pos7, pos8, (0, 0, 0), 4)

cv2.namedWindow("img", 2)     
cv2.imshow("img", img)       #顯示img
cv2.waitKey(0)        #等待按鍵，後退出
