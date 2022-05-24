# Mouse 相關

from PIL import Image, ImageTk
import numpy as np
import cv2
import dlib
from imutils import face_utils

global ShowPoint    # 0:unshow 1:circle 2:number 3:1+2
ShowPoint = 3

global Mouse_start,Mouse_end,border  # 49-67
Mouse_start = 49-1
Mouse_end = 68-1
border = 5


def ShowPoints_func(img):
    
    if ShowPoint == 1 or ShowPoint == 3:       
        # 利用cv2.circle給每個特徵點畫一個圈，共68個
        cv2.circle(img, pos, 5, color=(0, 255, 0))
        
    if ShowPoint == 2 or ShowPoint == 3: 
        # 利用cv2.putText輸出1-68
        font = cv2.FONT_HERSHEY_SIMPLEX    
        # 各引數依次是：圖片，新增的文字，座標，字型，字型大小，顏色，字型粗細
        cv2.putText(img, str(idx+1), pos, font, 0.8, (0, 0, 255), 1,cv2.LINE_AA)
        
    

def rection(img): # OpenCV 測68個點   
    global pos,idx,point,rects        
    #dlib預測器
    detector = dlib.get_frontal_face_detector()    #使用dlib庫提供的人臉提取器
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')   #構建特徵提取器
    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 人臉數rects
    rects = detector(img_gray, 0)
    
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])  #人臉關鍵點識別
        for idx, point in enumerate(landmarks):        #enumerate函式遍歷序列中的元素及它們的下標
            # 68點的座標
            pos = (point[0, 0], point[0, 1])
            # print(idx,pos) 
            ShowPoints_func(img)
    return img
    
def main():
    imgS = cv2.imread("face11.png") # 用opencv的方法
    rection(imgS)
    cv2.imshow("test",imgS)  # 顯示圖片
    cv2.waitKey(0)  # 按任意鍵退出
    cv2.destroyAllWindows()  # 關閉window
    
main()
