# GUI + 臉部68個偵測點(success)

# think: (solved)
    # 調整image大小能不能opencv完成? (solved)
    # 不行的話，PIL轉成opencv?
# notice: 每次show圖片之前要delete all
# Reference: https://www.796t.com/content/1547209506.html

import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import cv2
import dlib

global imgS,img_show,panel

# (對68個點要變動的程式碼，每對imgS做變動，都要呼叫Renew()函式更新)

def OpenCV_small():  # opencv 等比縮小圖片
    global imgS    
    nr,nc = imgS.shape[:2]
    if nr >= 670 or nc >= 670:
        if nr>= nc:
            scale = 670/nr 
        else:
            scale = 670/nc
        nr2 = int(nr * scale)
        nc2 = int(nc * scale)
        imgS = cv2.resize(imgS,(nc2,nr2),interpolation = cv2.INTER_LINEAR)

def open_file():
    global imgS,img_show,panel
    filename=filedialog.askopenfilename()  #獲取文件全路徑
    imgS = cv2.imread(filename) # 用opencv的方法    
    OpenCV_small() # opencv 等比縮小圖片
    
    # Label改成Canvas，不要用Label因為會有覆寫問題
    panel = tk.Canvas(block1,width=650,height=600) # 設定長寬   
    imgS = rection(imgS)  # imgS是rection後的opencv圖    
    Renew() # 更新

def Renew():
    global imgS,img_show,panel
    # 把img_show變成PIL格式圖
    img_show = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    img_show = Image.fromarray(img_show)
    img_show = ImageTk.PhotoImage(image = img_show)
    # 要更新
    panel.delete("all") 
    panel.create_image(0,0,image=img_show,anchor=NW)
    panel.place(x=0,y=0)

def rection(img): # OpenCV 測68個點
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
            print(idx,pos)

            # 利用cv2.circle給每個特徵點畫一個圈，共68個
            #cv2.circle(img, pos, 5, color=(0, 255, 0))
            # 利用cv2.putText輸出1-68
            #font = cv2.FONT_HERSHEY_SIMPLEX
            #各引數依次是：圖片，新增的文字，座標，字型，字型大小，顏色，字型粗細
            #cv2.putText(img, str(idx+1), pos, font, 0.8, (0, 0, 255), 1,cv2.LINE_AA)
    return img
    
window = tk.Tk()
window.title('人臉五官微調系統')
window.geometry('960x700')

div_size = 300
align_mode = 'nswe'
pad0 = 10
pad = 5

#GUI整體布局
blocktop = tk.Frame(window, width=650, height=30)
blocktop2 = tk.Frame(window, width=div_size, height=30)
block1 = tk.Frame(window, width=650, height=600,bg='#FFFFFF')
block2 = tk.Frame(window, width=div_size, height=250)
block3 = tk.Frame(window, width=div_size, height=250)
block4 = tk.Frame(window, width=div_size, height=250)

blocktop.grid(column=0, row=0, sticky=align_mode)
blocktop2.grid(column=1, row=0, sticky=align_mode)
block1.grid(column=0, row=1, rowspan=4, sticky=align_mode) 
block2.grid(column=1, row=2, sticky=align_mode)
block3.grid(column=1, row=3, sticky=align_mode)
block4.grid(column=1, row=4, sticky=align_mode)


#拉霸區塊布局
eyelabel = tk.Label(block2,text="眼睛",font=('新細明體', 12),padx=pad, pady=pad,fg='#007799')
font = ('Courier New', 20, 'bold')
eyescale1 = tk.Scale(
    block2, label='大小', from_=0, to=100, orient="horizontal",tickinterval=10,length=280)
eyescale2 = tk.Scale(
    block2, label='旋轉', from_=0, to=100, orient="horizontal",tickinterval=10,length=280)
eyescale3 = tk.Scale(
    block2, label='眼距', from_=0, to=100, orient="horizontal",tickinterval=10,length=280)

eyelabel.grid(row=0, column=0)
eyescale1.grid(row=1, column=0)
eyescale2.grid(row=2, column=0)
eyescale3.grid(row=3, column=0)

noselabel = tk.Label(block3,text="鼻子",font=('新細明體', 12),padx=pad, pady=pad,fg='#007799')
nosescale1 = tk.Scale(
    block3, label='大小', from_=0, to=100, orient="horizontal",tickinterval=10,length=280)
nosescale2 = tk.Scale(
    block3, label='鼻翼', from_=0, to=100, orient="horizontal",tickinterval=10,length=280)

noselabel.grid(row=0, column=0)
nosescale1.grid(row=1, column=0)
nosescale2.grid(row=2, column=0)

mouselabel = tk.Label(block4,text="嘴巴",font=('新細明體', 12),padx=pad, pady=pad,fg='#007799')
mousescale1 = tk.Scale(
    block4, label='大小', from_=0, to=100, orient="horizontal",tickinterval=10,length=280)
mousescale2 = tk.Scale(
    block4, label='薄厚', from_=0, to=100, orient="horizontal",tickinterval=10,length=280)

mouselabel.grid(row=0, column=0)
mousescale1.grid(row=1, column=0)
mousescale2.grid(row=2, column=0)


#開檔&存檔
import_btn = tk.Button(blocktop, text='開啟檔案', bg='#BBFFEE', fg='black',height = 1, 
          width = 20, command=open_file)
save_btn = tk.Button(blocktop, text="儲存檔案", bg='#BBFFEE', fg='black',height = 1, 
          width = 20)
import_btn.grid(column=0, row=0, padx=pad0, pady=pad0, sticky=align_mode)
save_btn.grid(column=1, row=0, padx=pad0, pady=pad0, sticky=align_mode)

#顯示照片
mainImage=tk.Label(block1,height = 1, width = 95, image=None,bg='#FFFFFF') 
mainImage.grid(row=0, column=0, sticky=align_mode)


window.mainloop()