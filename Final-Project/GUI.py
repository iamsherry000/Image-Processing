import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2

def open_file():
    global imgS,imgO
    filename=filedialog.askopenfilename()     #獲取文件全路徑
    imgS = Image.open(filename)
    imgO = Image.open(filename)
    thumb_size = (670,670)
    imgS.thumbnail(thumb_size)

    print(imgO.size,imgS.size) #初始照片不受影響 ，不能imgS=imgO去指會繼承

    panel = tk.Label(block1)
    panel.photo = ImageTk.PhotoImage(imgS)

    mainImage.config(image=panel.photo) 
    mainImage['height'] = 600
    mainImage['width'] = 670
    
    
window = tk.Tk()
window.title('人臉五官微調系統')
window.geometry('960x700')

div_size = 300
align_mode = 'nswe'
pad0 = 10
pad = 5

#整體布局
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



#window.configure(bg='#FFFFFF')
window.mainloop()