#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2         # 導入套件: cv2 
import numpy as np # 導入套件: numpy, 並命名成 np

# 定義function: findMask, 輸入為cv hsv_img(HSV格式), 用途為根據給定的HSV 上下界值域進行過濾 從影像中找出特定顏色的mask
def findMask(img):
  
  # 定義 HSV 值域的下界, 詳見投影片中 HSV 範圍圖中的左紅框
  lower_red_0 = np.array([0, 70, 0])
  # 定義 HSV 值域的上界, 詳見投影片中 HSV 範圍圖中的左紅框
  upper_red_0 = np.array([5, 255, 255])
  
  # 定義 HSV 值域的下界, 詳見投影片中 HSV 範圍圖中的右紅框
  lower_red_1 = np.array([175, 70, 0]) 
  # 定義 HSV 值域的上界, 詳見投影片中 HSV 範圍圖中的右紅框
  upper_red_1 = np.array([180, 255, 255])
  
  # 透過第一組左紅框的HSV值域進行影像過濾
  red_mask0 = cv2.inRange(img, lower_red_0, upper_red_0)
  
  # 透過第二組右紅框的HSV值域進行影像過濾
  red_mask1 = cv2.inRange(img, lower_red_1, upper_red_1)
  
  # 將第一組, 第二組紅框值域進行or運算, 將兩者結果合併
  red_mask = cv2.bitwise_or(red_mask0, red_mask1) 
  return red_mask # 回傳結果

### 因沒有main, 程式執行起點為此 ###

# 讀取影像檔案 'red.jpg', 會以cv img格式進行讀取, 預設為 BGR 影像, 檔名 'red.jpg' 表示此程式與red.jpg放在同一路徑下, 如 red.jpg 放在其他地方, 也可以透過絕對路徑來開啟
img = cv2.imread('red.jpg')

# 建立 VideoWriter, 負責執行video的寫入
# fourcc: video的編碼格式, 如 XVID, MP4V 等等...
fourcc = cv2.VideoWriter_fourcc('X',"V",'I','D')
print((img.shape[1], img.shape[0])) # 顯示讀取影像的shape, shape[1] 為影像寬, shape[0] 為影像高
# out: 建立 VideoWriter, video名稱為 test.avi, 寫入格式為 'X',"V",'I','D', FPS 為 20.0, video解析度為 (影像寬, 影像高)
out = cv2.VideoWriter('test.avi', fourcc, 20.0,(img.shape[1], img.shape[0]))

print(img.shape) # 顯示讀取影像的shape, 有些為 (寬, 高), 有些為 (寬, 高, 通道), 單通道為gray_scale影像, 3為BGR影像, 4為 BGRD 影像

# 為了執行HSV過濾, 需先將imread讀取的BGR影像轉換成HSV格式, 使用cvtColor進行轉換, 透過Flag cv2.COLOR_BGR2HSV, 告訴轉換function此圖像為BGR格式, 想要轉換成HSV格式
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 執行 function: findMask, 將 hsv_img 作為 參數輸入, red_mask為 HSV 過濾後的結果
red_mask = findMask(hsv_img) 

print(red_mask.shape) #顯示讀取影像的shape

# 對找出的HSV mask 進行輪廓搜尋: findContours
# findContours 第一個輸入參數為 二值化的mask, 此處可以直接使用我們先前進行HSV過濾的結果
# findContours 第二個輸入參數為 Flag, 可以設定要找最外圍的輪廓 或是連內部輪廓也一起尋找, 此處使用 RETR_EXTERNAL: 找最外圍的輪廓
# findContours 第三個輸入參數為 亦為Flag, 可以設定要找全部的點或是僅找角點, 此處使用 CHAIN_APPROX_NONE: 尋找全部的點
# findContours 會因為cv的版本而有不同的回傳, 如此處有發生問題,　請改成　　_, contour_contours, contour_h = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour_contours, contour_h = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 建立變數 show_image, 用於顯示找到的輪廓與其中心點
# 首先透過 np.zeros建立空的array, shape為讀取影像的大小, 種類為np.uint8
# 再來透過 cv2.cvtColor, 將空array 從 gray_scale 格式轉換成 BGR 格式
show_image = cv2.cvtColor(np.zeros(img.shape[:2], dtype=np.uint8), cv2.COLOR_GRAY2BGR)

# 透過 drawContours, 將 找到的 contour_contours 畫在 show_image上
# -1: 表示目標為 contour_contours 裡面所有的 contour, 如給 0 表示只畫 第一個
# (0, 0, 255): contour的顏色, 此處為 BGR 格式, 每一個值的範圍從0 ~ 255, 0最低, 255最大
# 2: 2表示畫輪廓線大小, -1 表示 塗滿
cv2.drawContours(show_image, contour_contours, -1, (0, 0, 255), 2)

# 計算輪廓中心點, 設計兩個list, 用來儲存 每個輪廓的點 x, y 值
avg_x = []
avg_y = []

# 遍歷每個輪廓, 將x, y 加入到 list 最後面
for cnt in contour_contours:
  for c in cnt:
    avg_x.append(c[0][0])
    avg_y.append(c[0][1])
    
# 透過np.mean 計算平均值
print(np.mean(avg_x))
print(np.mean(avg_y))

# 透過 circle 畫一個圓在show_image
# 圓心為 (int(np.mean(avg_x)), int(np.mean(avg_y))), 亦即所有 x, y的平均, 圓心僅接受int數值
# 1: 1表示圓的半徑
# (175,175,175): 畫圓顏色
# -1: 表示畫圓的外圍大小, -1表示 塗滿
cv2.circle(show_image, (int(np.mean(avg_x)), int(np.mean(avg_y))), 1, (175,175,175), -1)

# 透過 imshow function 進行 圖像顯示, 第一個參數為視窗的名字, 第二個參數為欲顯示的cv img
cv2.imshow("test_center", show_image)

# 透過 imwrite function 進行 影像寫檔, 第一個參數為寫檔名稱, 第二個參數為欲寫入的cv img
cv2.imwrite("test_center.jpg", show_image)
#cv2.waitKey(0) # 如有進行 imshow, 還需要搭配此 function進行視窗刷新才會正常顯示 後面的數字表示刷新頻率, 如0表示顯示後即停止在這, 等到使用者按下 enter或是q 才會再刷新

# 設定 video 影像寫入數量
frame = 600

# 執行 while 迴圈 進行寫檔, 當frame <= 0 就不再寫入
while frame > 0:
  print(frame) #顯示剩餘寫入數量
  show_image = cv2.flip(show_image,0) # 影像翻轉, 僅為讓影像看起來有在動
  out.write(show_image)               # 將cv img 寫入到video中
  cv2.imshow("red_mask", red_mask)    # 顯示 影像, 視窗名稱為red_mask, 欲顯示的cv_img為 red_mask
  cv2.imshow("test", show_image)      # 顯示 影像, 視窗名稱為test, 欲顯示的cv_img為 show_image
  cv2.waitKey(1)                      # 設定視窗刷新頻率 
  frame -= 1                          # 每寫入一次影像就將frame - 1

# 不再寫入video, 透過 release 釋放相關video資源, 結束程式
out.release()