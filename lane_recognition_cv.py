# 车道线识别
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

plt.ion()  # Turn interactive mode on

image_path='/Users/admin/Downloads/marker-test/GACRT015_1720159303/label（20）/front_wide/lb_1720159319999000000_50_170.jpg'
img=cv.imread(image_path)
img=cv.resize(img,(600,600))  #将原图改为600*600格式大小的，为了后续方便，最后就是在这张600*600的图上画出车道线
cv.imshow('image',img)
cv.waitKey(0)
img0=cv.imread(image_path,0)  #重新以灰度图读取原图，后续找出车道线的处理都在这张图的基础上
img0=cv.resize(img0,(600,600)) #改为600*600大小的
plt.imshow(img0)   #这里做出说明，用plt显示图片是为了顺利找出roi，因为plt显示图片，鼠标指哪里，就会自动显示改点坐标。
plt.show()

ret,img1=cv.threshold(img0,180,255,cv.THRESH_BINARY)  #二值化
cv.namedWindow('Image',0)
cv.imshow("Image",img1) 
cv.waitKey(0)
# cv.destroyAllWindows()

'''2.roi_mask(提取感兴趣的区域)'''
mask=np.zeros_like(img1)   #变换为numpy格式的图片
# mask=cv.fillPoly(mask,np.array([[[0,460],[580,480],[400,15],[330,15]]]),color=255)   #对感兴趣区域制作掩膜
mask=cv.fillPoly(mask,np.array([[[10,400],[590,400],[350,270],[250,270],[10,350]]]),color=255)   #对感兴趣区域制作掩膜 front_wide
#在此做出说明，实际上，车载相机固定于一个位置，所以对于感兴趣的区域的位置也相对固定，这个视相机位置而定。
cv.namedWindow('mask',0)
cv.resizeWindow('mask',800,1200)
cv.imshow('mask',mask)
cv.waitKey(0)

masked_edge_img=cv.bitwise_and(img1,mask)   #与运算
cv.namedWindow('masked_edge_img',0)
cv.resizeWindow('masked_edge_img',800,1200)
cv.imshow('masked_edge_img',masked_edge_img)
cv.waitKey(0)

'剔除小连通域1'
contours, hierarchy = cv.findContours(masked_edge_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)  #找出连通域
# print(len(contours),hierarchy)
for i in range(len(contours)):
    area = cv.contourArea(contours[i])  #将每一个连通域的面积赋值给area
    if area < 2:                         # '设定连通域最小阈值，小于该值被清理'
        cv.drawContours(masked_edge_img, [contours[i]], 0, 0, -1)
cv.namedWindow('masked_edge_img',0)
cv.resizeWindow('masked_edge_img',600,600)
cv.imshow('masked_edge_img',masked_edge_img)
cv.waitKey(0)

img2=masked_edge_img

'找出车道线每个点的坐标，并绘制在原图上'
points=[]    #创建一个空列表，用于存放点的坐标信息
for i in range(600):
    for j in range(600):
        if img2[i,j]!=0:
            img[i,j]=(0,0,255)  #将值不为零的点用红色代替在原图上，注意这里的img是600*600的原图
            points.append([i,j])
cv.imshow('image',img)
cv.waitKey(0)
