# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 20:46:45 2018

@author: sameer
"""

from matplotlib import pyplot as plt
import cv2 # used for resize. if you dont have it, use anything else
import numpy as np
from model import Deeplabv3


deeplab_model = Deeplabv3()

img = plt.imread("imgs/single1.jpg")
w, h, _ = img.shape
ratio = 512. / np.max([w,h])
resized = cv2.resize(img,(int(ratio*h),int(ratio*w)))
resized = resized / 127.5 - 1.
pad_x = int(512 - resized.shape[0])
resized2 = np.pad(resized,((0,pad_x),(0,0),(0,0)),mode='constant')
cv2.imshow('resize image', resized2)
res = deeplab_model.predict(np.expand_dims(resized2,0))
labels = np.argmax(res.squeeze(),-1)
plt.imshow(labels[:-pad_x])

kk=labels[:-pad_x]
plt.imsave('seg_map.jpg', kk )
image=cv2.imread('seg_map.jpg')
blank_image=np.zeros((image.shape[0],image.shape[1],3))
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray image",gray)
ret,threshold=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#cv2.imshow("threshold image",threshold)
th,contours,hierarchy=cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
ima=cv2.drawContours(image,contours,-1,(0,0,255),6)
imab=cv2.drawContours(blank_image,contours,-1,(0,0,255),6)
imabc=cv2.drawContours(resized2,contours,-1,(0,0,255),6)

#cv2.imshow("contourss",ima)
#cv2.imshow("contourson blank image",imab)
cv2.imshow("Outline",imabc)
plt.imsave('Outline.png', imabc)
#cv2.imwrite('finalcv.jpg',imabc)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imwrite('contour2.jpg',imag_cnt)
#print("done saving----- image------------")