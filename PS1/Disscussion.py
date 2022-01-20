# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:16:20 2022

@author: Labuser
"""

import cv2 


img = cv2.imread("southafricaflagface.png")

red_channel = img[:,:,2]
green_channel = img[:,:,1]
blue_channel = img[:,:,0]

cv2.imshow('red_channel', red_channel)
cv2.waitKey(0)

cv2.imshow('green_channel', green_channel)
cv2.waitKey(0)
cv2.imshow('blue_channel', blue_channel)
cv2.waitKey(0)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()