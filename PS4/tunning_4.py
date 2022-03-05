#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 20:51:22 2022

@author: labuser
"""

import cv2  
import numpy as np
import ps4
import os

# Utility code
def quiver(u, v, scale, stride, color=(0, 255, 0)):

    img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)

    for y in range(0, v.shape[0], stride):

        for x in range(0, u.shape[1], stride):

            cv2.line(img_out, (x, y),
                     (x + int(u[y, x] * scale), y + int(v[y, x] * scale)),
                     color, 1)
            cv2.circle(img_out,
                       (x + int(u[y, x] * scale), y + int(v[y, x] * scale)), 1,
                       color, 1)
    return img_out
#trackbar callback fucntion does nothing but required for trackbar
def nothing(x):
	pass

#create a seperate window named 'controls' for trackbar
cv2.namedWindow('controls')
#create trackbar in 'controls' window with name 'r''
cv2.createTrackbar('k_size','controls',3,200,nothing)
cv2.createTrackbar('sigma','controls',10,80,nothing)
cv2.createTrackbar('scale','controls',2,20,nothing)
cv2.createTrackbar('stride','controls',5,30,nothing)
cv2.createTrackbar('ktype','controls',0,1,nothing)
cv2.createTrackbar('level','controls',4,6,nothing)

input_dir = "input_images"
output_dir = "./"

# shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'),
#                      0) / 255.
# shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'),
#                        0) / 255.
# shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR20.png'),
#                        0) / 255.
# shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR40.png'),
#                        0) / 255.

urban_img_01 = cv2.imread(os.path.join(input_dir, 'Urban2', 'urban01.png'),
                          0) / 255.
urban_img_02 = cv2.imread(os.path.join(input_dir, 'Urban2', 'urban02.png'),
                          0) / 255.
#create a while loop act as refresh for the view 
while(1):
 
	
    k_size = int(cv2.getTrackbarPos('k_size','controls')) 
    ktype = int(cv2.getTrackbarPos('ktype','controls')) 
    if k_size%2 == 0 :
        k_size +=1
    sigma = int(cv2.getTrackbarPos('sigma','controls'))  # TODO: Select a sigma value if you are using a gaussian kernel
    scale = int(cv2.getTrackbarPos('scale','controls'))
    stride = int(cv2.getTrackbarPos('stride','controls'))
    levels = int(cv2.getTrackbarPos('level','controls'))
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101 
    urban_img_01 = cv2.GaussianBlur(src=urban_img_01,ksize=(k_size,k_size),sigmaX=sigma,sigmaY=sigma)
    urban_img_02 = cv2.GaussianBlur(src=urban_img_02,ksize=(k_size,k_size),sigmaX=sigma,sigmaY=sigma)
    u, v = ps4.hierarchical_lk(urban_img_01, urban_img_02, levels, k_size,
                               "uniform", sigma, interpolation, border_mode,method=1)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    urban_img_02_warped = ps4.warp(urban_img_02, u, v, interpolation,
                                   border_mode)

    diff_img = urban_img_01 - urban_img_02_warped
    #cv2.imwrite(os.path.join(output_dir, "ps4-4-b-2.png"),
    #            ps4.normalize_and_scale(diff_img))

    u_v = quiver(u, v, scale=scale, stride=stride)
    #cv2.imwrite(os.path.join(output_dir, "ps4-1-a-2.png"), u_v)
    cv2.imshow("flow", u_v)
    cv2.imshow("diff",ps4.normalize_and_scale(diff_img).astype(np.uint8))
	
	# waitfor the user to press escape and break the while loop 
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
		
#destroys all window
cv2.destroyAllWindows()