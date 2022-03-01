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
def scale_u_and_v(u, v, level, pyr):
    """Scales up U and V arrays to match the image dimensions assigned 
    to the first pyramid level: pyr[0].

    You will use this method in part 3. In this section you are asked 
    to select a level in the gaussian pyramid which contains images 
    that are smaller than the one located in pyr[0]. This function 
    should take the U and V arrays computed from this lower level and 
    expand them to match a the size of pyr[0].

    This function consists of a sequence of ps4.expand_image operations 
    based on the pyramid level used to obtain both U and V. Multiply 
    the result of expand_image by 2 to scale the vector values. After 
    each expand_image operation you should adjust the resulting arrays 
    to match the current level shape 
    i.e. U.shape == pyr[current_level].shape and 
    V.shape == pyr[current_level].shape. In case they don't, adjust
    the U and V arrays by removing the extra rows and columns.

    Hint: create a for loop from level-1 to 0 inclusive.

    Both resulting arrays' shapes should match pyr[0].shape.

    Args:
        u: U array obtained from ps4.optic_flow_lk
        v: V array obtained from ps4.optic_flow_lk
        level: level value used in the gaussian pyramid to obtain U 
               and V (see part_3)
        pyr: gaussian pyramid used to verify the shapes of U and V at 
             each iteration until the level 0 has been met.

    Returns:
        tuple: two-element tuple containing:
            u (numpy.array): scaled U array of shape equal to 
                             pyr[0].shape
            v (numpy.array): scaled V array of shape equal to 
                             pyr[0].shape
    """

    for i in range(level):
        u = 2*ps4.expand_image(u)
        v = 2*ps4.expand_image(v)
    
    #print(u.shape)
    #print(v.shape)
    h,w = pyr[0].shape[:2]
    h_ex,w_ex = u.shape[:2]
    u = cv2.copyMakeBorder(u, 0, h-h_ex, 0, w-w_ex, borderType=cv2.BORDER_CONSTANT, value=0)
    v = cv2.copyMakeBorder(v, 0, h-h_ex, 0, w-w_ex, borderType=cv2.BORDER_CONSTANT, value=0)
    
    return u,v

#trackbar callback fucntion does nothing but required for trackbar
def nothing(x):
	pass

#create a seperate window named 'controls' for trackbar
cv2.namedWindow('controls')
#create trackbar in 'controls' window with name 'r''
cv2.createTrackbar('k_size','controls',3,200,nothing)
cv2.createTrackbar('sigma','controls',10,80,nothing)
cv2.createTrackbar('level','controls',4,30,nothing)
cv2.createTrackbar('ktype','controls',0,1,nothing)

input_dir = "input_images"
output_dir = "./"

yos_img_01 = cv2.imread(
    os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.
yos_img_02 = cv2.imread(
    os.path.join(input_dir, 'DataSeq1', 'yos_img_03.jpg'), 0) / 255.


#create a while loop act as refresh for the view 
while(1):
 
	
    k_size = int(cv2.getTrackbarPos('k_size','controls')) 
    ktype = int(cv2.getTrackbarPos('ktype','controls')) 
    if ktype==0:
        k_type = "uniform"
    else:
        k_type = "gassuian" 
        if k_size%2 == 0 :
            k_size +=1
    sigma = int(cv2.getTrackbarPos('sigma','controls'))  # TODO: Select a sigma value if you are using a gaussian kernel
    scale = int(cv2.getTrackbarPos('scale','controls'))
    level_id = int(cv2.getTrackbarPos('level','controls'))
    
    levels = 8  # Define the number of pyramid levels
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)


    u, v = ps4.optic_flow_lk(yos_img_01_g_pyr[level_id],
                             yos_img_02_g_pyr[level_id], k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_02_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_02_warped = ps4.warp(yos_img_02, u, v, interpolation, border_mode)

    diff_yos_img_01_02 = yos_img_01 - yos_img_02_warped
    output = ps4.normalize_and_scale(diff_yos_img_01_02)
    print(output.mean())
    cv2.imshow("diff",
                output.astype(np.uint8))
	
	# waitfor the user to press escape and break the while loop 
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
		
#destroys all window
cv2.destroyAllWindows()