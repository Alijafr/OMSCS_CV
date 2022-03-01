"""Problem Set 4: Motion Detection"""

import os

import cv2
import numpy as np

import ps4

# I/O directories
input_dir = "input_images"
output_dir = "./"

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


# Functions you need to complete:


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
    u = cv2.copyMakeBorder(u, 0, h-h_ex, 0, w-w_ex, borderType=cv2.BORDER_REFLECT)
    v = cv2.copyMakeBorder(v, 0, h-h_ex, 0, w-w_ex, borderType=cv2.BORDER_REFLECT)
    
    return u,v


def part_1a():

    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'),
                         0) / 255.
    shift_r2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR2.png'),
                          0) / 255.
    shift_r5_u5 = cv2.imread(
        os.path.join(input_dir, 'TestSeq', 'ShiftR5U5.png'), 0) / 255.

    # Optional: smooth the images if LK doesn't work well on raw images
    k_size = 47  # TODO: Select a kernel size
    k_type = "gassuian"  # TODO: Select a kernel type
    sigma = 33  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r2, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=5, stride=19)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-1.png"), u_v)

    # Now let's try with ShiftR5U5. You may want to try smoothing the
    # input images first.

    k_size = 151  # TODO: Select a kernel size
    k_type = "gassuian"  # TODO: Select a kernel type
    sigma = 23  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r5_u5, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=12, stride=19)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-2.png"), u_v)


def part_1b():
    """Performs the same operations applied in part_1a using the images
    ShiftR10, ShiftR20 and ShiftR40.

    You will compare the base image Shift0.png with the remaining
    images located in the directory TestSeq:
    - ShiftR10.png
    - ShiftR20.png
    - ShiftR40.png

    Make sure you explore different parameters and/or pre-process the
    input images to improve your results.

    In this part you should save the following images:
    - ps4-1-b-1.png
    - ps4-1-b-2.png
    - ps4-1-b-3.png

    Returns:
        None
    """
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'),
                         0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'),
                           0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR20.png'),
                           0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR40.png'),
                           0) / 255.
    k_size = 133  # TODO: Select a kernel size
    k_type = "gassuian"  # TODO: Select a kernel type
    sigma = 32  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r10, k_size, k_type, sigma)
    
    # Flow image
    u_v = quiver(u, v, scale=12, stride=14)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-1.png"), u_v)
    
    k_size = 129  # TODO: Select a kernel size
    k_type = "gassuian"  # TODO: Select a kernel type
    sigma = 31  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r20, k_size, k_type, sigma)
    
    # Flow image
    u_v = quiver(u, v, scale=20, stride=15)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-2.png"), u_v)
    
    k_size = 127  # TODO: Select a kernel size
    k_type = "gassuian"  # TODO: Select a kernel type
    sigma = 28  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r40, k_size, k_type, sigma)
    
    # Flow image
    u_v = quiver(u, v, scale=20, stride=15)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-3.png"), u_v)
    
    
    
    


def part_2():

    yos_img_01 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.

    # 2a
    levels = 4
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_01_g_pyr_img = ps4.create_combined_img(yos_img_01_g_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-a-1.png"),
                yos_img_01_g_pyr_img)

    # 2b
    yos_img_01_l_pyr = ps4.laplacian_pyramid(yos_img_01_g_pyr)

    yos_img_01_l_pyr_img = ps4.create_combined_img(yos_img_01_l_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-b-1.png"),
                yos_img_01_l_pyr_img)


def part_3a_1():
    yos_img_01 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.

    levels = 5  # Define the number of pyramid levels
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)

    level_id = 2  # TODO: Select the level number (or id) you wish to use
    k_size = 75  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = 17  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_01_g_pyr[level_id],
                             yos_img_02_g_pyr[level_id], k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_02_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_02_warped = ps4.warp(yos_img_02, u, v, interpolation, border_mode)
    
    cv2.imwrite(os.path.join(output_dir, "ps4-4_warp2.png"),
                ps4.normalize_and_scale(yos_img_02_warped))
    diff_yos_img_01_02 = yos_img_01 - yos_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-1.png"),
                ps4.normalize_and_scale(diff_yos_img_01_02))


def part_3a_2():
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.
    yos_img_03 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_03.jpg'), 0) / 255.

    levels = 4  # Define the number of pyramid levels
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)
    yos_img_03_g_pyr = ps4.gaussian_pyramid(yos_img_03, levels)

    level_id = 2  # TODO: Select the level number (or id) you wish to use
    k_size = 75  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = 17  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_02_g_pyr[level_id],
                             yos_img_03_g_pyr[level_id], k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_03_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_03_warped = ps4.warp(yos_img_03, u, v, interpolation, border_mode)
    
    cv2.imwrite(os.path.join(output_dir, "ps4-4_warp3.png"),
                ps4.normalize_and_scale(yos_img_03_warped))
    
    diff_yos_img = yos_img_02 - yos_img_03_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-2.png"),
                ps4.normalize_and_scale(diff_yos_img))


def part_4a():
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'),
                         0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'),
                           0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR20.png'),
                           0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR40.png'),
                           0) / 255.

    levels = 4  # TODO: Define the number of levels
    k_size = 15  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = 2  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u10, v10 = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u10, v10, scale=1, stride=14)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-1.png"), u_v)

    # You may want to try different parameters for the remaining function
    # calls.
    u20, v20 = ps4.hierarchical_lk(shift_0, shift_r20, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u20, v20, scale=1, stride=14)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-2.png"), u_v)

    u40, v40 = ps4.hierarchical_lk(shift_0, shift_r40, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)
    u_v = quiver(u40, v40, scale=1, stride=14)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-3.png"), u_v)


def part_4b():
    urban_img_01 = cv2.imread(os.path.join(input_dir, 'Urban2', 'urban01.png'),
                              0) / 255.
    urban_img_02 = cv2.imread(os.path.join(input_dir, 'Urban2', 'urban02.png'),
                              0) / 255.

    levels = 4  # TODO: Define the number of levels
    k_size = 15  # TODO: Select a kernel size
    k_type = ""  # TODO: Select a kernel type
    sigma = 2  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u, v = ps4.hierarchical_lk(urban_img_01, urban_img_02, levels, k_size,
                               k_type, sigma, interpolation, border_mode)

    u_v = quiver(u, v, scale=1, stride=14)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-1.png"), u_v)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    urban_img_02_warped = ps4.warp(urban_img_02, u, v, interpolation,
                                   border_mode)

    diff_img = urban_img_01 - urban_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-2.png"),
                ps4.normalize_and_scale(diff_img))


def part_5a():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'),
                         0) / 255.
    shift_10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'),
                         0) / 255.
    
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u, v = ps4.hierarchical_lk(shift_0, shift_10, levels=4, k_size=21, k_type="gaussian",
                                   sigma=10, interpolation=interpolation, border_mode=border_mode)
    shift_02 = np.copy(shift_0)
    shift_04 = np.copy(shift_0)
    shift_06 = np.copy(shift_0)
    shift_08 = np.copy(shift_0)
    
    
    for i in range(shift_0.shape[0]):
        for j in range(shift_0.shape[1]):
            shift_02[np.clip(i+int(0.2*v[i,j]), 0, shift_0.shape[0]-1),np.clip(j+int(0.2*u[i,j]), 0, shift_0.shape[1]-1)] = (1-0.2)*shift_0[i,j] + 0.2*shift_10[i,j]
            shift_04[np.clip(i+int(0.4*v[i,j]), 0, shift_0.shape[0]-1),np.clip(j+int(0.4*u[i,j]), 0, shift_0.shape[1]-1)] = (1-0.4)*shift_0[i,j] + 0.4*shift_10[i,j]
            shift_06[np.clip(i+int(0.6*v[i,j]), 0, shift_0.shape[0]-1),np.clip(j+int(0.6*u[i,j]), 0, shift_0.shape[1]-1)] = (1-0.6)*shift_0[i,j] + 0.6*shift_10[i,j]
            shift_08[np.clip(i+int(0.8*v[i,j]), 0, shift_0.shape[0]-1),np.clip(j+int(0.8*u[i,j]), 0, shift_0.shape[1]-1)] = (1-0.8)*shift_0[i,j] + 0.8*shift_10[i,j]
    
    output_image = np.zeros((2*shift_0.shape[0],3*shift_0.shape[1]))
    h,w = shift_0.shape[:2]
    
    output_image[:h,:w]=shift_0
    output_image[:h,w:2*w]=shift_02
    output_image[:h,2*w:]=shift_04
    output_image[h:,:w]=shift_06
    output_image[h:,w:2*w]=shift_08
    output_image[h:,2*w:]=shift_10
    
    
    cv2.imwrite(os.path.join(output_dir, "ps4-5-a-1.png"), ps4.normalize_and_scale(output_image))
    
    


def part_5b():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    raise NotImplementedError


def part_6():
    """Challenge Problem

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    images_running_1 = ps4.read_video(os.path.join(input_dir, "videos/person01_running_d2_uncomp.avi"))
    images_running_2 = ps4.read_video(os.path.join(input_dir, "videos/person08_running_d4_uncomp.avi"))
    images_walking_1 = ps4.read_video(os.path.join(input_dir, "videos/person04_walking_d4_uncomp.avi"))
    images_walking_2 = ps4.read_video(os.path.join(input_dir, "videos/person06_walking_d1_uncomp.avi"))
    images_handclapping_1 = ps4.read_video(os.path.join(input_dir, "videos/person10_handclapping_d4_uncomp.avi"))
    images_handclapping_2 = ps4.read_video(os.path.join(input_dir, "videos/person14_handclapping_d3_uncomp.avi"))
    

def nothing(x):
	pass

if __name__ == '__main__':
    
    #part_1a()
    #part_1b()
    #part_2()
    #part_3a_1()
    #part_3a_2()
    #part_4a()
    part_4b()
    part_5a()
    # part_5b()
    # part_6()
