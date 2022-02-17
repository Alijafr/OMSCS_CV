
"""
CS6476: Problem Set 3 Experiment file
This script consists  a series of function calls that run the ps3
implementation and output images so you can verify your results.
"""

import os
import numpy as np
import cv2
import copy
import PIL
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

from ps3 import *
import ps3

IMG_DIR = "input_images"
VID_DIR = "input_videos"
OUT_DIR = "./"
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)


def helper_for_part_4_and_5(video_name, fps, frame_ids, output_prefix,
                            counter_init, is_part5):

    video = os.path.join(VID_DIR, video_name)
    image_gen = ps3.video_frame_generator(video)

    image = image_gen.__next__()
    h, w, d = image.shape

    out_path = "ar_{}-{}".format(output_prefix[4:], video_name)
    video_out = mp4_video_writer(out_path, (w, h), fps)

    # Optional template image
    template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))

    if is_part5:
        advert = cv2.imread(os.path.join(IMG_DIR, "img-3-a-1.png"))
        src_points = ps3.get_corners_list(advert)

    output_counter = counter_init

    frame_num = 1

    while image is not None:

        print("Processing fame {}".format(frame_num))

        markers = ps3.find_markers(image, template)

        if is_part5:
            # if len(set(markers))== 4: # make sure there are 4 unique markers 
            #     print(markers)
            homography = ps3.find_four_point_transform(src_points, markers)
            image = ps3.project_imageA_onto_imageB(advert, image, homography)

        else:
            
            for marker in markers:
                mark_location(image, marker)

        frame_id = frame_ids[(output_counter - 1) % 3]

        if frame_num == frame_id:
            out_str = output_prefix + "-{}.png".format(output_counter)
            save_image(out_str, image)
            output_counter += 1

        video_out.write(image)

        image = image_gen.__next__()

        frame_num += 1

    video_out.release()



def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


def save_image(filename, image):
    """Convenient wrapper for writing images to the output directory."""
    cv2.imwrite(os.path.join(OUT_DIR, filename), image)


def mark_location(image, pt):
    """Draws a dot on the marker center and writes the location as text nearby.

    Args:
        image (numpy.array): Image to draw on
        pt (tuple): (x, y) coordinate of marker center
    """
    color = (0, 50, 255)
    cv2.circle(image, pt, 3, color, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "(x:{}, y:{})".format(*pt), (pt[0]+15, pt[1]), font, 0.5, color, 1)


def part_1():

    print("\nPart 1:")

    input_images = ['sim_clear_scene.jpg', 'sim_noisy_scene_1.jpg',
                    'sim_noisy_scene_2.jpg']
    output_images = ['ps3-1-a-1.png', 'ps3-1-a-2.png', 'ps3-1-a-3.png']

    # Optional template image
    template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))

    for img_in, img_out in zip(input_images, output_images):

        print("Input image: {}".format(img_in))

        # Open image and identify the four marker positions
        scene = cv2.imread(os.path.join(IMG_DIR, img_in))

        marker_positions = ps3.find_markers(scene, template)

        for marker in marker_positions:
            mark_location(scene, marker)

        save_image(img_out, scene)


def part_2():

    print("\nPart 2:")

    input_images = ['ps3-2-a_base.jpg', 'ps3-2-b_base.jpg',
                    'ps3-2-c_base.jpg', 'ps3-2-d_base.jpg', 'ps3-2-e_base.jpg']
    output_images = ['ps3-2-a-1.png', 'ps3-2-a-2.png', 'ps3-2-a-3.png',
                     'ps3-2-a-4.png', 'ps3-2-a-5.png']

    # Optional template image
    template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))

    for img_in, img_out in zip(input_images, output_images):

        print("Input image: {}".format(img_in))

        # Open image and identify the four marker positions
        scene = cv2.imread(os.path.join(IMG_DIR, img_in))

        markers = ps3.find_markers(scene, template)
        image_with_box = ps3.draw_box(scene, markers, 3)

        save_image(img_out, image_with_box)


def part_3():

    print("\nPart 3:")

    input_images = ['ps3-3-a_base.jpg', 'ps3-3-b_base.jpg', 'ps3-3-c_base.jpg']
    output_images = ['ps3-3-a-1.png', 'ps3-3-a-2.png', 'ps3-3-a-3.png']

    # Advertisement image
    advert = cv2.imread(os.path.join(IMG_DIR, "img-3-a-1.png"))
    src_points = ps3.get_corners_list(advert)

    # Optional template image
    template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))

    for img_in, img_out in zip(input_images, output_images):
        print("Input image: {}".format(img_in))

        # Open image and identify the four marker positions
        scene = cv2.imread(os.path.join(IMG_DIR, img_in))

        markers = ps3.find_markers(scene, template)

        homography = ps3.find_four_point_transform(src_points, markers)

        projected_img = ps3.project_imageA_onto_imageB(advert, scene,
                                                       homography)

        save_image(img_out, projected_img)


def part_4_a():

    print("\nPart 4a:")

    video_file = "ps3-4-a.mp4"
    frame_ids = [355, 555, 725]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-4-a", 1, False)

    video_file = "ps3-4-b.mp4"
    frame_ids = [97, 407, 435]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-4-a", 4, False)


def part_4_b():

    print("\nPart 4b:")

    video_file = "ps3-4-c.mp4"
    frame_ids = [47, 470, 691]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-4-b", 1, False)

    video_file = "ps3-4-d.mp4"
    frame_ids = [207, 367, 737]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-4-b", 4, False)


def part_5_a():

    print("\nPart 5a:")

    video_file = "ps3-4-a.mp4"
    frame_ids = [355, 555, 725]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-5-a", 1, True)

    video_file = "ps3-4-b.mp4"
    frame_ids = [97, 407, 435]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-5-a", 4, True)


def part_5_b():

    print("\nPart 5b:")

    video_file = "ps3-4-c.mp4"
    frame_ids = [47, 470, 691]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-5-b", 1, True)

    video_file = "ps3-4-d.mp4"
    frame_ids = [207, 367, 737]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-5-b", 4, True)


    

def part_6(path1, path2):
    Mouse_Click_Correspondence().driver(path1, path2)


def part_7():
    ###load previously obtained points
    sxx1, sxx2, syy1, syy2 = [], [], [], []
    c1 = np.load('p1.npy')
    c2 = np.load('p2.npy')

    for cc1, cc2 in zip(c1, c2):
        sxx1.append(cc1[0])
        sxx2.append(cc2[0])
        syy1.append(cc1[1])
        syy2.append(cc2[1])

    sx1, sy1 = sxx1, syy1
    sx2, sy2 = sxx2, syy2

    points1, points2 = [], []
    for x, y in zip(sx1, sy1):
        points1.append((x, y))

    for x, y in zip(sx2, sy2):
        points2.append((x, y))

    #TODO - Use the previously implemented four point transform to obtain the 3x3 homography matrix by passing in the source and destination points to the function. Ensure
    #that the order of points1 and points2 correspond accurately to your implementation.
    homography_parameters = ps3.find_four_point_transform(points2,points1)

    return homography_parameters




def part_9_a(homography_parameters, path1, path2):

    im_src = cv2.imread(path1, 1)
    im_dst = cv2.imread(path2, 1)

    cv2.namedWindow("Source Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Destination Image", cv2.WINDOW_NORMAL)

    cv2.imshow("Source Image", im_src)
    cv2.imshow("Destination Image", im_dst)
    
    
    im_warped = Image_Mosaic().image_warp_inv(im_src, im_dst, homography_parameters)


    cv2.namedWindow("Warped Source Image", cv2.WINDOW_NORMAL)
    cv2.imwrite(IMG_DIR + '\warped_image_a.jpg', im_warped)
    im_warped = cv2.normalize(im_warped, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    cv2.imshow("Warped Source Image", im_warped)

    im_out_mosaic = Image_Mosaic().output_mosaic(im_src, im_warped)

    cv2.namedWindow("Output Mosaic Image", cv2.WINDOW_NORMAL)
    cv2.imwrite(IMG_DIR + '\image_mosiac_a.jpg', im_out_mosaic)
    cv2.imshow("Output Mosaic Image", im_out_mosaic)

    cv2.waitKey(0)





if __name__ == '__main__':
    print("--- Problem Set 3 ---")
    # Comment out the sections you want to skip

    # part_1()
    # part_2()
    part_3()  
    # part_4_a()
    # part_4_b()
    # part_5_a()
    # part_5_b()
    
    # path1 = os.path.join(IMG_DIR, "everest1.jpg")
    # path2 = os.path.join(IMG_DIR, "everest2.jpg")
    # part_6(path1,path2)  # use this when you generate p1.npy and p2.npy

    #Part 7
    #homography_parameters = part_7()
    
#     #Part 9
#     part_9_a(homography_parameters,path1,path2)
    




