import cv2
import numpy as np
from matplotlib import pyplot as plt


def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.
    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.
    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.
    It is recommended you use Hough tools to find these circles in
    the image.
    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.
    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.
    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    img_copy = np.copy(img_in)
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    #canny = cv2.Canny(gray,30,70)
    
    circles = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT,dp=1.5, minDist= 10,param1=70,param2=25,minRadius=radii_range[0],maxRadius=radii_range[-1]) 
    circles = circles.squeeze(0)
    
    hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    #sort from from the lowest (pixel) in y to largest (red,yellow, green)
    circles = circles[circles[:,1].argsort()]
    states = ['red','yellow','green']
    max_intesity = 0 # use value of HSV
    for i in range(len(circles)):
        #intensity = gray[int(circles[i][1]),int(circles[i][0])]
        intensity = hsv[int(circles[i][1]),int(circles[i][0])]
        if intensity[-1] > max_intesity: #use the value in HSV
            color_index= i
            coordinates = circles[i]
    # cv2.imshow("canny",canny)
    # cv2.waitKey()
    
    return coordinates, states[color_index]


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.
    Args:
        img_in (numpy.array): image containing a traffic light.
    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """ 
    img_copy = np.copy(img_in)
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray,30,70)
    # # Creating kernel
    kernel = np.ones((2, 2), np.uint8)
    canny = cv2.dilate(canny, kernel)
    # cv2.imshow("canny",canny)
    # cv2.waitKey()
    minLineLength = 0
    maxLineGap = 2
    lines = cv2.HoughLines(canny, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    result_image = img_copy
    ploygon_lines =[]
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            slope = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
            if abs(slope)==1:
                #print(slope)
                #print(pt1)
                ploygon_lines.append((pt1,pt2))
                cv2.line(result_image, pt1, pt2, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow("canny",result_image)
    cv2.waitKey()
    
    
    # template = cv2.imread('input_images/construction_template.png')
    # gray_tamplate = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # height, width = template.shape[:2]
    # canny_template = cv2.Canny(gray_tamplate, 30, 70)
    # #Creating kernel
    # kernel = np.ones((2, 2), np.uint8)
    # canny_template = cv2.dilate(canny_template, kernel)
    # # cv2.imshow("canny",canny_template)
    # # cv2.waitKey()
    # ght = cv2.createGeneralizedHoughGuil()
    
    # ght.setTemplate(canny)

    # ght.setMinDist(100)
    # ght.setMinAngle(0)
    # ght.setMaxAngle(50)
    # ght.setAngleStep(1)
    # ght.setLevels(360)
    # ght.setMinScale(1)
    # ght.setMaxScale(1.3)
    # ght.setScaleStep(0.05)
    # ght.setAngleThresh(100)
    # ght.setScaleThresh(100)
    # ght.setPosThresh(100)
    # ght.setAngleEpsilon(1)
    # ght.setLevels(360)
    # ght.setXi(0)
    
    # positions = ght.detect(gray)[0][0]
    # print(positions)
    # #return (center_x,center_y)
    
    # for position in positions:
    #     center_col = int(position[0])
    #     center_row = int(position[1])
    #     scale = position[2]
    #     angle = int(position[3])

    #     found_height = int(height * scale)
    #     found_width = int(width * scale)

    #     rectangle = ((center_col, center_row),
    #                  (found_width, found_height),
    #                  angle)

    #     box = cv2.boxPoints(rectangle)
    #     box = np.int0(box)
    #     cv2.drawContours(img_copy, [box], 0, (0, 0, 255), 2)

    #     for i in range(-2, 3):
    #         for j in range(-2, 3):
    #             img_copy[center_row + i, center_col + j] = 0, 0, 255
    # cv2.imshow("canny",img_copy)
    # cv2.waitKey()
        


def template_match(img_orig, img_template, method):
    """Returns the location corresponding to match between original image and provided template.
    Args:
        img_orig (np.array) : numpy array representing 2-D image on which we need to find the template
        img_template: numpy array representing template image which needs to be matched within the original image
        method: corresponds to one of the four metrics used to measure similarity between template and image window
    Returns:
        Co-ordinates of the topmost and leftmost pixel in the result matrix with maximum match
    """
    """Each method is calls for a different metric to determine
       the degree to which the template matches the original image
       We are required to implement each technique using the
       sliding window approach.
       Suggestion : For loops in python are notoriously slow
       Can we find a vectorized solution to make it faster?
    """
    result = np.zeros(
        (
            (img_orig.shape[0] - img_template.shape[0] + 1),
            (img_orig.shape[1] - img_template.shape[1] + 1),
        ),
        float,
    )
    top_left = []
    """Once you have populated the result matrix with the similarity metric corresponding to each overlap, return the topmost and leftmost pixel of
    the matched window from the result matrix. You may look at Open CV and numpy post processing functions to extract location of maximum match"""
    # Sum of squared differences
    if method == "tm_ssd":
        """Your code goes here"""
        raise NotImplementedError

    # Normalized sum of squared differences
    elif method == "tm_nssd":
        """Your code goes here"""
        raise NotImplementedError

    # Cross Correlation
    elif method == "tm_ccor":
        """Your code goes here"""
        raise NotImplementedError

    # Normalized Cross Correlation
    elif method == "tm_nccor":
        """Your code goes here"""
        raise NotImplementedError

    else:
        """Your code goes here"""
        # Invalid technique
    raise NotImplementedError
    return top_left


'''Below is the helper code to print images for the report'''
#     cv2.rectangle(img_orig,top_left, bottom_right, 255, 2)
#     plt.subplot(121),plt.imshow(result,cmap = 'gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(img_orig,cmap = 'gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(method)
#     plt.show()


def dft(x):
    """Discrete Fourier Transform for 1D signal
    Args:
        x (np.array): 1-dimensional numpy array of shape (n,) representing signal
    Returns:
        y (np.array): 1-dimensional numpy array of shape (n,) representing Fourier Transformed Signal

    """
    x = np.asarray(x, dtype=np.complex_)
    raise NotImplementedError


def idft(x):
    """Inverse Discrete Fourier Transform for 1D signal
    Args:
        x (np.array): 1-dimensional numpy array of shape (n,) representing Fourier-Transformed signal
    Returns:
        y (np.array): 1-dimensional numpy array of shape (n,) representing signal

    """
    x = np.asarray(x, dtype=np.complex_)
    raise NotImplementedError


def dft2(img):
    """Discrete Fourier Transform for 2D signal
    Args:
        img (np.array): 2-dimensional numpy array of shape (n,m) representing image
    Returns:
        y (np.array): 2-dimensional numpy array of shape (n,m) representing Fourier-Transformed image

    """
    raise NotImplementedError


def idft2(img):
    """Inverse Discrete Fourier Transform for 2D signal
    Args:
        img (np.array): 2-dimensional numpy array of shape (n,m) representing Fourier-Transformed image
    Returns:
        y (np.array): 2-dimensional numpy array of shape (n,m) representing image

    """
    raise NotImplementedError


def compress_image_fft(img_bgr, threshold_percentage):
    """Return compressed image by converting to fourier domain, thresholding based on threshold percentage, and converting back to fourier domain
    Args:
        img_bgr (np.array): numpy array of shape (n,m,3) representing bgr image
        threshold_percentage (float): between 0 and 1 representing what percentage of Fourier image to keep
    Returns:
        img_compressed (np.array): numpy array of shape (n,m,3) representing compressed image
        compressed_frequency_img (np.array): numpy array of shape (n,m,3) representing the compressed image in the frequency domain

    """
    raise NotImplementedError


def low_pass_filter(img_bgr, r):
    """Return low pass filtered image by keeping a circle of radius r centered on the frequency domain image
    Args:
        img_bgr (np.array): numpy array of shape (n,m,3) representing bgr image
        r (float): radius of low pass circle
    Returns:
        img_low_pass (np.array): numpy array of shape (n,m,3) representing low pass filtered image
        low_pass_frequency_img (np.array): numpy array of shape (n,m,3) representing the low pass filtered image in the frequency domain

    """
    raise NotImplementedError
