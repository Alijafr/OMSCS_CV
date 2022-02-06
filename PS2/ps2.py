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
    circles_x =[]
    circles_y = []
    for i in range(len(circles)):
        #intensity = gray[int(circles[i][1]),int(circles[i][0])]
        circles_x.append(circles[i][0])
        circles_y.append(circles[i][1])
        intensity = hsv[int(circles[i][1]),int(circles[i][0])]
        if intensity[-1] > max_intesity: #use the value in HSV
            color_index= i
            
            
    circles_x = np.array(circles_x)
    circles_y = np.array(circles_y)
    traffic_coordinates = (np.median(circles_x),np.median(circles_y))
    # cv2.imshow("canny",canny)
    # cv2.waitKey()
    
    return traffic_coordinates, states[color_index]


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
    #kernel = np.ones((2, 2), np.uint8)
    #canny = cv2.dilate(canny, kernel)
    # cv2.imshow("canny",canny)
    # cv2.waitKey()
    minLineLength = 0
    maxLineGap = 0
    lines = cv2.HoughLinesP(canny, 1, np.pi / 4, 45, minLineLength, maxLineGap)
    result_image = img_copy
    ploygon_x =[]
    ploygon_y = []
    #tolerance = 10
    if lines is not None:
       for line in lines:
           x1, y1, x2, y2 = line[0]
           #cv2.line(result_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
           
           if x2-x1 != 0:
               slope = (y2-y1)/(x2-x1)
               if abs(slope)==1:
                   print(slope)
                   ploygon_x.append(x1)
                   ploygon_x.append(x2)
                   ploygon_y.append(y1)
                   ploygon_y.append(y2)
                   cv2.line(result_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                   #print(pt1)
                   #ploygon_lines.append((pt1,pt2))

    ploygon_x = np.array(ploygon_x)
    ploygon_y = np.array(ploygon_y)
        
    #cv2.imshow("canny",result_image)
    #cv2.waitKey()
    
    return np.median(ploygon_x),np.median(ploygon_y)
    
        


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
    img_copy = np.copy(img_orig).astype(np.float32)
    img_temp_copy =  np.copy(img_template).astype(np.float32)
    #seems gradescope already pass gray scale iamge
    #img_orig_gray = cv2.cvtColor(img_orig,cv2.COLOR_BGR2GRAY)
    #img_temp_gray = cv2.cvtColor(img_template,cv2.COLOR_BGR2GRAY)
    result = np.zeros(
        (
            (img_orig.shape[0] - img_template.shape[0] + 1),
            (img_orig.shape[1] - img_template.shape[1] + 1),
        ),
        np.float32,
    )
    top_left = []
    """Once you have populated the result matrix with the similarity metric corresponding to each overlap, return the topmost and leftmost pixel of
    the matched window from the result matrix. You may look at Open CV and numpy post processing functions to extract location of maximum match"""
    rows_orig, cols_orig = img_orig.shape[:2]
    rows_temp, cols_temp = img_template.shape[:2]
    # Sum of squared differences
    if method == "tm_ssd":
        """Your code goes here"""
        for i in range(len(result)):
            for j in range(len(result[1])):
                #print("{},{}".format(i,j))
                sliding_winodw = img_copy[i:i+rows_temp,j:j+cols_temp]-img_temp_copy
                result[i,j] = np.sum(sliding_winodw*sliding_winodw)
        
        #res = cv2.matchTemplate(img_copy,img_temp_copy,cv2.TM_SQDIFF)
        min_pixel_value = np.argwhere(result==np.min(result))
        top_left = (min_pixel_value[0][1],min_pixel_value[0][0]) #points in (x,y) while pixel is in (y,x)
        # return top_left

    # Normalized sum of squared differences
    elif method == "tm_nssd":
        """Your code goes here"""
        for i in range(len(result)):
            for j in range(len(result[1])):
                sliding_winodw = img_copy[i:i+rows_temp,j:j+cols_temp]
                diff_sq =  (sliding_winodw-img_temp_copy)**2
                diff_sq_sum = np.sum(diff_sq)
                normalizer = np.sum(sliding_winodw**2) * np.sum(img_temp_copy**2)
                result[i,j] = diff_sq_sum/np.sqrt(normalizer)      
        #res = cv2.matchTemplate(img_orig,img_template,cv2.TM_SQDIFF_NORMED)
        min_pixel_value = np.argwhere(result==np.min(result))
        top_left = (min_pixel_value[0][1],min_pixel_value[0][0]) #points in (x,y) while pixel is in (y,x)

    # Cross Correlation
    elif method == "tm_ccor":
        """Your code goes here"""
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                sliding_winodw = img_copy[i:i+rows_temp,j:j+cols_temp]
                product =  sliding_winodw*img_temp_copy
                result[i,j] = np.sum(product)
        
        #res = cv2.matchTemplate(img_orig,img_template,cv2.TM_CCORR)
        #res = cv2.matchTemplate(img_copy,img_temp_copy,cv2.TM_CCORR)
        #min_pixel_value = np.argwhere(result==np.min(result))
        #top_left = (min_pixel_value[0][1],min_pixel_value[0][0]) #points in (x,y) while pixel is in (y,x)
        max_pixel_value = np.argwhere(result==np.max(result))
        top_left = (max_pixel_value[0][1],max_pixel_value[0][0])
    # Normalized Cross Correlation
    elif method == "tm_nccor":
        """Your code goes here"""
        for i in range(len(result)):
            for j in range(len(result[1])):
                sliding_winodw = img_copy[i:i+rows_temp,j:j+cols_temp]
                product =  sliding_winodw*img_temp_copy
                product_sum = np.sum(product)
                normalizer = np.sum(sliding_winodw**2) * np.sum(img_temp_copy**2)
                result[i,j] = product_sum/np.sqrt(normalizer)
        #res = cv2.matchTemplate(img_orig,img_template,cv2.TM_CCORR_NORMED)
        #min_pixel_value = np.argwhere(result==np.min(result))
        #top_left = (min_pixel_value[0][1],min_pixel_value[0][0])
        max_pixel_value = np.argwhere(result==np.max(result))
        top_left = (max_pixel_value[0][1],max_pixel_value[0][0])

    else:
        """Your code goes here"""
        # Invalid technique
        print("Invalid technique")
    
    return top_left

    # img_copy = np.copy(img_orig)
    # '''Below is the helper code to print images for the report'''
    # bottom_right = (top_left[0]+height_temp, top_left[1]+width_temp)
    # img_copy = cv2.rectangle(img_copy,top_left,bottom_right,(0,0,255),2)
    # cv2.imshow("result image",img_copy)
    # cv2.waitKey()
    # plt.subplot(121),plt.imshow(result,cmap = 'gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(img_copy)
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    # #plt.suptitle(method)
    # plt.show()


def dft(x):
    """Discrete Fourier Transform for 1D signal
    Args:
        x (np.array): 1-dimensional numpy array of shape (n,) representing signal
    Returns:
        y (np.array): 1-dimensional numpy array of shape (n,) representing Fourier Transformed Signal

    """
    N = len(x)
    m = np.arange(N) #is actually x in the equation but cannot called x since the input is x 
    k = m.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * m / N)
    
    #dft_ = np.sum(e*x,axis=1)
    dft_ = np.dot(e, x)
    
    return dft_


def idft(x):
    """Inverse Discrete Fourier Transform for 1D signal
    Args:
        x (np.array): 1-dimensional numpy array of shape (n,) representing Fourier-Transformed signal
    Returns:
        y (np.array): 1-dimensional numpy array of shape (n,) representing signal

    """
    
    N = len(x)
    m = np.arange(N) #is actually x in the equation but cannot called x since the input is x 
    k = m.reshape((N, 1))
    e = np.exp(2j * np.pi * k * m / N)
    #inv_dft = np.sum(e*x,axis=1)/N
    inv_dft = np.dot(e, x)/N
    
    return inv_dft


def dft2(img):
    """Discrete Fourier Transform for 2D signal
    Args:
        img (np.array): 2-dimensional numpy array of shape (n,m) representing image
    Returns:
        y (np.array): 2-dimensional numpy array of shape (n,m) representing Fourier-Transformed image

    """
    
    
    Nx= len(img)
    Ny= len(img[0])
    dft_2d = np.zeros((Nx,Ny),dtype=np.complex)
    # mx = np.arange(Nx)
    # kx = mx.reshape((Nx, 1))
    # my = np.arange(Ny)
    # ky = my.reshape((Ny, 1))
    
    
    # e = np.exp(-2j * np.pi * (kx * mx/Nx + ky*my/Ny))
    
    # dft_2d = np.dot(e,img)
    row_dft_2d = np.copy(dft_2d)
    
    for i in range(Ny):
        dft_ = dft(img[:,i])
        row_dft_2d[:,i] = dft_
    for i in range(Nx):
        dft_ = dft(row_dft_2d[i,:])
        dft_2d[i,:] = dft_
        
    
    return dft_2d
    


def idft2(img):
    """Inverse Discrete Fourier Transform for 2D signal
    Args:
        img (np.array): 2-dimensional numpy array of shape (n,m) representing Fourier-Transformed image
    Returns:
        y (np.array): 2-dimensional numpy array of shape (n,m) representing image

    """
    Nx= len(img)
    Ny= len(img[0])
    inv_dft_2d = np.zeros((Nx,Ny),dtype=np.complex)
    # mx = np.arange(Nx)
    # kx = mx.reshape((Nx, 1))
    # my = np.arange(Ny)
    # ky = my.reshape((Ny, 1))
    
    
    # e = np.exp(-2j * np.pi * (kx * mx/Nx + ky*my/Ny))
    
    # dft_2d = np.dot(e,img)
    row_inv_dft_2d = np.copy(inv_dft_2d)
    
    for i in range(Ny):
        inv_dft_ = idft(img[:,i])
        row_inv_dft_2d[:,i] = inv_dft_
    for i in range(Nx):
        inv_dft_ = idft(row_inv_dft_2d[i,:])
        inv_dft_2d[i,:] = inv_dft_
        
    
    return inv_dft_2d


def compress_image_fft(img_bgr, threshold_percentage):
    """Return compressed image by converting to fourier domain, thresholding based on threshold percentage, and converting back to fourier domain
    Args:
        img_bgr (np.array): numpy array of shape (n,m,3) representing bgr image
        threshold_percentage (float): between 0 and 1 representing what percentage of Fourier image to keep
    Returns:
        img_compressed (np.array): numpy array of shape (n,m,3) representing compressed image
        compressed_frequency_img (np.array): numpy array of shape (n,m,3) representing the compressed image in the frequency domain
        

    """
    #convert the image to flaot64 to preform the calcualtions 
    img_compressed = np.copy(img_bgr).astype(np.float64)
    # variable to save the compressed frequence image
    compressed_frequency_img = np.zeros((img_bgr.shape),dtype=np.complex128)
    for i in range(3):#iterate over the 3 channels
        #save the channel image
        img = img_bgr[:,:,i]
        #dft_=dft2(img) #too slow
        #convert to DFT
        dft_ = np.fft.fft2(img) 
        # flatten the DFT to sort it
        flattened_dft = dft_.flatten()
        # sort from largest to smallest 
        sorted_f = -np.sort(-np.abs(flattened_dft)) #the 2 negative are needed to sort from largest to smallest (otherwise it will be reverse)
        #find the index at which the threshold will be taken 
        n_2= len(flattened_dft)
        threshold_index = int(threshold_percentage*n_2)
        threshold = sorted_f[threshold_index]
        
        dft_[np.abs(dft_) <= threshold] = 0
        img_freq_channel= dft_
        img_compressed_channel = np.fft.ifft2(img_freq_channel)
        
        
        #img_compressed[:,:,i] = np.abs(img_compressed_channel)
        img_compressed[:,:,i] = img_compressed_channel.real

        compressed_frequency_img[:,:,i] = img_freq_channel
    
    return img_compressed , compressed_frequency_img
        
        


def low_pass_filter(img_bgr, r):
    """Return low pass filtered image by keeping a circle of radius r centered on the frequency domain image
    Args:
        img_bgr (np.array): numpy array of shape (n,m,3) representing bgr image
        r (float): radius of low pass circle
    Returns:
        img_low_pass (np.array): numpy array of shape (n,m,3) representing low pass filtered image
        low_pass_frequency_img (np.array): numpy array of shape (n,m,3) representing the low pass filtered image in the frequency domain

    """
    #convert the image to flaot64 to preform the calcualtions 
    #img_compressed = np.copy(img_bgr).astype(np.float64) 
    #converted into double in the experiment
    img_compressed = img_bgr
    # variable to save the compressed frequence image
    compressed_frequency_img = np.zeros((img_bgr.shape),dtype=np.complex128)
    for i in range(3):#iterate over the 3 channels
        #save the channel image
        img = img_bgr[:,:,i]
        #convert to DFT
        dft_ = np.fft.fft2(img)
        #shift the spectral freq so that low frequencies are in teh center of the image
        dft_ = np.fft.fftshift(dft_)
        mask = np.zeros(img.shape)
        mask = cv2.circle(mask, (int(img.shape[1]/2),int(img.shape[0]/2)), r, 1,-1)
        img_freq_channel = mask * dft_
        img_freq_channel = np.fft.ifftshift(img_freq_channel)
        img_compressed_channel = np.fft.ifft2(img_freq_channel)
        
        img_compressed[:,:,i] = img_compressed_channel.real
        compressed_frequency_img[:,:,i] = img_freq_channel
    
    return img_compressed , compressed_frequency_img
