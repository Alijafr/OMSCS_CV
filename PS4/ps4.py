"""Problem Set 4: Motion Detection"""

import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided


# Utility function
def read_video(video_file, show=False):
    """Reads a video file and outputs a list of consecuative frames
  Args:
      image (string): Video file path
      show (bool):    Visualize the input video. WARNING doesn't work in
                      notebooks
  Returns:
      list(numpy.ndarray): list of frames
  """
    frames = []
    cap = cv2.VideoCapture(video_file)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        # Opens a new window and displays the input
        if show:
            cv2.imshow("input", frame)
            # Frames are read by intervals of 1 millisecond. The
            # programs breaks out of the while loop when the
            # user presses the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # The following frees up resources and
    # closes all windows
    cap.release()
    if show:
        cv2.destroyAllWindows()
    return frames
    
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """
    scale = 1/8
    delta = 0
    #ddepth = cv2.CV_16S
    ddepth = -1

    grad_x = cv2.Sobel(image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    return grad_x


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """

    scale = 1/8
    delta = 0
    #ddepth = cv2.CV_16S
    ddepth = -1

    grad_y = cv2.Sobel(image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    return grad_y


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """
    m,n = img_a.shape[:2]
    Ix = gradient_x(img_a)
    Iy = gradient_y(img_a)
    It = img_b-img_a
    Ixx = Ix*Ix
    Ixy = Ix*Iy
    Iyy = Iy*Iy
    
    Ixt = Ix*It
    Iyt = Iy*It
    
    
    if k_type=='uniform':
        kernel = np.ones((k_size,k_size))/(k_size**2)
        sx2 = cv2.filter2D(Ixx, ddepth=-1, kernel= kernel)
        sy2 = cv2.filter2D(Iyy, ddepth=-1, kernel=kernel)
        sxsy = cv2.filter2D(Ixy, ddepth=-1, kernel= kernel)
        sxst = -1*cv2.filter2D(Ixt, ddepth=-1, kernel= kernel)
        syst = -1*cv2.filter2D(Iyt, ddepth=-1, kernel= kernel)
    else:
        #get the wighted sums 
        sx2 = cv2.GaussianBlur(Ixx,(k_size,k_size),sigma)
        sy2 = cv2.GaussianBlur(Iyy,(k_size,k_size),sigma)
        sxsy = cv2.GaussianBlur(Ixy,(k_size,k_size),sigma)
        sxst = -1*cv2.GaussianBlur(Ixt,(k_size,k_size),sigma)
        syst = -1*cv2.GaussianBlur(Iyt,(k_size,k_size),sigma)
   
    #A matrix 
    A = np.array([[sx2,sxsy],
                 [sxsy,sy2]]) # this is 2,2,m,n
    A = np.transpose(A,(2,3,0,1))  # this is now m,n ,2 ,2 --> converted this way to use np.linalg
    det = np.linalg.det(A)
    
    #B matrix
    B = np.array([sxst,syst]) # 2,m,n
    B = np.transpose(B,(1,2,0)) # this is now m,n ,2 --> converted this way to use np.linalg
    
    uv = np.zeros((m,n,2))
    epsilon = 1e-20
    #add more to the filter --> eig values need to be big && almost equal
    mask = det > epsilon
    
    #vu = np.linalg.pinv(A)@B
    uv[mask,:] = np.linalg.solve(A[mask,:,:], B[mask,:])
    
    return uv[:,:,0], uv[:,:,1]
    
    


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    #naive reduce operator --> better way is using the average
    #sigma = 1
    #image=cv2.GaussianBlur(image,(5,5),sigma)
    kernel = np.array([1, 4, 6, 4, 1])/16
    image = cv2.sepFilter2D(image, -1, kernel, kernel)
    m,n = image.shape[:2]
    if m%2 != 0 : m-=1
    if n%2 != 0 : n-=1
    image = image[np.arange(0,m,2)]
    image = image[:,np.arange(0,n,2)]
    
    return image

    


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    pyramid = []
    pyramid.append(image)
    for i in range(levels-1): #level 0 has been done outside of the loop
        image = reduce_image(image)
        pyramid.append(image)
        
    return pyramid
        
        
        


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """
    output_image = normalize_and_scale(img_list[0])
    w_orig, h_orig = output_image.shape[:2]
    for i in range(1,len(img_list)):
        image = normalize_and_scale(img_list[i])
        w,h = image.shape[:2]
        image = cv2.copyMakeBorder(image, 0, w_orig-w, 0, 0, borderType=cv2.BORDER_CONSTANT, value=0)
        output_image = np.hstack((output_image,image))
        
    
    return output_image
        

    


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """
    m,n = image.shape[:2]
    out_img = np.zeros((2*m,2*n))
    out_img[::2,::2] = image #insert 0 between each rows and columns of image (::2 menas step of 2)
    #now apply the seperable fitler
    kernel = np.array([1, 4, 6, 4, 1])/8
    out_img = cv2.sepFilter2D(out_img, -1, kernel, kernel)
    
    return out_img
    
    


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    levels = len(g_pyr)-1
    laplacian_pyramids = []
    laplacian_pyramids.append(g_pyr[-1])
    for i in range(levels):
        img_gauss = g_pyr[levels-i]
        expand_img = expand_image(img_gauss)
        laplacian_pyramids.append(g_pyr[levels-i-1]-expand_img)
    
    
    laplacian_pyramids.reverse()
    return laplacian_pyramids


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """
    warp = np.copy(image)
    M, N = image. shape
    X, Y = np.meshgrid( range (N) , range (M) )
    map_x = X + U
    map_y = Y + V
    map_x = np.clip(map_x, 0, N-1)
    map_y = np.clip(map_y, 0, M-1)
    # for i in range(M):
    #     for j in range(N):
    #         warp[i,j] = image[int(i+V[i,j]),int(j+U[i,j])] 
    warp = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), interpolation,border_mode,borderValue = 0)    
    return warp
    


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """
    gauss_a = gaussian_pyramid(img_a, levels)
    #laplacian_a = laplacian_pyramid(gauss_a)
    gauss_b = gaussian_pyramid(img_b, levels)
    #laplacian_b = laplacian_pyramid(gauss_b)
    pre_U = None
    pre_V = None
    iter_ = levels-1
    for i in range(iter_):
        if i ==0:
            reduced_a = gauss_a[iter_-i]
            reduced_b = gauss_b[iter_-i]
            U,V = optic_flow_lk(reduced_a, reduced_b, k_size, k_type,sigma)
            #multiply the fields by 2, and expand them. It will be used for warping the next level of a --> should be almost equal to the next level of b 
            expand_U = expand_image(2*U)
            expand_V = expand_image(2*V)
            #now create the warp image from the the lowest level to next 
            reduced_a = gauss_a[iter_-i-1]
            warped_image_a =warp(reduced_a, expand_U, expand_V, interpolation, border_mode)
        else:
             #next level of b
             reduced_b = gauss_b[iter_-i]
             #do the optical flow for the next level
             delt_U,delt_V = optic_flow_lk(warped_image_a, reduced_b, k_size, k_type,sigma)
             U = pre_U +delt_U 
             V = pre_V + delt_V
             #expand , and warp again
             expand_U = expand_image(2*U)
             expand_V = expand_image(2*V)
             reduced_a = gauss_a[iter_-i-1]
             warped_image_a =warp(reduced_a, expand_U, expand_V, interpolation, border_mode)
             
        pre_U = expand_U
        pre_V = expand_V
        
    #now just apply the optical flow for the highest level 
    delt_U,delt_V = optic_flow_lk(warped_image_a, img_b, k_size, k_type,sigma)
    #add it to the previous displacements 
    U = pre_U +delt_U 
    V = pre_V + delt_V
        
    return U,V
        
    

    

def classify_video(images):
  """Classifies a set of frames as either
    - int(1) == "Running"
    - int(2) == "Walking"
    - int(3) == "Clapping"
  Args:
      images list(numpy.array): greyscale floating-point frames of a video
  Returns:
      int:  Class of video
  """
  return 0 
