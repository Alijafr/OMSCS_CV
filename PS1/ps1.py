import math
import numpy as np
import cv2

# # Implement the functions below.


def extract_red(image):
    """ Returns the red channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the red channel.
    """
    temp_image = np.copy(image)
    red_channel = temp_image[:,:,2]
    return red_channel


def extract_green(image):
    """ Returns the green channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the green channel.
    """
    temp_image = np.copy(image)
    green_channel = temp_image[:,:,1]
    return green_channel


def extract_blue(image):
    """ Returns the blue channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the blue channel.
    """
    temp_image = np.copy(image)
    blue_channel = temp_image[:,:,0]
    
    return blue_channel


def swap_green_blue(image):
    """ Returns an image with the green and blue channels of the input image swapped. It is highly
    recommended to make a copy of the input image in order to avoid modifying the original array.
    You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 3D array with the green and blue channels swapped.
    """
    temp_image = np.copy(image)
    green_channel = np.copy(temp_image[:,:,1])
    blue_channel = np.copy(temp_image[:,:,0])
    temp_image[:,:,1] = blue_channel
    temp_image[:,:,0] = green_channel
    
    #img = img[0,0, [1,0,2]]
    
    return temp_image

def copy_paste_middle(src, dst, shape):
    """ Copies the middle region of size shape from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

        Note: Where 'middle' is ambiguous because of any difference in the oddness
        or evenness of the size of the copied region and the image size, the function
        rounds downwards.  E.g. in copying a shape = (1,1) from a src image of size (2,2)
        into an dst image of size (3,3), the function copies the range [0:1,0:1] of
        the src into the range [1:2,1:2] of the dst.

    Args:
        src (numpy.array): 2D array where the rectangular shape will be copied from.
        dst (numpy.array): 2D array where the rectangular shape will be copied to.
        shape (tuple): Tuple containing the height (int) and width (int) of the section to be
                       copied.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """
    crop_size_y = int(shape[0]/2)
    crop_size_x = int(shape[1]/2)
    src_copy = np.copy(src)
    center_y_src = int(src_copy.shape[0]/2)
    center_x_src = int(src_copy.shape[1]/2)
    dst_copy =np.copy(dst)
    center_y_dst = int(dst_copy.shape[0]/2)
    center_x_dst = int(dst_copy.shape[1]/2)
    
    src_middle = src_copy[center_y_src-crop_size_y:center_y_src+crop_size_y,center_x_src-crop_size_x:center_x_src+crop_size_x]
    dst_copy[center_y_dst-crop_size_y:center_y_dst+crop_size_y,center_x_dst-crop_size_x:center_x_dst+crop_size_x]=src_middle
    
    return dst_copy
    



def copy_paste_middle_circle(src, dst, radius):
    """ Copies the middle circle region of radius "radius" from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

    Args:
        src (numpy.array): 2D array where the circular shape will be copied from.
        dst (numpy.array): 2D array where the circular shape will be copied to.
        radius (scalar): scalar value of the radius.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """
    
    # works locally, doesnot work in gradescope
    
    # src_copy = np.copy(src)
    # center_y_src = int(src_copy.shape[0]/2)
    # center_x_src = int(src_copy.shape[1]/2)
    
    # # draw filled circles in white on black background as masks
    # mask = np.zeros_like(src)
    # mask = cv2.circle(mask, (center_x_src,center_y_src), radius, 255, -1)
    
    # foreground = cv2.bitwise_and(src_copy, mask)
    # #crop only the circle as square for now 
    # foreground = foreground[center_y_src-radius:center_y_src+radius+1,center_x_src-radius:center_x_src+radius+1]
    
    # dst_copy =np.copy(dst)
    
    # mask2 = np.zeros_like(dst)
    # center_y_dst = int(dst_copy.shape[0]/2)
    # center_x_dst = int(dst_copy.shape[1]/2)
    # mask2[center_y_dst-radius:center_y_dst+radius+1,center_x_dst-radius:center_x_dst+radius+1]= foreground
    
    # mask3 = np.full(dst.shape, 255,dtype=np.uint8)
    # mask3 = cv2.circle(mask3, (center_x_dst,center_y_dst), radius, 0, -1)
    
    # dst_croped = cv2.bitwise_and(dst_copy,mask3)
    
    # result = cv2.bitwise_or(dst_croped, mask2)
    # return result
    src_copy = np.copy(src)
    center_y_src = int((src_copy.shape[0]-1)/2)
    center_x_src = int((src_copy.shape[1]-1)/2)
    
    # draw filled circles in white on black background as masks
    mask = np.zeros_like(src,dtype=np.uint8)
    mask = cv2.circle(mask, (center_x_src,center_y_src), radius, 255, -1)
    mask = mask.astype(np.bool)
    
    
    dst_copy =np.copy(dst)
    center_y_dst = int((dst.shape[0]-1)/2)
    center_x_dst = int((dst.shape[1]-1)/2)
    
    mask2 = np.zeros_like(dst,dtype=np.uint8)
    mask2 = cv2.circle(mask2, (center_x_dst,center_y_dst), radius, 255, -1)
    mask2 = mask2.astype(np.bool)

    dst_copy[mask2] = src_copy[mask]
    
    
    return dst_copy


def image_stats(image):
    """ Returns the tuple (min,max,mean,stddev) of statistics for the input monochrome image.
    In order to become more familiar with Numpy, you should look for pre-defined functions
    that do these operations i.e. numpy.min.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.

    Returns:
        tuple: Four-element tuple containing:
               min (float): Input array minimum value.
               max (float): Input array maximum value.
               mean (float): Input array mean / average value.
               stddev (float): Input array standard deviation.
    """
    temp_image = np.copy(image)
    min_ = float(temp_image.min())
    max_ = float(temp_image.max())
    mean = temp_image.mean()
    std = temp_image.std()
    
    return min_,max_,mean,std

def center_and_normalize(image, scale):
    """ Returns an image with the same mean as the original but with values scaled about the
    mean so as to have a standard deviation of "scale".

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        scale (int or float): scale factor.

    Returns:
        numpy.array: Output 2D image.
    """
    temp_image = np.copy(image)
    mean = temp_image.mean()
    std = temp_image.std()
    
    
    temp_image = ((temp_image-mean)/std)*scale + mean
    #temp_image = temp_image.astype(np.uint8)
    
    return temp_image
    
    


def shift_image_left(image, shift):
    """ Outputs the input monochrome image shifted shift pixels to the left.

    The returned image has the same shape as the original with
    the BORDER_REPLICATE rule to fill-in missing values.  See

    http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBorder.html?highlight=copy

    for further explanation.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        shift (int): Displacement value representing the number of pixels to shift the input image.
            This parameter may be 0 representing zero displacement.

    Returns:
        numpy.array: Output shifted 2D image.
    """
    img = np.copy(image)
    # img = img.astype(np.float32)
    # rows,columns = image.shape
    # shift_x = -shift
    # shift_y = 0
    
    # matrix = np.float32([[1,0,shift_x],[0,1,shift_y]])
    # shifted_image = cv2.warpAffine(img,matrix,(columns,rows))
    # shifted_image = shifted_image[:,:-shift]
    # shifted_image = shifted_image.astype(np.uint8)
    
    #remove the shifted pixels
    
    img = img[:,shift:]
    
    borderType = cv2.BORDER_REPLICATE
    result = cv2.copyMakeBorder(img, 0, 0, 0, shift, borderType, None, None)
    # result = shifted_image[:,:-shift]
    return result


def difference_image(img1, img2):
    """ Returns the difference between the two input images (img1 - img2). The resulting array must be normalized
    and scaled to fit [0, 255].

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        img1 (numpy.array): Input 2D image.
        img2 (numpy.array): Input 2D image.

    Returns:
        numpy.array: Output 2D image containing the result of subtracting img2 from img1.
    """
    img_diff = img1-img2
    # min_ = img_diff.min()
    # max_ = img_diff.max()
    result = cv2.normalize(img_diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    # np.seterr(invalid='ignore')
    # img_diff = 255.0*(img_diff-min_)/(max_-min_)
    # img_diff = img_diff.astype(np.uint8)
    return result


def add_noise(image, channel, sigma):
    """ Returns a copy of the input color image with Gaussian noise added to
    channel (0-2). The Gaussian noise mean must be zero. The parameter sigma
    controls the standard deviation of the noise.

    The returned array values must not be clipped or normalized and scaled. This means that
    there could be values that are not in [0, 255].

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): input RGB (BGR in OpenCV) image.
        channel (int): Channel index value.
        sigma (float): Gaussian noise standard deviation.

    Returns:
        numpy.array: Output 3D array containing the result of adding Gaussian noise to the
            specified channel.
    """
    temp_image = np.copy(image)
    temp_image = temp_image.astype(np.float64)
    
    temp_image[:,:,channel] += np.random.normal(0,sigma,(temp_image.shape[:2]))
    return temp_image


def build_hybrid_image(image1, image2, cutoff_frequency):
    """ 
    Takes two images and creates a hybrid image given a cutoff frequency.
    Args:
        image1: numpy nd-array of dim (m, n, c)
        image2: numpy nd-array of dim (m, n, c)
        cutoff_frequency: scalar
    
    Returns:
        hybrid_image: numpy nd-array of dim (m, n, c)

    Credits:
        Assignment developed based on a similar project by James Hays. 
    """

    filter = cv2.getGaussianKernel(ksize=cutoff_frequency*4+1,
                                   sigma=cutoff_frequency)
    filter = np.dot(filter, filter.T)
    
    low_frequencies = cv2.filter2D(image1,-1,filter)

    high_frequencies = image2 - cv2.filter2D(image2,-1,filter)
    
    return high_frequencies + low_frequencies


def vis_hybrid_image(hybrid_image):
    """ 
    Tools to visualize the hybrid image at different scale.

    Credits:
        Assignment developed based on a similar project by James Hays. 
    """


    scales = 5
    scale_factor = 0.5
    padding = 5
    original_height = hybrid_image.shape[0]
    num_colors = 1 if hybrid_image.ndim == 2 else 3

    output = np.copy(hybrid_image)
    cur_image = np.copy(hybrid_image)
    for scale in range(2, scales+1):
      # add padding
      output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                          dtype=np.float32)))

      # downsample image
      cur_image = cv2.resize(cur_image, (0, 0), fx=scale_factor, fy=scale_factor)

      # pad the top to append to the output
      pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                     num_colors), dtype=np.float32)
      tmp = np.vstack((pad, cur_image))
      output = np.hstack((output, tmp))

    return output
