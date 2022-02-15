"""
CS6476 Assignment 3 imports. Only Numpy and cv2 are allowed.
"""


import cv2
import numpy as np
from scipy import ndimage
from typing import Tuple


class Mouse_Click_Correspondence(object):

    def __init__(self,path1='',path2='',img1='',img2=''):
        self.sx1 = []
        self.sy1 = []
        self.sx2 = []
        self.sy2 = []
        self.img = img1
        self.img2 = img2
        self.path1 = path1
        self.path2 = path2


    def click_event(self,event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print('x y', x, ' ', y)

            sx1=self.sx1
            sy1=self.sy1

            sx1.append(x)
            sy1.append(y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img, str(x) + ',' +
                        str(y), (x, y), font,
                        1, (255, 0, 0), 2)
            cv2.imshow('image 1', self.img)

            # checking for right mouse clicks
        if event == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = self.img[y, x, 0]
            g = self.img[y, x, 1]
            r = self.img[y, x, 2]
            cv2.putText(self.img, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x2, y2), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image 1', self.img)

        # driver function

    def click_event2(self,event2, x2, y2, flags, params):
        # checking for left mouse clicks
        if event2 == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print('x2 y2', x2, ' ', y2)

            sx2= self.sx2
            sy2 = self.sy2

            sx2.append(x2)
            sy2.append(y2)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img2, str(x2) + ',' +
                        str(y2), (x2, y2), font,
                        1, (0, 255, 255), 2)
            cv2.imshow('image 2', self.img2)

            # checking for right mouse clicks
        if event2 == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x2, ' ', y2)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = self.img2[y2, x2, 0]
            g = self.img2[y2, x2, 1]
            r = self.img2[y2, x2, 2]
            cv2.putText(self.img2, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x2, y2), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image 2', self.img2)

    # driver function
    def driver(self,path1,path2):
        # reading the image
        # path = r'D:\GaTech\TA - CV\ps05\ps05\ps5-1-b-1.png'
        #path1 = r'1a_notredame.jpg'
        #path2 = r'1b_notredame.jpg'


        #path1 = self.path1
        #path2 = self.path2

        # path1 = r'crop1.jpg'
        # path2 = r'crop2.jpg'

        self.img = cv2.imread(path1, 1)
        self.img2 = cv2.imread(path2, 2)

        # displaying the image
        cv2.namedWindow("image 1", cv2.WINDOW_NORMAL)
        cv2.imshow('image 1', self.img)
        cv2.namedWindow("image 2", cv2.WINDOW_NORMAL)
        cv2.imshow('image 2', self.img2)

        # setting mouse hadler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image 1', self.click_event)
        cv2.setMouseCallback('image 2', self.click_event2)

        # wait for a key to be pressed to exit

        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()

        print('sx1 sy1', self.sx1, self.sy1)
        print('sx2 sy2', self.sx2, self.sy2)

        points1, points2 = [], []
        for x, y in zip(self.sx1, self.sy1):
            points1.append((x, y))

        points_1 = np.array(points1)

        for x, y in zip(self.sx2, self.sy2):
            points2.append((x, y))

        points_2 = np.array(points2)

        np.save('p1.npy', points_1)
        np.save('p2.npy', points_2)



def euclidean_distance(p0, p1):
    """Get the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1
        p1 (tuple): Point 2
    Return:
        float: The distance between points
    """

    
    return ((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)**0.5


def get_corners_list(image):
    """List of image corner coordinates used in warping.

    Args:
        image (numpy.array of float64): image array.
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
    """
    height, width = image.shape[:2]
    corners = [(0,0),(0,height-1),(width-1,0),(width-1,height-1)]

    return corners


# def find_markers(image, template=None):
#     """Finds four corner markers.

#     Use a combination of circle finding and convolution to find the
#     four markers in the image.

#     Args:
#         image (numpy.array of uint8): image array.
#         template (numpy.array of unint8): template of the markers
#     Returns:
#         list: List of four (x, y) tuples
#             in the order [top-left, bottom-left, top-right, bottom-right]
#     """
#     out_list = []
#     img_copy = np.copy(image)
#     # median = cv2.medianBlur(img_copy,1)
#     gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
#     #canny = cv2.Canny(gray,30,70)
    
#     circles = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT,dp=1.5, minDist= 10,param1=70,param2=25,minRadius=10,maxRadius=30) 
#     circles = circles.squeeze(0)
#     for i in range(len(circles)):
#         #check if the circles has a conrner in the middle
#         center = (int(circles[i][0]),int(circles[i][1]))
#         r = int(circles[i][2] *0.6) #1.2 is just scale to make sure we crop the whole image
#         circle_image = gray[center[1]-r:center[1]+r,center[0]-r:center[0]+r]
#         dst = cv2.cornerHarris(circle_image,2,3,0.04)
#         if dst.max() > 0.05:
#             #there is a corner in the middle, add the center of the circle 
#             out_list.append(center)
        
#     return out_list

def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding and convolution to find the
    four markers in the image.

    Args:
        image (numpy.array of uint8): image array.
        template (numpy.array of unint8): template of the markers
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
    """
    out_list = []
    img_copy = np.copy(image)
    orig_w ,orig_h = image.shape[:-1]
    #median = cv2.medianBlur(img_copy,5)
    #gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    templates = []
    rot = [0,45,90,135]
    
    
    
    w, h = template.shape[:-1]
    res = np.zeros((4,orig_w-w+1,orig_h-h+1))
    for i in range(len(rot)):
        if rot[i] ==90:
            temp = ndimage.rotate(template, rot[i])
        elif rot[i] == 45 or rot[i] ==135:
            temp = ndimage.rotate(template, rot[i])
            rot_w,rot_h = temp.shape[:2]
            temp = temp[int((rot_w-w)/2):int((rot_w+w)/2),int((rot_h-h)/2):int((rot_h+h)/2)]
        else:
            temp = template
        
            
        mask = np.zeros_like(temp)
        
        cv2.circle(mask,(int(mask.shape[1]/2),int(mask.shape[0]/2)),int(mask.shape[1]/2)-1,(1,1,1),-1)
        mask = mask.astype(int)
        #masked_temp = mask*temp
        #templates.append(masked_temp)
        #res[i,:,:] = cv2.matchTemplate(img_copy,masked_temp,cv2.TM_SQDIFF_NORMED)
        res[i,:,:] = cv2.matchTemplate(img_copy,temp,cv2.TM_SQDIFF,mask=mask)
    
    #take the min of all the template results
    res = np.min(res,axis=0)
    #find the four markers 
    points = []
    padding_factor = 0.5 # make sure that we are not chossing the same marker twice
    for i in range(4):
        marker = np.argwhere(res==np.min(res))[0]
        points.append(marker)
        #remove all the neighoring points (instead of non max supersion)
        res[int(marker[0]-padding_factor*w):int(marker[0]+padding_factor*w),int(marker[1]-padding_factor*h):int(marker[1]+padding_factor*h) ] = np.inf
    
    #sort the markers [top-left, bottom-left, top-right, bottom-right]
    #change the top left corner to the center of the marker 
    points = sorted(points, key= lambda k:k[1].min()) # min in x
    if points[0][0] <= points[1][0]:
        out_list.append((int(points[0][1]+h/2),int(points[0][0]+w/2)))
        out_list.append((int(points[1][1]+h/2),int(points[1][0]+w/2)))
    else:
        out_list.append((int(points[1][1]+h/2),int(points[1][0]+w/2)))
        out_list.append((int(points[0][1]+h/2),int(points[0][0]+w/2)))
    
    #remove the first 2 elements that they are already added
    points.remove(points[0]) 
    points.remove(points[0])
    #sort the remaining (top-right, bottom-right) by min rows (y)
    points = sorted(points, key= lambda k:k[0].min()) # min in y
    out_list.append((int(points[0][1]+h/2),int(points[0][0]+w/2)))
    out_list.append((int(points[1][1]+h/2),int(points[1][0]+w/2)))
    # if points[2][0] <= points[2][0]:
    #     out_list.append((int(points[2][1]+h/2),int(points[2][0]+w/2)))
    #     out_list.append((int(points[3][1]+h/2),int(points[3][0]+w/2)))
    # else:
    #     out_list.append((int(points[3][1]+h/2),int(points[3][0]+w/2)))
    #     out_list.append((int(points[2][1]+h/2),int(points[2][0]+w/2)))
        
            
          
        
    return out_list



def draw_box(image, markers, thickness=1):
    """Draw 1-pixel width lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line and leave the default "thickness" and "lineType".

    Args:
        image (numpy.array of uint8): image array
        markers(list of tuple): the points where the markers were located
        thickness(int): thickness of line used to draw the boxes edges
    Returns:
        numpy.array: image with lines drawn.
    """
    out_image = np.copy(image)
    cv2.line(out_image, markers[0], markers[1], (0,255,0),thickness)
    cv2.line(out_image, markers[0], markers[2], (0,255,0),thickness)
    cv2.line(out_image, markers[3], markers[2], (0,255,0),thickness)
    cv2.line(out_image, markers[3], markers[1], (0,255,0),thickness)
    return out_image


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Using the four markers in imageB, project imageA into the marked area.

    You should have used your find_markers method to find the corners and then
    compute the homography matrix prior to using this function.

    Args:
        image (numpy.array of uint8): image array
        image (numpy.array of uint8): image array
        homography (numpy.array): Perspective transformation matrix, 3 x 3
    Returns:
        numpy.array: combined image
    """
    wall = np.copy(imageB)
    scene = np.copy(imageA)
    src_points = get_corners_list(scene)
    #recover the markers using the homography and src_points 
    homog_src = np.vstack((np.array(src_points).T,np.array([[1,1,1,1]])))
    markers = homography@homog_src
    markers = (markers/markers[-1])[:2,:].T.astype(int) #remove the ones and transpose it back --> each row is a point 
    
    #do np.fillpoly to identify the area to place imageB
    
    #change the markers orders for cv2.fillpoly 
    m = markers[[0,2,3,1],:]
    mask = np.zeros_like(wall[:,:,1])
    cv2.fillPoly(mask,[m],255)
    indices = np.where(mask==255) #ignore the channel
    dst_points = np.vstack((indices[1],indices[0],np.ones((1,len(indices[0]))))) # make it x, y , 1
    
    H_inv = np.linalg.inv(homography)
    h_scene, w_scene = scene.shape[:2]
    
    src_points_ = np.dot(H_inv,dst_points)
    src_points_ = src_points_/src_points_[-1]
    
    #round to the nearst integer 
    src_points_ = np.rint(np.abs(src_points_)).astype(int)
    #change from x,y to rows and cols, and removes the ones 
    src_points_ = src_points_[[1,0],:]
    #just make sure that there is no index out of range because of the rint
    src_points_[0,np.where(src_points_[0] >=scene.shape[0])] = scene.shape[0]-1 #if the index is the height, make it height-1 (index starts from zero)
    src_points_[1,np.where(src_points_[1] >=scene.shape[1])] = scene.shape[1]-1
    
    #clean up the dst_points
    dst_points = np.vstack((indices[0],indices[1])).astype(int) # make back to rows and cols and integers 
    out_image = imageB.copy()
    out_image[dst_points[0],dst_points[1]] = scene[src_points_[0],src_points_[1]]
    


    
    return out_image


def find_four_point_transform(srcPoints, dstPoints):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform
    Hint: You will probably need to use least squares to solve this.
    Args:
        srcPoints (list): List of four (x,y) source points
        dstPoints (list): List of four (x,y) destination points
    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values
    """
    #copied from opencv implmentation
    """
    /* Calculates coefficients of perspective transformation
     * which maps (xi,yi) to (ui,vi), (i=1,2,3,4):
     *
     *      c00*xi + c01*yi + c02
     * ui = ---------------------
     *      c20*xi + c21*yi + c22
     *
     *      c10*xi + c11*yi + c12
     * vi = ---------------------
     *      c20*xi + c21*yi + c22
     *
     * Coefficients are calculated by solving linear system:
     * / x0 y0  1  0  0  0 -x0*u0 -y0*u0 \ /c00\ |u0|
     * | x1 y1  1  0  0  0 -x1*u1 -y1*u1 | |c01| |u1|
     * | x2 y2  1  0  0  0 -x2*u2 -y2*u2 | |c02| |u2|
     * | x3 y3  1  0  0  0 -x3*u3 -y3*u3 |.|c10|=|u3|,
     * |  0  0  0 x0 y0  1 -x0*v0 -y0*v0 | |c11| |v0|
     * |  0  0  0 x1 y1  1 -x1*v1 -y1*v1 | |c12| |v1|
     * |  0  0  0 x2 y2  1 -x2*v2 -y2*v2 | |c20| |v2|
     * \  0  0  0 x3 y3  1 -x3*v3 -y3*v3 / \c21/ |v3|
     *
     * where:
     *   cij - matrix coefficients, c22 = 1
     */
    """
    x0,y0 = srcPoints[0]
    x1,y1 = srcPoints[1]
    x2,y2 = srcPoints[2]
    x3,y3 = srcPoints[3]
    u0,v0 = dstPoints[0]
    u1,v1 = dstPoints[1]
    u2,v2 = dstPoints[2]
    u3,v3 = dstPoints[3]
    A = np.array([[x0, y0,1,0,0,0,-x0*u0, -y0*u0],
                  [x1, y1,1,0,0,0,-x1*u1, -y1*u1],
                  [x2, y2,1,0,0,0,-x2*u2, -y2*u2],
                  [x3, y3,1,0,0,0,-x3*u3, -y3*u3],
                  [0, 0,0,x0,y0,1,-x0*v0, -y0*v0],
                  [0, 0,0,x1,y1,1,-x1*v1, -y1*v1],
                  [0, 0,0,x2,y2,1,-x2*v2, -y2*v2],
                  [0, 0,0,x3,y3,1,-x3*v3, -y3*v3], 
                  ])
    X = np.array([[u0],
                  [u1],
                  [u2],
                  [u3],
                  [v0],
                  [v1],
                  [v2],
                  [v3]])
    M = np.linalg.inv(A)@X
    #append M by 1 --> c22 = 1, then reshape to 3x3
    homography = np.vstack((M,np.array([[1]]))).reshape(3,3)
    return homography


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename
    """

    # Open file with VideoCapture and set result to 'video'. (add 1 line)
    video = cv2.VideoCapture(filename)

    # TODO
    raise NotImplementedError

    # Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None



class Automatic_Corner_Detection(object):

    def __init__(self):

        self.SOBEL_X = np.array(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]).astype(np.float32)
        self.SOBEL_Y = np.array(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]).astype(np.float32)



    def gradients(self, image_bw):
        '''Use convolution with Sobel filters to calculate the image gradient at each
            pixel location
            Input -
            :param image_bw: A numpy array of shape (M,N) containing the grayscale image
            Output -
            :return Ix: Array of shape (M,N) representing partial derivatives of image
                    in x-direction
            :return Iy: Array of shape (M,N) representing partial derivative of image
                    in y-direction
        '''

        raise NotImplementedError

        return Ix, Iy



    def second_moments(self, image_bw, ksize=7, sigma=10):
        """ Compute second moments from image.
            Compute image gradients, Ix and Iy at each pixel, the mixed derivatives and then the
            second moments (sx2, sxsy, sy2) at each pixel,using convolution with a Gaussian filter. You may call the
            previously written function for obtaining the gradients here.
            Input -
            :param image_bw: array of shape (M,N) containing the grayscale image
            :param ksize: size of 2d Gaussian filter
            :param sigma: standard deviation of Gaussian filter
            Output -
            :return sx2: np array of shape (M,N) containing the second moment in x direction
            :return sy2: np array of shape (M,N) containing the second moment in y direction
            :return sxsy: np array of shape (M,N) containing the second moment in the x then the
                    y direction
        """

        sx2, sy2, sxsy = None, None, None

        raise NotImplementedError

        return sx2, sy2, sxsy


    def harris_response_map(self, image_bw, ksize=7, sigma=5, alpha=0.05):
        """Compute the Harris cornerness score at each pixel (See Szeliski 7.1.1)
            R = det(M) - alpha * (trace(M))^2
            where M = [S_xx S_xy;
                       S_xy  S_yy],
                  S_xx = Gk * I_xx
                  S_yy = Gk * I_yy
                  S_xy  = Gk * I_xy,
            and * is a convolutional operation over a Gaussian kernel of size (k, k).
            (You can verify that this is equivalent to taking a (Gaussian) weighted sum
            over the window of size (k, k), see how convolutional operation works here:
                http://cs231n.github.io/convolutional-networks/)
            Ix, Iy are simply image derivatives in x and y directions, respectively.
            You may find the Pytorch function nn.Conv2d() helpful here.
            Input-
            :param image_bw: array of shape (M,N) containing the grayscale image
            :param ksize: size of 2d Gaussian filter
            :param sigma: standard deviation of gaussian filter
            :param alpha: scalar term in Harris response score
            Output-
            :return R: np array of shape (M,N), indicating the corner score of each pixel.
            """


        raise NotImplementedError

        return R


    def nms_maxpool(self, R, k, ksize):
        """ Get top k interest points that are local maxima over (ksize,ksize)
        neighborhood.
        One simple way to do non-maximum suppression is to simply pick a
        local maximum over some window size (u, v). Note that this would give us all local maxima even when they
        have a really low score compare to other local maxima. It might be useful
        to threshold out low value score before doing the pooling.
        Threshold globally everything below the median to zero, and then
        MaxPool over a 7x7 kernel. This will fill every entry in the subgrids
        with the maximum nearby value. Binarize the image according to
        locations that are equal to their maximum. Multiply this binary
        image, multiplied with the cornerness response values.
        Args:
            R: np array of shape (M,N) with score response map
            k: number of interest points (take top k by confidence)
            ksize: kernel size of max-pooling operator
        Returns:
            x: np array of shape (k,) containing x-coordinates of interest points
            y: np array of shape (k,) containing y-coordinates of interest points
        """



        raise NotImplementedError

        return x, y


    def harris_corner(self, image_bw, k=100):
        """
            Implement the Harris Corner detector. You can call harris_response_map(), nms_maxpool() functions here.
            Input-
            :param image_bw: array of shape (M,N) containing the grayscale image
            :param k: maximum number of interest points to retrieve
            Output-
            :return x: np array of shape (p,) containing x-coordinates of interest points
            :return y: np array of shape (p,) containing y-coordinates of interest points
            """

        raise NotImplementedError

        return x1, y1





class Image_Mosaic(object):

    def __int__(self):
        pass

    def image_warp_inv(self, im_src, im_dst, H):
        '''
        Input -
        :param im_src: Image 1
        :param im_dst: Image 2
        :param H: numpy ndarray - 3x3 homography matrix
        Output -
        :return: Inverse Warped Resulting Image
        '''


        raise NotImplementedError

        return warped_img


    def output_mosaic(self, img_src, img_warped):
        '''
        Input -
        :param img_src: Image 1
        :param img_warped: Warped Image
        Output -
        :return: Output Image Mosiac
        '''


        raise NotImplementedError

        return im_mos_out




