### Part 1:
Q: Some of the coordinates drawn on the images are outside the image. Am I supposed to fix it? Or should I just write down the coordinates in the report? \
A: If the text outbounds the image in a few frames of a video you can leave it as it is. You can then include the coordinates in the report.


### Part 2:
Q: How far away can our vertices be from the center of the markers for part 2 of the assignment (marker detection in real scenes)? \
A: The rectangle corners have to be in the marker areas.

Q: How should we handle the "ps3-2-c_base.jpg" image that is rotated 90 degrees in terms of "top left" or "bottom right"? \
A: you can return the points in the requested order [top-left, bottom-left, top-right, bottom-right] as they appear in the image. And then draw the rectangle.


### Part 3:
Q: It seems part 3, 4, 5 all depend on accurately identifying the markers. If one marker fail, will I still receive partial grades even if these questions do not ask for markers directly? \
A: The corners should be in the marker area. Part2 and 3: -1 per missed corner. Part 4: -0.25 per missed corner. Part 5: -0.5 if one corner missed.

### Part 4:
Q: For the video portion of the assignment, will it okay for "a few" frames to glitch out, or do we need to meet the "within-the-rectangles" requirement for 100% of the frames ? \
A: Ideally, you are expected to meet the requirement for 100%. You won’t loose any points if one or two frames have are off. If we see a lot of them, then there will be a penalty.

### Part 5:
Q: Will the TAs be re-running our code for Part 5 and 6 (Augmented Reality Video)? My code works but it runs so slow (~30 mins for 1 video in part 5). \
A: We do run the code. For those parts you should be able to optimize the code to run under 2min.


### Part 8:
There are two parts to the Part 8 Harris Corner Detection. You need both functions to pass the test in Autograder:
harris_response_map: Follow the algorithm in Module 4A-L2 \
You may wish to normalize the response map after getting it from harris_response_map before non-maximum suppression, so the maximum confidence would be 1.0 and easier to debug by converting to percentage. \
nms_maxpool_numpy: 
- Suppress any point below the median
- maxpool in 2d with padding so that the output of the maxpool is same size as the non-padded Response Map. Use a ksize of 7.
- Filter out the Response Map to remove points that are not the same in maxpooled response map
- sort on confidences acquired from the filtered Response Map, get the top k values

Do not change the ransac_homography_matrix apart from the parameters p, s, e. \
While using sobel filter and gaussian blur, make sure to add padding of 0's before doing the filtering. \
Without padding with 0, blurring causes the edges of the image to have higher values causing corners to be detected on the edges of the image. I ran a few tests showing this effect and here are some results \
"image":
```[[0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]] 
 ```
 Gaussian Blur without padding
 ```
 [[0.06651863 0.03675723 0.03903017 0.03981863 0.03903017 0.03675723
  0.06651863]
 [0.03675723 0.02031151 0.0215675  0.0220032  0.0215675  0.02031151
  0.03675723]
 [0.03903017 0.0215675  0.02290116 0.0233638  0.02290116 0.0215675
  0.03903017]
 [0.03981863 0.0220032  0.0233638  0.02383578 0.0233638  0.0220032
  0.03981863]
 [0.03903017 0.0215675  0.02290116 0.0233638  0.02290116 0.0215675
  0.03903017]
 [0.03675723 0.02031151 0.0215675  0.0220032  0.0215675  0.02031151
  0.03675723]
 [0.06651863 0.03675723 0.03903017 0.03981863 0.03903017 0.03675723
  0.06651863]]
```
You can see that the corners of the image have higher value than the centers leading to problems. \
Gaussian Blur after Padding (padding removed afterward blurring) provides a much smoother blurring:
```[[0.01662966 0.01837862 0.01951509 0.01990932 0.01951509 0.01837862
  0.01662966]
 [0.01837862 0.02031151 0.0215675  0.0220032  0.0215675  0.02031151
  0.01837862]
 [0.01951509 0.0215675  0.02290116 0.0233638  0.02290116 0.0215675
  0.01951509]
 [0.01990932 0.0220032  0.0233638  0.02383578 0.0233638  0.0220032
  0.01990932]
 [0.01951509 0.0215675  0.02290116 0.0233638  0.02290116 0.0215675
  0.01951509]
 [0.01837862 0.02031151 0.0215675  0.0220032  0.0215675  0.02031151
  0.01837862]
 [0.01662966 0.01837862 0.01951509 0.01990932 0.01951509 0.01837862
  0.01662966]]
  ```
### Autograder:
Q: What are the banned functions? \
A: ```cv2.findHomography, cv2.getPerspectiveTransform, cv2.findFundamentalMat, cv2.warpPerspective, cv2.goodFeaturesToTrack, cv2.warpAffine()```

Q: What packages are allowed to be imported? \
A: ```"numpy", "scipy", "cv2", "scipy.ndimage", "scipy.ndimage.rotate", "matplotlib", "math", "itertools", "itertools.combinations", "collections", "random", "enum", "time"```

Q: What functions are allowed? \
A: Few of the functions(that students have asked/used in the previous semesters) that are allowed are below.  
But understand these functions before you use them.
```cv2.matchTemplate, numpy.rot90, cv2.pyrDown(), cv2.remap, SimpleBlobDetector, all linalg functions, cv2.fillPoly, cv2.cornerEigenValsAndVecs, approxPolyDP, scipy.ndimage.rotate...```


If you use cv2.remap, be sure to use the keyward arg for destination, as assignment may not always work, ex. cv2.remap(...,dst=imageB_copy,...)


Q: What is the order for the corners? \
A: ```[top-left, bottom-left, top-right, bottom-right]```


Q: What colour should I use for draw_box? \
A: Red.


Q. What is the video grading criteria?  \
A: You need to be within 2 pixel tolerance at least 80% of the times. For the rest of the cases, it will be partially graded.
