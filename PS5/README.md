# Problem Set 5: Object Tracking and Pedestrian Detection

## NOTE: Please refer to the PDF file for instructions. GitHub markdown does not render formulae and algorithms correctly. The information is identical between the markdown and PDF, but the PDF renders more cleanly.

Please refer to [FAQs.md](FAQs.md) for frequently asked questions.

# Assignment Description

## Description
In Problem set 5, you are going to implement tracking methods for image sequences and videos.
The main algorithms you will be using are the Kalman and Particle Filters.
Methods to be used:You will design and implement a Kalman and a Particle filters from the ground up.
RULES: You may use image processing functions to find color channels, load images, find edges(such as with Canny), resize images.  Don’t forget that those have a variety of parameters andyou may need to experiment with them.
There are certain functions that may not be allowedand are specified in the assignment’s autograder Piazza post.Refer to this problem set’s autograder post for a list of banned function calls.Please do not use absolute paths in your submission code.
All paths should be relative to thesubmission directory. Any submissions with absolute paths are in danger of receiving a penalty!


## Learning Objectives

 - Identify  which  image  processing  methods  work  best  in  order  to  locate  an  object  in  ascene.
 - Learn to how object tracking in images works.
 - Explore different methods to build a tracking algorithm that relies on measurements anda prior state.
 - Create methods that can track an object when occlusions are present in the scene.

## Base Kalman Filter:
The autograder generates an environment where your code has to track a moving object. The simulated movement is different in each run. The shape to track is a circle as shown in the images below:

![image](https://github.gatech.edu/storage/user/13277/files/c165b4ec-47f6-4e1f-ab39-7f7cedf064f1)

Your Filter is initialized with the shape's position at frame 0. The sensor used is cv2.matchTemplate to feed the measurement values to process(). The autograder measures how far your prediction is to the actual center using the euclidean distance.

## Kalman Filter with noise 1 and 2:
The environment is the same as above. The coordinates obtained from cv2.matchTemplate are corrupted with some gaussian noise with zero mean and standard deviations of:
```
NOISE_1 = {'x': 2.5, 'y': 2.5}
NOISE_2 = {'x': 7.5, 'y': 7.5}
```
In these cases your Kalman filter is initialized with specific measurement and state prediction noise model matrices (Q and R). The tolerance used for these tests is greater than the base case. 

## Base particle_filter:
The particle filter is initialized with the template coordinates, the first frame, and the template cutout. The parameters that the test uses are: num_particles, sigma_mse, and sigma_dyn. See the problem set documentation for more details.

The autograder will use the information in your particles and weights array to calculate the weighted mean and therefore obtain the center estimate. We have provided how to compute this estimate in the render( ) function in the assignment files.

## Appearance Particle Filter
The idea behind this test is to simulate an environment where the object to be tracked changes in appearance.
![image](https://github.gatech.edu/storage/user/13277/files/99572157-3055-4131-9617-1390c200ad08)

Additionally, we will run your code locally to verify your algorithm's behavior with the videos provided in the problem set. Therefore, **it is important that your latest submission contains the exact files you used in order to generate the images in your report**. Given that there are several steps that use random numbers these images may not match but they should still track the object.

Your code must not fail to track the object in order to receive credit for the images you present in the report. **We will not accept regrade requests with changes in these files** so keep this in mind.

Banned function calls:

Here is the preliminary list of banned function calls:
```
cv2.KalmanFilter
cv2.matchTemplate (should not be part of the filter process)
```
This list may change as we find more examples so when in doubt please ask. In the event that you are using these functions you will see a warning message in your submission. You will receive zero points if you use any of these functions in the assignment. This check will also be performed during the grading process to tests the methods that are not graded by the autograder.

 
When coding make sure you optimize your memory usage, the autograder environment allows around 500MB per submission. Additionally, Gradescope has a global quota that is max 10 submissions in a 1h.

Note: The grade distribution for this problem set is 77.5% code and 22.5% report.

It is important to consider corner cases when implementing your Particle Filter. You will have to address cases where the object is near the image borders and how you will treat particles that may cause Indexing Errors. 

We hope you have fun with this assignment!

PS: The autograder runs your code without checking the PF precision during the first 10 frames. This allows you to use a higher number of particles to find the object in case you decide to initially place particles covering the entire image. Once the analysis of the 10-th frame starts, you should reduce your particles and weights array to the original number of particles passed by the autograder.


