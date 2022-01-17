Frequently asked questions: Please check for your questions here before asking below.  

Q: Where can I find LaTeX files for the report?
A: You will find them attached, along with other implementation files. 

Q : For Question 1, we have to pick two images - ps1-1-a-1.png and ps1-1-a-2.png. Is it necessary that these are .png files? Could we use other image formats (such as .jpg)?
A : Since .jpg has its own compression, and the PDF will eventually get compressed, it is suggested that you have a .png file.

Q : Question 1 talks about "tall" and "wide" images. Does this require that one image have more rows than columns and the other vice- versa?
A : Yes.

Q: What are ps1-1-a-1.png and ps1-1-a-2.png?
A: ps1-1-a-1.png and ps1-1-a-2.png are the names used in the report for the sub question, and to refer them in the rest of the assignments. For images, the naming convention is assignment-questionNum-subQuestionNum-....png.

Q : For Question 4, what does it mean for the pixel values to be normalized and scaled using a min max method so they are in [0, 255]?
A : x′=(x−min(x))/(max(x)−min(x))∗255

Q : For copy_paste_middle(), what does the center of the image refer to when the image shapes are odd numbers?
A : The centers will be rounded down.

Q : What is the protocol for writing text obtained from other sources (such as books or websites)?
A : Text taken from other sources must be cited. In case it is taken from a website, provide the link. If it is a paper, please add the reference.

Q : In shift_image_left(), what does shifting to the left by 2 pixels mean?
A : You can remove the first 2 columns, and add 2 blank columns to the right and replicate the last 2 columns on the right.

Q : Do the instructors use the same autograder tests when grading the assignment? So, if all the autograder test cases have passed when the student submits, does the student get full grade for that problem?
A : We use the output of the autograder for the code grade. But, we do go through the code to see if you have met all the requirements and if we feel something is off (hardcoding the answers to test cases, for example), we might run the code. Do keep in mind that there is the report too. A 100 on the autograder doesn't guarantee a 100 on the report (especially if you leave it blank - be sure to double check your submission!)

Q : What are the functions from OpenCV that we can use for this assignment?
A : cv2.copyMakeBorder, cv2.normalize, cv2.split, cv2.merge

Q : What does a successful submission look like?
A : Refer to the assignment write up for PS1 - autograder’s feedback in the terminal window is a good indicator. It is also good practice to take a look at your history of submissions on gradescope

Q : In Question 5, should be gaussian noise added be zero mean? Or should be mean be 128 (which is in the middle of [0, 255])?
A : The mean of the Gaussian should be 0. Gaussian noise with mean 128 would simply be adding highly positive values to the image and making it brighter.

Q: If we pass everything in the last submission, will we get all gradescope points, or TAs will run our code in Gradescope again to finalize the points based on the new running result?
A: If you pass all the tests in gradescope, we will give you those points. But we do go through the code to see if you have met all the requirements and if we feel something is off, we might run the code.

Q: Username and password failed (Do you use two-factor?) errors.
A: Make sure you are using your gtid as your username (i.e. jsmith123) and not your email alias name. It should be the same you use to login to Canvas.

Q: What submission is graded?
A: The last submissions on gradescope for the report. (Make sure all required files are in the folder)

Q: Will there be more test cases for the final grade after passing all test cases?
A: No, unless specified.

Q: Does PS1 have challenge component?
A: No, it does not.

Q: filter2D gives an error on Gradescope.
A: The filter2D function in openCV takes in inputs of the formats CV_8U, CV_16S, CV_32F, CV_64F only. In your function, if you haven't checked for the image type, then you could run into that error. The images generated in the autograder doesn't necessarily have to be of the format required for these functions.

Q: The writeup says that picture should be rectangular in shape (NOT square). However in the hyperlink wrt some vision examples, they are all square-256*256, 512*515 and 1024*1024. What should I do?
A: Use opencv functions or other tools to either crop or reshape the image.

Q: extract_green function (and the other similar ones) should we assume the image format is BGR
A: OpenCV usually reads an image in the BGR format. 

Q: How to submit the report?
A: login to gradescope and submit it.

Q : The assignment indicates there is a LaTeX template we can clone. I've never used LaTeX and thought I'd just write the report in Open Office and save it as a pdf. Are we required to use LaTeX? 
A : We only allow the use of the LaTeX template from this semester onwards. 

Q : What are some good LaTeX packages and editors?
A : You would want to install TeX-Live (https://www.tug.org/texlive/acquire-netinstall.html) and Texmaker (http://www.xm1math.net/texmaker/download.html). Georgia Tech also offers professional version Overleaf https://www.overleaf.com/edu/gatech

Q: When I write report, can I add extra picture to support my answer? 
A: Yes, as long as you don’t add or remove a slide. (pptx or latex) 

Q: If you get this error while submitting on Gradescope:

```Test Failed: 3 != 0 : Missing some required files! , experiment.py, ps1-1-a-1.png, ps1-1-a-2.png```

A: One or more of the required files are missing and need to be uploaded to the latest submission. 

Q: I get the following error, what do I do?

```The autograder failed to execute correctly. Please ensure that your submission is valid. Contact your course staff for help in debugging this issue. Make sure to include a link to this page so that they can help you most effectively.```

A: Please check for display functions like cv2.imshow, or save functions in ps1.py. Make sure you do not import any additional packages within ps1.py. If you still face the error, contact us.

Q: Can we use additional slides to answer the text question?
A: You are highly discouraged from using additional slides. If you aren't able to answer the question in the given slide, please give the questions another thought. Having said that, 1-2 sentences will NOT be accepted either.


General points:

- It is okay if the input image gets rewritten in 1-a. The outputs have to have the name ps1-1-a-1.png and ps1-1-a-2.png.

- Citation can go at the bottom of the report AND the function in the code.

- It is good to know all the math in the course, but the final exam is open book and non proctored.

- "Check out the xyz() function in opencv" is not giving help, but giving the values of the parameters is.

- Difference between using min max, and mean and std for normalisation - think in terms of the distribution of the data. mean and std are parameters of a gaussian distribution.
