"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os

from helper_classes import WeakClassifier, VJ_Classifier


# assignment code
def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   (tuple): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    """
    
    images_files = [f for f in os.listdir(folder)]
    y = np.zeros(len(images_files),dtype=np.int16)
    X = np.zeros((len(images_files),size[0]*size[1]))
    for i in range(len(images_files)):
        img = cv2.imread(os.path.join(folder,images_files[i]),0)
        img = cv2.resize(img,size,interpolation=cv2.INTER_CUBIC)
        X[i,:] = img.flatten()
        split_words = images_files[i].split(".")
        y[i] = int(split_words[0][-2:])
        
    return X,y
        


def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """
    M = X.shape[0]
    data_set = np.hstack((X,y[:,np.newaxis]))
    shuffled = np.random.permutation(data_set)
    
    train_x = shuffled[:int(p*M),:-1]
    train_y = shuffled[:int(p*M),-1]
    
    test_x = shuffled[int(p*M):,:-1]
    test_y = shuffled[int(p*M):,-1]
    
    return train_x,train_y,test_x,test_y
    
    


def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """
    return np.mean(x,axis=0)

    


def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                          col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """
    #M = X.shape[0]
    u = get_mean_face(X)
    
    #sigma = ((X-u)@(X-u).T) #this is wrong, you want to sum over the number of images 
    
    #no need to normalized since np.linalg.eigh gives a normalized eigen vectors 
    sigma = (X-u).T@(X-u)
    #get the eigenvalues and eigenvectors
    eigvalues, eigvectors = np.linalg.eigh(sigma)
    #reverse the eig values and vectors (given in asending order)
    eigvalues = eigvalues[::-1][:k]
    #reverse the matrix in the columns directoin then pick the highest k eigenvectors 
    eigvectors =eigvectors[:,::-1][:,:k] 
    
    return eigvectors, eigvalues

class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    def train(self):
        """Implement the for loop shown in the problem set instructions."""
        for i in range(self.num_iterations):
            self.weights /= np.sum(self.weights)
            weakLearner= WeakClassifier(self.Xtrain,self.ytrain,self.weights)
            weakLearner.train()
            y_pre = weakLearner.predict(self.Xtrain.T)
            mismatch = [self.weights[i] for i in range(self.Xtrain.shape[0]) if y_pre[i] != self.ytrain[i]]
            ej= np.sum(mismatch)
            alpha = 0.5*np.log((1-ej)/ej)
            self.weakClassifiers.append(weakLearner)
            self.alphas.append(alpha)
            if ej > self.eps:
                #update weights so that the mismatch are focused on more
                self.weights *= np.exp(-y_pre*alpha*self.ytrain)
            else:
                return 
            
            

    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        y_pred = self.predict(self.Xtrain)
        corrects_list = [1 for i in range(len(y_pred)) if y_pred[i]==self.ytrain[i]]
        corrects = len(corrects_list)
        incorrects = self.ytrain.shape[0] - corrects
        return corrects, incorrects 
        

    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        classifers_result = np.zeros((len(self.weakClassifiers),X.shape[0]))
        for i in range(len(self.weakClassifiers)):
            classifers_result[i,:] = self.alphas[i]*self.weakClassifiers[i].predict(X.T)
        
        y_pre = np.sum(classifers_result,axis=0)
        
        return np.sign(y_pre)


class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        two_horizontal_feature = np.zeros((shape),dtype=np.uint8)
        y_start , x_start = self.position
        h,w = self.size
        #the whight side 
        two_horizontal_feature[y_start:y_start+(h//2),x_start:x_start+w] = 255
        #do the gray feature 
        two_horizontal_feature[y_start+(h//2):y_start+h,x_start:x_start+w] = 126 

        
        return two_horizontal_feature

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        
        two_vertical_feature = np.zeros((shape),dtype=np.uint8)
        y_start , x_start = self.position
        h,w = self.size
        #the whight side 
        two_vertical_feature[y_start:y_start+h,x_start:x_start+(w//2)] = 255
        #do the gray feature 
        two_vertical_feature[y_start:y_start+h,x_start+(w//2):x_start+w] = 126 
        
        return two_vertical_feature

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        
        three_horizontal_feature = np.zeros((shape),dtype=np.uint8)
        y_start , x_start = self.position
        h,w = self.size
        #the whight side 
        three_horizontal_feature[y_start:y_start+int(h/3),x_start:x_start+w] = 255
        #do the gray feature 
        three_horizontal_feature[y_start+int(h/3):y_start+(2*int(h/3)),x_start:x_start+w] = 126 
        #the whight side 
        three_horizontal_feature[y_start+(2*int(h/3)):y_start+h,x_start:x_start+w] = 255
        
        return three_horizontal_feature
        

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        three_vertical_feature = np.zeros((shape),dtype=np.uint8)
        y_start , x_start = self.position
        h,w = self.size
        #the whight side 
        three_vertical_feature[y_start:y_start+h,x_start:x_start+int(w/3)] = 255
        #do the gray feature 
        three_vertical_feature[y_start:y_start+h,x_start+int(w/3):x_start+2*int(w/3)] = 126 
        #the whight side 
        three_vertical_feature[y_start:y_start+h,x_start+2*int(w/3):x_start+w] = 255

        return three_vertical_feature

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        square_feature = np.zeros((shape))
        y_start , x_start = self.position
        h,w = self.size
        #gray
        square_feature[y_start:y_start+(h//2),x_start:x_start+(w//2)]=126
        #whight
        square_feature[y_start:y_start+(h//2),x_start+(w//2):x_start+w]=255
        #whight again
        square_feature[y_start+(h//2):y_start+h,x_start:x_start+(w//2)]=255
        #gray
        square_feature[y_start+(h//2):y_start+h,x_start+(w//2):x_start+w]=126
        
        return square_feature
        

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("output/{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("output/{}.png".format(filename), X)

        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """
        ii = np.float64(ii)
        y,x = self.position
        #x,y = self.position
        h,w = self.size
        
        #exludes the edge points (so that the result sum iclude the corners points)
        if x>0:
            x -= 1
        elif x ==0:#edge case, padd
            ii = cv2.copyMakeBorder(ii, 0, 0, 1, 0, cv2.BORDER_CONSTANT, None, value = 0)
        if y>0:
            y -= 1
        elif y==0:#edge case, padd
            ii = cv2.copyMakeBorder(ii, 1, 0, 0, 0, cv2.BORDER_CONSTANT, None, value = 0)
            
       
        #check the type of the feature
        score = 0
        #two horiziontal feature
        if self.feat_type == (2,1):
            #add the sum of whight, and substract the sum of gray
            
            #whight sum (half the rectangle)
            A = ii[y,x]
            B = ii[y,x+w]
            C = ii[y+int(h/2),x]
            D = ii[y+int(h/2),x+w]
            score += (D+A-B-C)
            
            
            #the gray (second half the rectangle)
            A = ii[y+int(h/2),x]
            B = ii[y+int(h/2),x+w]
            C = ii[y+h,x]
            D = ii[y+h,x+w]
            score -= (D+A-B-C)
            
        
        #two vertical feature 
        elif self.feat_type == (1, 2):
            #whight sum (half the rectangle)
            A = ii[y, x]
            B = ii[y, x + int(w / 2)]
            C = ii[y + h, x]
            D = ii[y + h, x + int(w / 2)]
            score += (D+A-B-C)
            #the gray (second half the rectangle)
            A = ii[y, x + int(w / 2)]
            B = ii[y, x + w]
            C = ii[y + h, x + int(w / 2)]
            D = ii[y + h, x + w]
            score -= (D+A-B-C)
            
            
        #three horiziontal feature
        elif self.feat_type == (3,1):
            
            #whight sum (third of the rectangle)
            A = ii[y,x]
            B = ii[y,x+w]
            C = ii[y+int(h/3),x]
            D = ii[y+int(h/3),x+w]
            score += (D+A-B-C)
            
            # gray (second third of the rectangle)
            A = ii[y+int(h/3),x]
            B = ii[y+int(h/3),x+w]
            C = ii[y+2*int(h/3),x]
            D = ii[y+2*int(h/3),x+w]
            score -= (D+A-B-C)
            
            #whight (last third )
            A = ii[y+2*int(h/3),x]
            B = ii[y+2*int(h/3),x+w]
            C = ii[y+h,x]
            D = ii[y+h,x+w]
            score += (D+A-B-C)
        
        #three vertical feature 
        elif self.feat_type == (1,3):
            
            #whight sum (third of the rectangle)
            A = ii[y,x]
            B = ii[y,x+int(w/3)]
            C = ii[y+h,x]
            D = ii[y+h,x+int(w/3)]
            score += (D+A-B-C)
            
            #gray (second third of  the rectangle)
            A = ii[y,x+int(w/3)]
            B = ii[y,x+2*int(w/3)]
            C = ii[y+h,x+int(w/3)]
            D = ii[y+h,x+2*int(w/3)]
            score -= (D+A-B-C)
            
            #whight sum (last third )
            A = ii[y,x+2*int(w/3)]
            B = ii[y,x+w]
            C = ii[y+h,x+2*int(w/3)]
            D = ii[y+h,x+w]
            score += (D+A-B-C)
        
        #square feature
        elif self.feat_type == (2,2):
            
            #left top (gray)
            A = ii[y,x]
            B = ii[y,x+int(w/2)]
            C = ii[y+int(h/2),x]
            D = ii[y+int(h/2),x+int(w/2)]
            score -= (D+A-B-C)
            
            #right top (whight)
            A = ii[y,x +int(w/2)]
            B = ii[y,x+w]
            C = ii[y+int(h/2),x + int(w/2)]
            D = ii[y+int(h/2),x+w]
            score += (D+A-B-C)
            
            #left down (whight)
            A = ii[y+int(h/2),x]
            B = ii[y+int(h/2),x+int(w/2)]
            C = ii[y+h,x]
            D = ii[y+h,x+int(w/2)]
            score += (D+A-B-C)
            
            #right down (gray)
            A = ii[y+int(h/2),x+int(w/2)]
            B = ii[y+int(h/2),x+w]
            C = ii[y+h,x+int(w/2)]
            D = ii[y+h,x+w]
            score -= (D+A-B-C)
    
        return score 
            
            
            
            
            
        
            


def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """
    integral_images = []
    for i in range(len(images)):
        integral_image = np.cumsum(np.cumsum(images[i],0),1)
        integral_images.append(integral_image)
    
    return integral_images

class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """
    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))
        self.threshold = 1.0

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in FeatureTypes.items():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei-1, sizej-1]))
        self.haarFeatures = haarFeatures

    def set_threshold(self, threshold):
        self.threshold = threshold

    def init_train(self):
        """ This function initializes self.scores, self.weights

        Args:
            None

        Returns:
            None
        """
    
        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.
        if not self.integralImages or not self.haarFeatures:
            print("No images provided. run convertImagesToIntegralImages() first")
            print("       Or no features provided. run creatHaarFeatures() first")
            return

        self.scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        print(" -- compute all scores --")
        for i, im in enumerate(self.integralImages):
            self.scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                           2*len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                           2*len(self.negImages))
        self.weights = np.hstack((weights_pos, weights_neg))

    def train(self, num_classifiers):
        """ Initialize and train Viola Jones face detector

        The function should modify self.weights, self.classifiers, self.alphas, and self.threshold

        Args:
            None

        Returns:
            None
        """
        self.init_train()
        print(" -- select classifiers --")
        for i in range(num_classifiers):
            self.weights /= np.sum(self.weights)
            #each row in the score represent the sum of each feature in the image
            vj = VJ_Classifier(self.scores,self.labels,self.weights)
            vj.train()
            et = vj.error
            self.classifiers.append(vj)
            B = et/(1.0-et)
            
            #to find ei for all images (whether to classifed correctly or not)
            y_pre = vj.predict(self.scores.T)
            #calcualte the new weights based on the equation provided in the instructions 
            #paper: ei = 0 if correct, ei=1 if wrong
            #instuctions: ei = -1 if correct, ei=1 if wrong
            self.weights = [self.weights[i]*np.power(B,2) if y_pre[i] == self.labels[i] else self.weights[i]*np.power(B,0) for i in range(len(self.integralImages))]
            self.alphas.append(np.log(1.0/B))


    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        ii = convert_images_to_integral_images(images)

        scores = np.zeros((len(ii), len(self.haarFeatures)))

        # Populate the score location for each classifier 'clf' in
        # self.classifiers.

        # Obtain the Haar feature id from clf.feature

        # Use this id to select the respective feature object from
        # self.haarFeatures

        # Add the score value to score[x, feature id] calling the feature's
        # evaluate function. 'x' is each image in 'ii'

        result = []

        # Append the results for each row in 'scores'. This value is obtained
        # using the equation for the strong classifier H(x).
        for clf in self.classifiers:
            #get the score for each ID of the feature that produced that least error in the classifer 
            id_feature = clf.feature
            #get the feature that supposed to be used for this classifer 
            feature = self.haarFeatures[id_feature]
            #now you can get the needed scores to evaluate the weak learners 
            for i in range(len(ii)):
                scores[i,id_feature] = feature.evaluate(ii[i])

        for x in scores:
            #it is easier to do in loop becasue it three is a comparision ">="
            #each image now, has to go through all the classifers 
            left_side = np.sum([self.alphas[i]*self.classifiers[i].predict(x) for i in range(len(self.classifiers))])
            right_side = 0.5*np.sum(self.alphas)
            if left_side>=right_side:
                result.append(1)
            else:
                result.append(-1)
                
        
        return result

    def faceDetection(self, image, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """
        image_copy = np.copy(image)
        #the windos size from the instructions is 24x24 pixels
        window_size = (24,24)
        
        stride = 12
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h,w = img_gray.shape
        num_vertical_slides = h//stride
        num_horiziontal_slides = w//stride
        #the 2 here is overfitting to get the image as best as possible 
        img_gray = img_gray[2:int(h/window_size[0])*window_size[0],:int(w/window_size[1])*window_size[1]]
        for i in range(num_vertical_slides):
            for j in range (num_horiziontal_slides):
                window = img_gray[i*stride:(i*stride)+window_size[0],j*stride:(j*stride)+window_size[1]]
                if window.shape == window_size:#laziness --> to solve the stride problem of having mismatch shape
                    result = self.predict([window])
                    if result[0]==1:
                        #draw a rectangle
                        p1 = (j*stride,i*stride)
                        p2 = (p1[0]+window_size[1],p1[1]+window_size[0])
                        cv2.imwrite("face_{}_{}.png".format(i,j), window)
                        cv2.rectangle(image_copy, p1, p2, (0,255,0),1)
        cv2.imwrite(filename, image_copy)

class CascadeClassifier:
    """Viola Jones Cascade Classifier Face Detection Method

    Lesson: 8C-L2, Boosting and face detection

    Args:
        f_max (float): maximum acceptable false positive rate per layer
        d_min (float): minimum acceptable detection rate per layer
        f_target (float): overall target false positive rate
        pos (list): List of positive images.
        neg (list): List of negative images.

    Attributes:
        f_target: overall false positive rate
        classifiers (list): Adaboost classifiers
        train_pos (list of numpy arrays):  
        train_neg (list of numpy arrays): 

    """
    def __init__(self, pos, neg, f_max_rate=0.30, d_min_rate=0.70, f_target = 0.07):
        
        train_percentage = 0.85

        pos_indices = np.random.permutation(len(pos)).tolist()
        neg_indices = np.random.permutation(len(neg)).tolist()

        train_pos_num = int(train_percentage * len(pos))
        train_neg_num = int(train_percentage * len(neg))

        pos_train_indices = pos_indices[:train_pos_num]
        pos_validate_indices = pos_indices[train_pos_num:]

        neg_train_indices = neg_indices[:train_neg_num]
        neg_validate_indices = neg_indices[train_neg_num:]

        self.train_pos = [pos[i] for i in pos_train_indices]
        self.train_neg = [neg[i] for i in neg_train_indices]

        self.validate_pos = [pos[i] for i in pos_validate_indices]
        self.validate_neg = [neg[i] for i in neg_validate_indices]

        self.f_max_rate = f_max_rate
        self.d_min_rate = d_min_rate
        self.f_target = f_target
        self.classifiers = []

    def predict(self, classifiers, img):
        """Predict face in a single image given a list of cascaded classifiers

        Args:
            classifiers (list of element type ViolaJones): list of ViolaJones classifiers to predict 
                where index i is the i'th consecutive ViolaJones classifier
            img (numpy.array): Input image

        Returns:
            Return 1 (face detected) or -1 (no face detected) 
        """

        # TODO
        raise NotImplementedError

    def evaluate_classifiers(self, pos, neg, classifiers):
        """ 
        Given a set of classifiers and positive and negative set
        return false positive rate and detection rate 

        Args:
            pos (list): Input image.
            neg (list): Output image file name.
            classifiers (list):  

        Returns:
            f (float): false positive rate
            d (float): detection rate
            false_positives (list): list of false positive images
        """

        # TODO
        raise NotImplementedError

    def train(self):
        """ 
        Trains a cascaded face detector

        Sets self.classifiers (list): List of ViolaJones classifiers where index i is the i'th consecutive ViolaJones classifier

        Args:
            None

        Returns:
            None
             
        """
        # TODO
        raise NotImplementedError


    def faceDetection(self, image, filename="ps6-5-b-1.jpg"):
        """Scans for faces in a given image using the Cascaded Classifier.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """
        raise NotImplementedError