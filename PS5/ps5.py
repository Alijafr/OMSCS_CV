"""Problem Set 5: Object Tracking and Pedestrian Detection"""

import cv2
import numpy as np
import os

from ps5_utils import run_kalman_filter, run_particle_filter

# I/O directories
input_dir = "input"
output_dir = "output"



# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.array([[init_x], [init_y], [0.], [0.]])  # state
        self.Q = Q
        self.R = R
        self.state_predicted = np.zeros_like(self.state)
        self.P = np.zeros_like(self.Q)
        #initalize P for x,y
        self.P[0][0] = 10
        self.P[1][1] = 10
        self.dt = 1.0
        #D is the transition matrix
        self.D = np.array([[1.0, 0.0,self.dt,0.0],
                           [0.0,1.0,0.0,self.dt],
                           [0.0,0.0,1.0,0.0],
                           [0.0,0.0,0.0,1.0]])
        #M is the sensor matrix
        self.M = np.array([[1.0,0.0,0.0,0.0],
                           [0.0,1.0,0.0,0.0]])

    def predict(self):
        self.state_predicted = self.D @ self.state
        self.P = self.D@self.P@self.D.T +self.Q

    def correct(self, meas_x, meas_y):
        Y = np.array([[meas_x],
                           [meas_y]])
        K = self.P@self.M.T@np.linalg.inv(self.M@self.P@self.M.T + self.R)
        
        self.state = self.state_predicted + K@(Y - self.M@self.state_predicted)
        I = np.eye(4)
        self.P = (I-K@self.M)@self.P
    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0][0], self.state[1][0]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.template = template
        self.frame = frame
        self.h ,self.w  = self.frame.shape[:2]
        self.h_temp = self.template.shape[0]
        self.w_temp = self.template.shape[1]
        #xy_min = [0, 0]
        #xy_max = [ self.w-1, self.h-1] # (x,y)
        #self.particles = np.random.uniform(low=xy_min, high=xy_max, size=(self.num_particles,2))  # Initialize your particles array. Read the docstring.
        templete_mean = np.array([self.template_rect["x"],self.template_rect["y"]])
        cov = 10*np.eye(2)
        self.particles = np.random.multivariate_normal(templete_mean, cov,size=(self.num_particles)) #initialize to the initial value of the particles
        self.weights = np.ones(self.num_particles)/self.num_particles  # Initialize your weights array. Read the docstring.
        # Initialize any other components you may need when designing your filter.

      

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        diff_2 = (template - frame_cutout)**2
        return diff_2.sum()
        

    def resample_particles(self,partcles = None):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.

        Returns:
            numpy.array: particles data structure.
        """
        if partcles == None:
            #to alter the number of particles 
            particles = self.num_particles
            
        
        indices = np.random.choice(np.arange(self.num_particles),size=particles,p=self.weights)
        self.particles = self.particles[indices]
        self.num_particles=  particles #in case the number of the partacles changed
        
        return self.particles
        

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        #resample the partcles according to their weights with replacement
        self.particles = self.resample_particles() 
        #update the particles assuming random movments 
        self.particles += np.random.normal(0,self.sigma_dyn**2,size=self.particles.shape)
        #update the weights
        for i in range(self.num_particles):
            particle = self.particles[i]
            #make sure that the edge situation is handled
            x_left = np.clip(int(particle[0] - 0.5*self.w_temp), 0, frame.shape[1]-1)
            x_right = np.clip(int(particle[0] + 0.5*self.w_temp), 0,frame.shape[1]-1)
            y_up = np.clip(int(particle[1] - 0.5*self.h_temp), 0, frame.shape[0]-1)
            y_down = np.clip(int(particle[1] + 0.5*self.h_temp), 0, frame.shape[0]-1)
            
            if x_left == 0: #croped from the left
                if y_up == 0: #croped both left and up 
                    #get the cut image
                    frame_cutout = frame[0:y_down, 0:x_right,:]
                    template = self.template[self.h_temp-frame_cutout.shape[0]:,self.w_temp-frame_cutout.shape[1]:,:]
                    
                elif y_down == frame.shape[0]-1:
                    #croped left and down
                    print(y_up)
                    print(y_down)
                    print(self.h_temp)
                    frame_cutout = frame[y_up:, 0:x_right,:]
                    template = self.template[:frame_cutout.shape[0],self.w_temp-frame_cutout.shape[1]:,:]
                    print(frame_cutout.shape)
                    print(template.shape)
                else:
                    #only left cropped
                    frame_cutout = frame[y_up:y_down, 0:int(particle[0]+0.5*self.w_temp)]
                    template = self.template[:frame_cutout.shape[0],self.w_temp-frame_cutout.shape[1]:]
            
            elif x_right == frame.shape[1]-1:
                if y_up == 0: #croped both left and up 
                    #get the cut image
                    frame_cutout = frame[0:int(particle[1]+0.5*self.h_temp), x_left:]
                    template = self.template[self.h_temp-frame_cutout.shape[0]:,:frame_cutout.shape[1]]
                elif y_down == frame.shape[0]-1:
                    #croped left and down
                    frame_cutout = frame[y_up:, x_left:]
                    template = self.template[:frame_cutout.shape[0],:frame_cutout.shape[1]]
                else:
                    #only left cropped
                    frame_cutout = frame[y_up:y_down, x_left:]
                    template = self.template[:frame_cutout.shape[0],:frame_cutout.shape[1]]
                    
            else:
                #no cropping 
                frame_cutout = frame[y_up:y_down, x_left:x_right]
                template = self.template[:frame_cutout.shape[0],:frame_cutout.shape[1]]
                
            MSE = self.get_error_metric(template, frame_cutout)
            self.weights[i] = np.exp(-MSE/(2*self.sigma_exp**2))  
                    
            
            
        #normalize the weights 
        self.weights = self.weights/self.weights.sum()
        
    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]
            cv2.circle(frame_in, (self.particles[i, 0],self.particles[i, 1]), 5, (0,0,255),-1)

        
        center = [int(x_weighted_mean),int(y_weighted_mean)]
        #make sure that the edge situation is handled 
        if center[0] - int(self.w_temp/2) < 0:
            #the particles is in the left of the image 
            new_w = center[0] #just take the part that is in the image
        elif center[0] + int(self.w_temp/2) > frame_in.shape[1]:
            new_w = frame_in.shape[1] - center[0]
        else:
            new_w = self.w_temp
        
        if center[1] - int(self.h_temp/2) < 0:
            #the particles is in the left of the image 
            new_h = center[1] #just take the part that is in the image
        elif center[1] + int(self.h_temp/2) > frame_in.shape[0]:
            new_h = frame_in.shape[0] - center[1]
        else:
            new_h = self.h_temp
        
        pt1 = ( int(center[0] -0.5*new_w),int(center[1]-0.5*new_h))
        pt2 = ( int(center[0] +0.5*new_w),int(center[1]+0.5*new_h))
        cv2.rectangle(frame_in, pt1, pt2, (255,0,0))


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        raise NotImplementedError


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        raise NotImplementedError


def part_1b(obj_class, template_loc, save_frames, input_folder):
    Q = 0.1 * np.eye(4)  # Process noise array
    R = 0.1 * np.eye(2)  # Measurement noise array
    NOISE_2 = {'x': 7.5, 'y': 7.5}
    out = run_kalman_filter(obj_class, input_folder, NOISE_2, "matching",
                            save_frames, template_loc, Q, R)
    return out


def part_1c(obj_class, template_loc, save_frames, input_folder):
    Q = 0.1 * np.eye(4)  # Process noise array
    R = 0.1 * np.eye(2)  # Measurement noise array
    NOISE_1 = {'x': 2.5, 'y': 2.5}
    out = run_kalman_filter(obj_class, input_folder, NOISE_1, "hog",
                            save_frames, template_loc, Q, R)
    return out


def part_2a(obj_class, template_loc, save_frames, input_folder):
    num_particles = 0  # Define the number of particles
    sigma_mse = 0  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 0  # Define the value of sigma for the particles movement (dynamics)

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_2b(obj_class, template_loc, save_frames, input_folder):
    num_particles = 0  # Define the number of particles
    sigma_mse = 0  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 0  # Define the value of sigma for the particles movement (dynamics)

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_3(obj_class, template_rect, save_frames, input_folder):
    num_particles = 0  # Define the number of particles
    sigma_mse = 0  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 0  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0  # Set a value for alpha

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        # input video
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        alpha=alpha,
        template_coords=template_rect)  # Add more if you need to
    return out


def part_4(obj_class, template_rect, save_frames, input_folder):
    num_particles = 0  # Define the number of particles
    sigma_md = 0  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 0  # Define the value of sigma for the particles movement (dynamics)

    out = run_particle_filter(
        obj_class,
        input_folder,
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_md,
        sigma_dyn=sigma_dyn,
        template_coords=template_rect)  # Add more if you need to
    return out
