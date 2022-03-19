"""Problem Set 5: Object Tracking and Pedestrian Detection"""

import os

import cv2
import numpy as np

import ps5
from ps5_utils import run_kalman_filter, run_particle_filter

# I/O directories
input_dir = "input_images"
output_dir = "output"

NOISE_1 = {'x': 2.5, 'y': 2.5}
NOISE_2 = {'x': 7.5, 'y': 7.5}


def part_1b():
    print("Part 1b")

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}
    save_frames = {
        12: os.path.join(output_dir, 'ps5-1-b-1.png'),
        28: os.path.join(output_dir, 'ps5-1-b-2.png'),
        57: os.path.join(output_dir, 'ps5-1-b-3.png'),
        97: os.path.join(output_dir, 'ps5-1-b-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_1b(ps5.KalmanFilter, template_loc, save_frames,
                os.path.join(input_dir, "circle"))


def part_1c():
    print("Part 1c")

    template_loc = {'x': 311, 'y': 217}
    save_frames = {
        12: os.path.join(output_dir, 'ps5-1-c-1.png'),
        30: os.path.join(output_dir, 'ps5-1-c-2.png'),
        81: os.path.join(output_dir, 'ps5-1-c-3.png'),
        155: os.path.join(output_dir, 'ps5-1-c-4.png')
    }

    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_1c(ps5.KalmanFilter, template_loc, save_frames,
                os.path.join(input_dir, "walking"))


def part_2a():

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}

    save_frames = {
        8: os.path.join(output_dir, 'ps5-2-a-1.png'),
        28: os.path.join(output_dir, 'ps5-2-a-2.png'),
        57: os.path.join(output_dir, 'ps5-2-a-3.png'),
        97: os.path.join(output_dir, 'ps5-2-a-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_2a(
        ps5.ParticleFilter,  # particle filter model class
        template_loc,
        save_frames,
        os.path.join(input_dir, "circle"))


def part_2b():

    template_loc = {'x': 360, 'y': 141, 'w': 127, 'h': 179}

    save_frames = {
        12: os.path.join(output_dir, 'ps5-2-b-1.png'),
        28: os.path.join(output_dir, 'ps5-2-b-2.png'),
        57: os.path.join(output_dir, 'ps5-2-b-3.png'),
        97: os.path.join(output_dir, 'ps5-2-b-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_2b(
        ps5.ParticleFilter,  # particle filter model class
        template_loc,
        save_frames,
        os.path.join(input_dir, "pres_debate_noisy"))


def part_3():
    template_rect = {'x': 538, 'y': 377, 'w': 73, 'h': 117}

    save_frames = {
        20: os.path.join(output_dir, 'ps5-3-a-1.png'),
        48: os.path.join(output_dir, 'ps5-3-a-2.png'),
        158: os.path.join(output_dir, 'ps5-3-a-3.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_3(
        ps5.AppearanceModelPF,  # particle filter model class
        template_rect,
        save_frames,
        os.path.join(input_dir, "pres_debate"))


def part_4():
    template_rect = {'x': 210, 'y': 37, 'w': 103, 'h': 285}

    save_frames = {
        40: os.path.join(output_dir, 'ps5-4-a-1.png'),
        100: os.path.join(output_dir, 'ps5-4-a-2.png'),
        240: os.path.join(output_dir, 'ps5-4-a-3.png'),
        300: os.path.join(output_dir, 'ps5-4-a-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_4(
        ps5.MDParticleFilter,  # particle filter model class
        template_rect,
        save_frames,
        os.path.join(input_dir, "pedestrians"))


def part_5():
    """Tracking multiple Targets.

    Use either a Kalman or particle filter to track multiple targets
    as they move through the given video.  Use the sequence of images
    in the TUD-Campus directory.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    template_loc = [{'y': 160, 'x': 65, 'w': 90, 'h': 280,'start':1,'end':63},
                    {'y': 240, 'x': 285, 'w': 70, 'h': 100,'start':1,'end':50},
                    {'y': 175, 'x': 0, 'w': 70, 'h': 180,'start':29,'end':72}] # the third person comes at frame 29
                    
    save_frames = {
        29: os.path.join(output_dir, 'ps5-5-a-1.png'),
        56: os.path.join(output_dir, 'ps5-5-a-2.png'),
        71: os.path.join(output_dir, 'ps5-5-a-3.png')
    }
    # Define process and measurement arrays if you want to use other than the
    imgs_dir = os.path.join(input_dir, "TUD-Campus")
    
    #try kalman filter 
    # NOISE_2 = {'x': 2, 'y': 3}
    # Q = 0.01 * np.eye(4)  # Process noise array
    # R = 0.1 * np.eye(2)  # Measurement noise array
    # run_multiple_kalman_filter(ps5.KalmanFilter, imgs_dir, NOISE_2, "matching",
    #                         save_frames, template_loc[:2],Q,R)
    
    #try particle fitler 
    num_particles = 500  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 20 # Define the value of sigma for the particles movement (dynamics)
    sigma_start = 0
    out = run_multiple_particle_filter(
        ps5.MDParticleFilter_part5,  # particle filter model class
        imgs_dir,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_loc,
        sigma_start=sigma_start)
    
    


def part_6():
    """Tracking pedestrians from a moving camera.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    template_rect = {'x': 90, 'y': 35, 'w': 38, 'h': 180//2}

    save_frames = {
        60: os.path.join(output_dir, 'ps5-6-a-1.png'),
        160: os.path.join(output_dir, 'ps5-6-a-2.png'),
        186: os.path.join(output_dir, 'ps5-6-a-3.png')
    }
    
    imgs_dir = os.path.join(input_dir, "follow")
    # Define process and measurement arrays if you want to use other than the
    # default.
    num_particles = 500  # Define the number of particles
    sigma_mse = 5  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 8 # Define the value of sigma for the particles movement (dynamics)
    sigma_start = 0
    out = run_particle_filter(
        ps5.MDParticleFilter_part6,  # particle filter model class
        imgs_dir,
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_rect,
        sigma_start=sigma_start,
        alpha=0.1,
        max_weight_thre=0.2)
    

def run_multiple_kalman_filter(filter_class,
                      imgs_dir,
                      noise,
                      sensor,
                      save_frames={},
                      template_loc=None,
                      Q=0.1 * np.eye(4),
                      R=0.1 * np.eye(2)):
    kfs = []
    templates = []
    # for i in range(len(template_loc)):
    #     kf = filter_class(template_loc[i]['x'], template_loc[i]['y'], Q, R)
    #     kfs.append(kf)

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    frame_num = 0


    if sensor == "matching":
        frame = cv2.imread(os.path.join(imgs_dir, imgs_list[0]))
        for i in range(len(template_loc)):
            template = frame[template_loc[i]['y']:
                             template_loc[i]['y'] + template_loc[i]['h'],
                             template_loc[i]['x']:
                             template_loc[i]['x'] + template_loc[i]['w']]
            templates.append(template)
    else:
        raise ValueError("Unknown sensor name. Choose between 'hog' or "
                         "'matching'")

    for img in imgs_list:
        x_values = []
        y_values = []
        zx_values = []
        zy_values = []
        frame = cv2.imread(os.path.join(imgs_dir, img))
        for i in range(len(template_loc)):
            if template_loc[i]['start']==frame_num:
                kf = filter_class(template_loc[i]['x'], template_loc[i]['y'], Q, R)
                kfs.append(kf)
            elif template_loc[i]['end']==frame_num:
                kfs.remove(kfs[i])
                template_loc.remove(template_loc[i])
                

        if sensor == "matching":
            for i in range(len(kfs)):
                corr_map = cv2.matchTemplate(frame, templates[i], cv2.TM_SQDIFF)
                z_y, z_x = np.unravel_index(np.argmin(corr_map), corr_map.shape)
    
                z_w = template_loc[i]['w']
                z_h = template_loc[i]['h']
    
                z_x += z_w // 2 + np.random.normal(0, noise['x'])
                z_y += z_h // 2 + np.random.normal(0, noise['y'])

                x, y = kfs[i].process(z_x, z_y)
                x_values.append(x)
                y_values.append(y)
                zx_values.append(z_x)
                zy_values.append(z_y)

        if True:  # For debugging, it displays every frame
            out_frame = frame.copy()
            for i in range(len(template_loc)):
                #cv2.circle(out_frame, (int(zx_values[i]), int(zy_values[i])), 20, (0, 0, 255), 2)
                cv2.circle(out_frame, (int(x_values[i]), int(y_values[i])), 10, (255, 0, 0), 2)
                # cv2.rectangle(out_frame, (int(zx_values[i]) - template_loc[i]['w'] // 2, int(zy_values[i]) - template_loc[i]['h'] // 2),
                #               (int(zx_values[i]) + template_loc[i]['w'] // 2, int(zy_values[i]) + template_loc[i]['h'] // 2),
                #               (0, 0, 255), 2)
    
            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            for i in range(len(template_loc)):
                cv2.circle(frame_out, (int(x_values[i]), int(y_values[i])), 10, (255, 0, 0), 2)
                # cv2.rectangle(frame_out, (int(zx_values[i]) - template_loc[i]['w'] // 2, int(zy_values[i]) - template_loc[i]['h'] // 2),
                #               (int(zx_values[i]) + template_loc[i]['w'] // 2, int(zy_values[i]) + template_loc[i]['h'] // 2),
                #               (0, 0, 255), 2)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print('Working on frame {}'.format(frame_num))
    return 0

def run_multiple_particle_filter(filter_class, imgs_dir, template_rect,
                        save_frames={}, num_particles=100,sigma_exp=10,sigma_dyn=10,template_coords={},sigma_start=0,alpha=0.01):
    """Runs a particle filter on a given video and template.

    Create an object of type pf_class, passing in initial video frame,
    template (extracted from first frame using template_rect), and any
    keyword arguments.

    Do not modify this function except for the debugging flag.

    Args:
        filter_class (object): particle filter class to instantiate
                           (e.g. ParticleFilter).
        imgs_dir (str): path to input images.
        template_rect (dict): template bounds (x, y, w, h), as float
                              or int.
        save_frames (dict): frames to save
                            {<frame number>|'template': <filename>}.
        **kwargs: arbitrary keyword arguments passed on to particle
                  filter class.

    Returns:
        None.
    """

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    # Initialize objects
    templates = []
    pfs = []
    frame_num = 1

    # Loop over video (till last frame or Ctrl+C is presssed)
    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))
        #estimations= []
        # Extract template and initialize (one-time only)
        
        for i in range(len(template_rect)):
            if template_rect[i]['start']==frame_num:
                template = frame[int(template_rect[i]['y']):
                                 int(template_rect[i]['y'] + template_rect[i]['h']),
                                 int(template_rect[i]['x']):
                                 int(template_rect[i]['x'] + template_rect[i]['w'])]
                #templates.append(template)
                pf = filter_class(frame, template, **{"num_particles":num_particles,"sigma_exp":sigma_exp,"sigma_dyn":sigma_dyn,"template_coords":template_coords[i],"sigma_start":sigma_start,"alpha":alpha})
                pfs.append(pf)
            elif template_rect[i]['end']==frame_num:
                pfs.remove(pfs[i])
                template_rect.remove(template_rect[i])
                break
        # Process frame
        for i in range(len(pfs)):
            pfs[i].process(frame)
            #best_particle = pfs[i].particles[np.argmax(pfs[i].weights)]
            #estimations.append(best_particle)

        if True:  # For debugging, it displays every frame
            out_frame = frame.copy()
            for i in range(len(pfs)):
                pfs[i].render(out_frame)
            cv2.imshow('Tracking', out_frame)
            for i in range(len(pfs)):
                cv2.imshow('template {}'.format(i), pfs[i].template.astype(np.uint8))
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            print("frame number {} is saving".format(frame_num)) 
            frame_out = frame.copy()
            for i in range(len(pfs)):
                pfs[i].render(frame_out)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print('Working on frame {}'.format(frame_num))
    return 0
if __name__ == '__main__':
    # part_1b()
    # part_1c()
    # part_2a()
    # part_2b()
    # part_3()
    # part_4()
    # part_5()
    part_6()
