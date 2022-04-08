import numpy as np
from scipy.io import loadmat
import os
import h5py
import cv2 



#format2_mat_path_train = "dataset/train/format1/train_32x32.mat"
#format2__mat_path_test = "dataset/test/format1/test_32x32.mat"
#format1_mat_path_train = "dataset/train/format1/digitStruct.mat"
#forma1_path = "dataset/train/format1/"

#this fucntion used to load .mat data
def load_mat_data(relative_path): 
    '''
    

    Parameters
    ----------
    relative_path : dataset in .mat (format2) 

    Returns
    -------
    X : TYPE
        dataset
    Y : TYPE
        labels.

    '''
    #the data comes in .mat file (matlab format)
    
    data = loadmat(os.path.join(relative_path))
    
    X = data["X"] #images --> np.unit8
    Y = data["y"] #labels
    return X,Y
        
    

#this data is meant to prepare the dataset for in the standard foramt
def preproces_data(train_file,test_file):
    '''
    This function is meant to read the .mat files (format 2 of SVHN dataset)

    Parameters
    ----------
    train_file : String
        .mat file for training 
    test_file : String
        .mat file for training 

    Returns
    -------
    train_x : np.array
        np.array of the images to train in this format (num_images,H,W,num_channels)
    train_y : np.array
        np.array of the labels to the images to train in this format (num_images,).
    test_x : np.array
        np.array of the images to test in this format (num_images,H,W,num_channels).
    test_y : np.array
        np.array of the labels to the images to test in this format (num_images,).

    '''
    train_x , train_y = load_mat_data(train_file)
    test_x , test_y = load_mat_data(test_file)
    #the data comes in this format (w,h,ch,images)
    #need to changes it to the common format (images,w,h,ch)
    train_x = train_x.transpose((3, 0, 1, 2))
    test_x = test_x.transpose((3, 0, 1, 2))
    #labels comes in (num_images,1) --> make it (num_images,)
    train_y = train_y.squeeze(axis=1)
    test_y = test_y.squeeze(axis=1)
    #change the label for number 0 from 10 to 0
    train_y[train_y==10] = 0
    test_y [test_y==10] =0
    return train_x,train_y , test_x, test_y

#need to create a neg_dataset from format2, only images not in the

def load_forma1_data(file):
    '''
    #format1 data gives an error when using loadmat from scipy.io
    #stackoverflow -- > use h5py

    Parameters
    ----------
    file : .mat file from format 1 
        DESCRIPTION.

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    '''
    data =  h5py.File(file,'r')
    return data


def get_imageName_bbox_and_labels(index,file):
    '''
    This function is designed to get the image name, label, and boudning box of format1 dataset 

    Parameters
    ----------
    index : Index of the image to its info
    file : the opened .mat file in h5py format 
        

    Returns
    -------
    image_name : image name e.g 1.png
        
    bbox_dict : bouding boxes and their labels 
        

    '''

    #get the image name
    name = file['digitStruct/name'][index][0] #index 0 because it has a length of 1
    dataset_object = file[name] #type _hl.dataset.Datset
    image_name = ''.join(chr(char[0]) for char in dataset_object[:])
    
    #get the image label and bbox
    bbox_dict = {}
    
    bbox = file['digitStruct/bbox'][index].item() # the h5r.Reference
    #file[bbox].keys() --> <KeysViewHDF5 ['height', 'label', 'left', 'top', 'width']>
    for key in ['height', 'label', 'left', 'top', 'width']:
        val_ref = file[bbox][key]
        values = [int(file[val_ref[i].item()][0][0])
                  for i in range(len(val_ref))] if len(val_ref) > 1 else [int(val_ref[0][0])]
        
        bbox_dict[key] = values
        
    
    return image_name, bbox_dict
    

def create_non_digit_dataset(forma1_path,format1_mat_path_train,p=1.0):
    '''
    

    Parameters
    ----------
    forma1_path : String
        relative file that contains the images of the format1
    format1_mat_path_train : relative file to the .mat file of format1
        contains the names, bbox, and labels of the digits in the image
    p : float, optional
        presentage of the images in format 1 to used. The default is 0.7.

    Returns
    -------
    X : np.array
        negative generated images
    Y : TYPE
        their labels, all 10

    ''' 
    threshold = 20 #threshold used to decide the minimum width to accept
    X = []
    Y = []
    #get the data 
    data = load_forma1_data(format1_mat_path_train)
    num_images = int(p*len(data['digitStruct/name']))
    for i in range(num_images):
        #get the image name, bboxes 
        name,bbox_dict = get_imageName_bbox_and_labels(i, data)
        #open image
        img = cv2.imread(os.path.join(forma1_path, name))
        h,w = img.shape[:2]
        #get the start and end of the numbers in the image
        x_start = np.min(bbox_dict["left"])
        #y_start = np.min(bbox_dict["top"])
        x_end = np.max(np.array(bbox_dict["left"]) + np.array(bbox_dict["width"]))
        #y_end = np.max(np.array(bbox_dict["top"]) + np.array(bbox_dict["height"]))
        
        #check if the right side has larger area than the left 
        if x_start> (w-x_end) and x_start > threshold:
            neg_img = img[:,:x_start]
            X.append(cv2.resize(neg_img,(32,32)))
            Y.append(10) #non-digit
        elif x_start < (w-x_end) and x_start > threshold:
            neg_img = img[:,x_end:]
            X.append(cv2.resize(neg_img,(32,32)))
            Y.append(10) #non-digit
    
    X = np.array(X)
    Y = np.array(Y)
    #torch takes the images in forms of (num_images,num_chennels,H,W)
    #X = np.transpose(X,[0,3,2,1])

    return X,Y       
        
        
        
    
    
    
    
    
    
    
