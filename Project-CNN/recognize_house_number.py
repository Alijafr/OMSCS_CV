import cv2 
import numpy as np 
import torch.nn as nn
import torchvision.transforms as transforms
from train_models import apply_transforms
from models import custom_model, VGG16
import torch
import sys
import argparse


def get_bboxes_MSER(img,expand=False,percentages = [0.2,-0.2]):
    '''
    This function takes an image, and return the bbox of interest

    Parameters
    ----------
    img : cv2.image (unit8)
       Image of interest
    expand : Bool, optional
        If expand is true, the bounding boxes is expanded or shrunk based on the percentages parameters. The default is 0.
    percentages : list, optional
        The list of percentages of expansion or shrinkage e.g [0.2,-0.2]. The default is [0.2,-0.2].

    Returns
    -------
    bboxes : List
        list of the bounding boxes extracted from the image using MSER detector.

    '''
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    h_img, w_img = img.shape[:2]
    img_area=h_img*w_img
    #Create MSER detector 
    #mser = cv2.MSER_create(min_area=int(0.01*img_area),max_area=int(0.2*img_area),max_variation=0.25)
    mser = cv2.MSER_create(max_variation=0.5)
    ROIs, _ = mser.detectRegions(img)
    
    bboxes = []
    for roi in ROIs:
        x2, y2 = np.max(roi, axis = 0)
        x1, y1 = np.min(roi, axis = 0)
        #just make sure that the boudning box is reasonable
        h_box = y2-y1
        w_box = x2-x1
        if (w_box <= 0.5 * w_img and h_box <= 0.8*h_img) and (w_box>= 0.05* w_img and h_box >= 0.08 * h_img):
            #append box
            bboxes.append((x1, y1, x2, y2))
    if expand:
        bboxes = fuzz_bboxes(img,bboxes,percentages)
    return bboxes


def fuzz_bboxes(img,bboxes, percentages):
    '''
    expand , and reduce the bounding box by a certain percentages 

    Parameters
    ----------
    img : cv2.image [uint8]
        Image of interest.
    bboxes : List
        list of the bounding boxes to be expanded.
    percentages : List
        The list of percentages of expansion or shrinkage e.g [0.2,-0.2].

    Returns
    -------
    new_bboxes : TYPE
        list of the expanded bounding boxes..

    '''
    
    h_img, w_img = img.shape[:2]
    new_bboxes = []
    for i in range(len(bboxes)):
        new_bboxes.append(bboxes[i])
        x1,y1,x2,y2 = bboxes[i]
        for scale in percentages: # expand in percentage
            x1_ = int(np.clip(x1*(1-(scale/2)),0,w_img-1))
            x2_ = int(np.clip(x2*(1+(scale/2)),0,w_img-1))
            y1_ = int(np.clip(y1*(1-(scale/2)),0,h_img-1))
            y2_ = int(np.clip(y2*(1+(scale/2)),0,h_img-1))
            
            new_bboxes.append((x1_, y1_, x2_, y2_))
    
    return new_bboxes

def draw_bboxes(img,bboxes,labels=[],prob=[]):
    '''
    Helper to draw the bboxes of an image and labels if provides

    Parameters
    ----------
    img : cv2.image [uint8]
        image of interest
    bboxes : List
        list of the ROI.

    Returns
    -------
    vis : TYPE
        visualization image that contains the drawinig of the bboxes.

    '''
    vis = np.copy(img)
    for i,box in enumerate(bboxes):
        x1,y1,x2,y2 = box
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),1)
        if len(labels) :
            cv2.putText(vis, str(labels[i]), org = (x1, y2 - 3), fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = (255, 0, 0), thickness = 2, fontScale = 1.0)
            cv2.putText(vis, "p: "+str(round(prob[i],2)), org = (x1, y1 - 3), fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = (255, 0, 0), thickness = 1, fontScale = 0.7)
    return vis
    
def extract_images(image,bboxes):
    '''
    Helper to extract the images based on the ROIs


    Returns
    -------
    np.array
        return the images of ROI in numpy array format.

    '''
    images = np.zeros((len(bboxes),32,32,3),dtype=np.uint8)
    for i,box in enumerate(bboxes):
        x1,y1,x2,y2 = box
        images[i,:] =cv2.resize(image[y1:y2,x1:x2],(32,32))
    return images

def predict_labels(model,images_np,use_cuda):
    
    #apply transform to prepare the images to be inptuted to the model 
    images_tensor = apply_transforms(images_np,train=False)
    if use_cuda:
        model.cuda()
        images_tensor = images_tensor.cuda()
    model.eval()      
    output = model(images_tensor)
    #apply softmax to the output 
    output=nn.functional.softmax(output, dim=0) #apply softmax to the output
    # convert output probabilities to predicted class
    max_prob, pred_labels = output.data.max(1, keepdim=True)
    max_prob = max_prob.squeeze(1)
    pred_labels = pred_labels.squeeze(1)
    return pred_labels,max_prob

#skelton obtained from: 
#https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def nms(bboxes,max_prob=None,overlap_thresh = 0.1):
    if len(bboxes) == 0:
        return []
    
    accepted_idx = []
    #the cooredinates for each vertix 
    x1s = bboxes[:,0]
    y1s = bboxes[:,1]
    x2s = bboxes[:,2]
    y2s = bboxes[:,3]
    
    #calculate the area 
    areas = (x2s-x1s+1) * (y2s-y1s+1)
    #sort the based on their probalities 
    if max_prob !=None:
        sorted_indices = np.argsort(max_prob.numpy())#sorted from lowest to hightest
    else:
        sorted_indices = np.argsort(y2s) #sort on the lowest y2 first 
    
    while len(sorted_indices) > 0: #loop until no more ovelap above the threshold
        #start from the last index (hihest prob bbox)
        last_idx = len(sorted_indices) - 1 
        i = sorted_indices[last_idx]
        accepted_idx.append(i)
        
		#np.maximum returns a new array containing the element-wise maxima between the first arg and second 
        xx1 = np.maximum(x1s[i], x1s[sorted_indices[:last_idx]])
        yy1 = np.maximum(y1s[i], y1s[sorted_indices[:last_idx]])
        xx2 = np.minimum(x2s[i], x2s[sorted_indices[:last_idx]])
        yy2 = np.minimum(y2s[i], y2s[sorted_indices[:last_idx]])
		# compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
        overlap = (w * h) / areas[sorted_indices[:last_idx]]
		# delete all indexes from the index list that have
        sorted_indices = np.delete(sorted_indices, np.concatenate(([last_idx],
			np.where(overlap > overlap_thresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
    return list(bboxes[accepted_idx].astype("int")),accepted_idx
        
        
'''
this is the main function in this script 

'''
def read_house_numbers(image,model,use_cuda,min_prob_thre=0.03):
    '''
    This funciton shows the digits in any given image

    Parameters
    ----------
    image : uint8
        input image
    model : CNN
        CNN trained model
    use_cuda : Bool
        whether GPU is available or not.
    min_prob_thre : float, optional
        Min pro to consider as true positive. The default is 0.03.

    Returns
    -------
    TYPE
        visualization image.

    '''
    #get the bboxes
    bboxes = get_bboxes_MSER(image)
    #get the corresponding images of these ROIs
    images_np = extract_images(image, bboxes)
    if len(images_np):
        #run the images throught the model to get the probs and labels
        pred_labels, max_prob = predict_labels(model, images_np, use_cuda)
        #filer the result
        mask = (pred_labels!=10) & (max_prob >min_prob_thre) #the label is not 10 (non-digit), and it has a high prob
        #update the bboxes and pred_labels
        if use_cuda:
            #convert variable to cpu again 
            mask = mask.cpu().numpy()
            max_prob = max_prob.cpu()
            pred_labels = pred_labels.cpu()
        max_prob = max_prob[mask]
        pred_labels=pred_labels[mask]
        
        bboxes = np.array(bboxes)[mask]
        if len(bboxes)==0:
            print("No digits detected")
            return image
        else:
            #visualize the bboxes
            #vis = draw_bboxes(image, bboxes)
            
            #use NMS to remove overlapping bouding boxes
            bboxes,accepted_idx = nms(bboxes,max_prob)
            #visualize result 
            vis = draw_bboxes(image, bboxes,pred_labels.numpy()[accepted_idx],max_prob.numpy()[accepted_idx])
           
            return vis
        
    else:
        print("Opps, no ROIs detected")
        return image
        



# img = cv2.imread('dataset/train/format1/2.png')
# bboxes = get_bboxes_MSER(img)
# bboxes = nms(np.array(bboxes))   
# vis = draw_bboxes(img, bboxes)

# #vis = read_house_numbers(img, model, False)
# cv2.imshow("vis",vis)
# cv2.waitKey(0)

# cv2.destroyAllWindows()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",default="VGG16",required=False)
    parser.add_argument("--weights_file",default="vgg_pretrained.pt",required=False) #only for VGG16 
    parser.add_argument("--input_image",default="dataset/train/format1/2.png",required=False) #name of the output image
    parser.add_argument("--out_image",required=False) #name of the output image
    args = parser.parse_args()
    
    if args.model_type == "custom_model":    
        model = custom_model(in_channels=3,num_classes= 11)
        
    elif args.model_type == "VGG16":
        model = VGG16(pretrained=False,in_channels=3,num_classes=11)
    else:
        print("wrong input for model_type, should be either custom_model or VGG16")
        sys.exit()
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        #model trained using gpu
        model.load_state_dict(torch.load(args.weights_file,map_location=torch.device('cuda')))
    else:
        model.load_state_dict(torch.load(args.weights_file,map_location=torch.device('cpu')))
    image_path =  args.input_image
    image = cv2.imread(image_path)
    vis = read_house_numbers(image, model, use_cuda)
    if args.out_image != None:
        cv2.imwrite("{}.png".format(args.out_image),vis)
    cv2.imshow("vis",vis)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
