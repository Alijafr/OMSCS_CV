from models import custom_model, VGG16
import torch
import sys
import cv2
import argparse
import os
from recognize_house_number import read_house_numbers


test_folder = "./test_images/"
out_folder = "./graded_images/"

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",default="custom_model",required=False)
    parser.add_argument("--weights_file",default="custom_model.pt",required=False) #only for VGG16 
    args = parser.parse_args()
    if args.model_type == "custom_model":    
        model = custom_model(in_channels=3,num_classes= 11)
    elif args.model_type == "VGG16":
        model = VGG16(pretrained=False,in_channels=3,num_classes=11)
    else:
        print("wrong input for model_type, should be either custom_model or VGG16")
        sys.exit()
    
    images = load_images_from_folder(test_folder)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        #model trained using gpu
        model.load_state_dict(torch.load(args.weights_file,map_location=torch.device('cuda')))
    else:
        model.load_state_dict(torch.load(args.weights_file,map_location=torch.device('cpu')))
    for i in range(len(images)):
        
        vis = read_house_numbers(images[i], model, use_cuda)
        
        cv2.imwrite("{}.png".format(out_folder+str(i)),vis)
    

