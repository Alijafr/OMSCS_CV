from models import custom_model, VGG16
import torch
import sys
import cv2
import argparse
import os
from recognize_house_number import read_house_numbers


test_folder = "./dataset/test_images/"
out_folder = "./graded_images/"
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",default="custom_model",required=False)
    parser.add_argument("--weights_file",default="custom_model.pt",required=False) #only for VGG16 
    args = parser.parse_args()
    if args.model_type == "custom_model":    
        model = custom_model(in_channels=3,num_classes= 11)
        model.load_state_dict(torch.load(args.weights_file))
    elif args.model_type == "VGG16":
        model = VGG16(pretrained=False,in_channels=3,num_classes=11)
        model.load_state_dict(torch.load(args.weights_file))
    else:
        print("wrong input for model_type, should be either custom_model or VGG16")
        sys.exit()
    
    image_paths =  [f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))]
    for i, path in enumerate(image_paths):
        image = cv2.imread(path)
        use_cuda = torch.cuda.is_available()
        vis = read_house_numbers(image, model, use_cuda)
        
        cv2.imwrite("{}.png".format(args.test_folder+str(i)),vis)
    

