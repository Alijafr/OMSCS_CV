from models import custom_model, VGG16
import torch
import sys
import cv2
import numpy
import argparse
from recognize_house_number import read_house_numbers




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",default="custom_model",required=False)
    parser.add_argument("--weights_file",default="custom_model.pt",required=False) #only for VGG16 
    parser.add_argument("--input_image",default="dataset/train/format1/2.png",required=False) #name of the output image
    parser.add_argument("--out_image",required=False) #name of the output image
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
    
    image_path =  args.input_image
    image = cv2.imread(image_path)
    use_cuda = torch.cuda.is_available()
    vis = read_house_numbers(image, model, use_cuda)
    if len(args.out_image):
        cv2.imwrite("{}.png".format(args.out_image),vis)
    cv2.imshow("vis",vis)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

