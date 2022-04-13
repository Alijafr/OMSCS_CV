import numpy as np
import cv2 as cv
from models import custom_model, VGG16
import torch
import sys
import argparse
import os

from recognize_house_number import read_house_numbers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",default="VGG16",required=False)
    parser.add_argument("--weights_file",default="vgg_pretrained_extra2.pt",required=False) #only for VGG16
    parser.add_argument("--input_video",required=False)
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
        
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    #out = cv.VideoWriter('output.avi', fourcc, 15.0, (640,480))  
    demo = cv.VideoWriter('video_demo.avi', fourcc, 15.0, (640,480))
    if args.input_video is None:
        cap = cv.VideoCapture(0)
    else:
        cap = cv.VideoCapture(args.input_video)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        #frame = cv.flip(frame,0)
        #out.write(frame)
        vis = read_house_numbers(frame, model, use_cuda,min_prob_thre=0.0)
        demo.write(vis)
        # Display the resulting frame
        cv.imshow('frame', vis)
        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()