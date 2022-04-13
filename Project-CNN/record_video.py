import torch
import sys
import cv2 as cv
import argparse
import os

from recognize_house_number import read_house_numbers

if __name__ == "__main__":
   
        
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    demo = cv.VideoWriter('input.avi', fourcc, 15.0, (640,480))
    cap = cv.VideoCapture(0)
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
        demo.write(frame)
        # Display the resulting frame
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
