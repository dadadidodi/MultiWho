import numpy as np
import cv2
import sys
sys.path.append("../")
import os 
import utils 

# Extract the video features from down-sampled video files 
def extract_image(inpath, outpath):
    if not os.path.exists(inpath):
        pass 
    vidcap = cv2.VideoCapture(inpath)
    success, image = vidcap.read()
    count = 0 
    while success:
        print("New frame read %s"%inpath, success, outpath)
        cv2.imwrite(outpath.replace('.', '%d.'%count), image)
        success, image = vidcap.read()
        count += 1

def extract_surf(config):
    pass 

if __name__=="__main__":
    print(cv2.__version__)
    inpath = '~/11775_data/down_samp_video/HVC933.mp4'
    outpath = inpath.replace('down_samp_video', 'surf_images')
    outpath = outpath.replace('mp4', 'jpg')
    print(inpath, outpath)
    extract_image(inpath, outpath)
