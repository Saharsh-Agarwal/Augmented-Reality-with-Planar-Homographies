import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from matchPics import matchPics
from opts import get_opts
from planarH import computeH_ransac
from planarH import compositeH
import subprocess as sp
import multiprocessing as mp
#from helper import plotMatches

opts = get_opts()

#Write script for Q3.2
##############################################################

path_book = "../data/book.mov"
path_ars = "../data/ar_source.mov"

capbook = cv2.VideoCapture(path_book)
capars = cv2.VideoCapture(path_ars)

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_cover_gray = cv2.cvtColor(cv_cover,cv2.COLOR_BGR2GRAY)

framesbook = []
framesars = []

prev_frame_time = 0
new_frame_time = 0
count = 0

if capbook.isOpened()== False or capars.isOpened() == False:
    print("Error opening video stream or file")

while(capbook.isOpened() and capars.isOpened()):
    retbook, framebook = capbook.read()
    retars, framears = capars.read()

    if retbook and retars:
        if count%5 ==0 :
            centerx = framears.shape[1]//2
            x = centerx - cv_cover.shape[1]//2
            a = framears[45:315]
            a = a[:,int(x):int(x+cv_cover.shape[1]),:]
            print(count, count)
            framears = cv2.resize(a, (cv_cover.shape[1], cv_cover.shape[0]))
            matches, locs1, locs2 = matchPics(framebook, cv_cover, opts)
            
            locs1 = locs1[matches[:,0]]
            locs1 = locs1[:,[1,0]]
            locs2 = locs2[matches[:,1]]
            locs2 = locs2[:,[1,0]]
            bestH2to1, inliers = computeH_ransac(locs1, locs2, opts) #tell us how to go from locs2 to locs1
            
            comp_img = compositeH(bestH2to1, framears, framebook)
            dst = cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB)
            
            # font which we will be using to display FPS
            font = cv2.FONT_HERSHEY_SIMPLEX
            # time when we finish processing for this frame
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            
            fps = str(int(fps))
            # putting the FPS count on the frame
            cv2.putText(dst, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
            
            # displaying the frame with fps
            cv2.imshow('frame', dst)

        count = count + 1
        print(count)
        # press 'Q' if you want to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the video capture object and destroy windows
capars.release()
capbook.release()
cv2.destroyAllWindows()

###########################################################