import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH
import matplotlib.pyplot as plt
from helper import plotMatches

# Import necessary functions

# Q2.2.4

def warpImage(opts):
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    cv_desk = cv2.imread('../data/cv_desk.png')
    hp_cover = cv2.imread('../data/hp_cover.jpg')
    ### print(cv_cover.shape,cv_desk.shape,hp_cover.shape)
    
    matches, locs1, locs2 = matchPics(cv_desk, cv_cover, opts)  
    #plotMatches(cv_desk, cv_cover, matches, locs1, locs2) ### already prints only true matches 
    
    locs1 = locs1[matches[:,0]]
    locs1 = locs1[:,[1,0]]
    locs2 = locs2[matches[:,1]]
    locs2 = locs2[:,[1,0]]
    bestH2to1, inliers = computeH_ransac(locs1, locs2, opts) #tell us how to go from locs2 to locs1
    ### H2to1 goes from cover to desk  #plt.imshow(image2)
    print(hp_cover.shape,cv_cover.shape)
    hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))
    comp_img = compositeH(bestH2to1, hp_cover, cv_desk)
    comp_img = cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB)
    
    plt.imshow(comp_img)
    


if __name__ == "__main__":

    opts = get_opts()
    warpImage(opts)


