import numpy as np
import cv2
import skimage.color
import helper
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

# Q2.1.4

def matchPics(I1, I2, opts):
        """
        Match features across images

        Input
        -----
        I1, I2: Source images
        opts: Command line args

        Returns
        -------
        matches: List of indices of matched features across I1, I2 [p x 2]
        locs1, locs2: Pixel coordinates of matches [N x 2]
        """
        
        ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
        sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
        
        # TODO: Convert Images to GrayScale
        real_image1 = I1 # just to keep a copy
        real_image2 = I2
        if (len(I1.shape)>=3):
            I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
        if (len(I2.shape)>=3):
            I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
        
        # TODO: Detect Features in Both Images
        locs1 = corner_detection(I1,sigma)
        locs2 = corner_detection(I2,sigma)
        
        # TODO: Obtain descriptors for the computed feature locations
        desc1, locs1 = computeBrief(I1,locs1)
        desc2, locs2 = computeBrief(I2,locs2)
        # TODO: Match features using the descriptors
        matches = briefMatch(desc1,desc2,ratio)
        #helper.plotMatches(I1,I2,matches,locs1,locs2) 
        
        return matches, locs1, locs2


