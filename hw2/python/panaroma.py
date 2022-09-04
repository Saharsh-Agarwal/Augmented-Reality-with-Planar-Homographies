import numpy as np
import cv2
import matplotlib.pyplot as plt
from planarH import computeH_ransac
from planarH import compositeH
from opts import get_opts
from matchPics import matchPics
from helper import plotMatches

# Import necessary functions

# Q4
opts = get_opts()

pl = cv2.imread('../data/pano_left_cmu.jpeg')
pr = cv2.imread('../data/pano_right_cmu.jpeg')

matches, locs1, locs2 = matchPics(pl, pr, opts)  
locs1 = locs1[matches[:,0]]
locs1 = locs1[:,[1,0]]
locs2 = locs2[matches[:,1]]
locs2 = locs2[:,[1,0]]

bestH2to1, inliers = computeH_ransac(locs1, locs2, opts) ## right se left kaise jaye

r,c = int(2*pl.shape[1]),int(1.5*pl.shape[0])

dst1 = cv2.warpPerspective(pr, bestH2to1, (r,c))
dst1[0:pl.shape[0],0:pl.shape[1]]=pl
cv2.imwrite('../data/output_cmu.jpg',dst1)
plt.imshow(dst1[:,:,::-1])

