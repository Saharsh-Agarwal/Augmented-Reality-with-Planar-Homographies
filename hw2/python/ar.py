import numpy as np
import cv2
import matplotlib.pyplot as plt

#Import necessary functions
from matchPics import matchPics
from helper import loadVid
from opts import get_opts
from planarH import computeH_ransac
from planarH import compositeH
#from helper import plotMatches

opts = get_opts()

#Write script for Q3.1

book = loadVid("../data/book.mov")
shape_book = np.shape(book)

cv_cover = cv2.imread('../data/cv_cover.jpg')
shape_cover = cv_cover.shape

ars = loadVid("../data/ar_source.mov")
ars = ars[:,45:315,:,:] ## Removes the black strips from top and bottom
shape_ars = np.shape(ars)
### the range of 40 to 320 is found by the for loop below
###for i in range(shape_ars[1]):
    ###print(i, ars[1,i,100,1])
    
    # 0.8 15 1000 2
centerx = shape_ars[2]//2
x = centerx - shape_cover[1]/2

ars = ars[:,:, int(x):int(x+shape_cover[1]),:] ### central region extracted from ars
shape_ars = np.shape(ars)

f = min(shape_ars[0],shape_book[0])
frame_size = (shape_book[2],shape_book[1])

out = cv2.VideoWriter('../result/ar.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, frame_size)

for i in range(f):
    print(i)
    ars_frame = cv2.resize(ars[i,:,:,:], (cv_cover.shape[1], cv_cover.shape[0]))
    ars_frame_gray = cv2.cvtColor(ars_frame,cv2.COLOR_BGR2GRAY)
    book_frame = book[i,:,:,:]
    book_frame_gray = cv2.cvtColor(book_frame,cv2.COLOR_BGR2GRAY)
    cv_cover_gray = cv2.cvtColor(cv_cover,cv2.COLOR_BGR2GRAY)

    matches, locs1, locs2 = matchPics(book_frame, cv_cover, opts)
    #plotMatches(cv_desk, cv_cover, matches, locs1, locs2) ### already prints only true matches 
    
    locs1 = locs1[matches[:,0]]
    locs1 = locs1[:,[1,0]]
    locs2 = locs2[matches[:,1]]
    locs2 = locs2[:,[1,0]]
    bestH2to1, inliers = computeH_ransac(locs1, locs2, opts) #tell us how to go from locs2 to locs1
    
    comp_img = compositeH(bestH2to1, ars_frame, book_frame)
    
    dst = cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB)
    #plt.imshow(dst)
    
    #path = '../images/'+str(i)+'.png'
    #plt.imsave(path, dst)
    out.write(dst[:,:,::-1])

out.release()






