import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
import scipy
import matplotlib.pyplot as plt
from displayMatch import displayMatched

#Q2.1.6

def rotTest(opts):

    #Read the image and convert to grayscale, if necessary
    img = cv2.imread('../data/cv_cover.jpg')
    test_img = cv2.imread('../data/cv_cover.jpg')
    
    #if (len(img.shape)>=3):
    #    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #if (len(test_img.shape)>=3):
    #    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
       
    count=[]
    bins = np.arange(36)
    
    for i in range(36):
        print(i)
        rot_img = scipy.ndimage.rotate(test_img, angle=i*10)
        matches,_,_ = matchPics(img, rot_img, opts)
        if (i == 1 or i == 10 or i == 20):
            #displaying 2 images 
            displayMatched(opts, img, rot_img)
        count.append(len(matches))
    
    #Display histogram
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.bar(bins, count)
    ax.set_xticks(bins)
    ax.set_xticklabels(bins)
    plt.xlabel('Rotation Angle')
    plt.ylabel('Number of Matches')
    plt.show()


if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)


