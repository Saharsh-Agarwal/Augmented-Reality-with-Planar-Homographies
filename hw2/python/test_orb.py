import cv2
from opts import get_opts
import matplotlib.pyplot as plt
from displayMatch import displayMatched
import scipy

#Q2.1.7 - extra

def siftTest(opts):

    #Read the image and convert to grayscale, if necessary
    img = cv2.imread('../data/cv_cover.jpg')
    test_img = cv2.imread('../data/cv_cover.jpg')
    test_img = scipy.ndimage.rotate(test_img, angle=110)
    
    if (len(img.shape)>=3):
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if (len(test_img.shape)>=3):
        test_img1 = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create()
    kp1, desc1 = orb.detectAndCompute(img1,None)
    kp2, desc2 = orb.detectAndCompute(test_img1, None)
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    
    matches = bf.match(desc1,desc2)            
    img3 = cv2.drawMatches(img,kp1,test_img,kp2,matches,None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()
    
    displayMatched(opts, img, test_img)


if __name__ == "__main__":

    opts = get_opts()
    siftTest(opts)


