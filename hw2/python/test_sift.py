import cv2
from opts import get_opts
import matplotlib.pyplot as plt
from displayMatch import displayMatched

#Q2.1.7 - extra

def siftTest(opts):

    #Read the image and convert to grayscale, if necessary
    img = cv2.imread('../data/hp_cover.jpg')
    test_img = cv2.imread('../data/hp_cover.jpg')
    test_img = cv2.resize(test_img, (img.shape[1]//2,img.shape[0]//2))
    
    if (len(img.shape)>=3):
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if (len(test_img.shape)>=3):
        test_img1 = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1,None)
    kp2, desc2 = sift.detectAndCompute(test_img1, None)
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1,desc2,k=2)
    # Apply ratio test
    pts = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            pts.append([m])
            
    img3 = cv2.drawMatchesKnn(img,kp1,test_img,kp2,pts,None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()
    
    displayMatched(opts, img, test_img)


if __name__ == "__main__":

    opts = get_opts()
    siftTest(opts)


