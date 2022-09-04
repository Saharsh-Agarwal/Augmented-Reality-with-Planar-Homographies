import numpy as np
import cv2
from displayMatch import displayMatched
from opts import get_opts
import matplotlib.pyplot as plt

def computeH(x1, x2):
    #Q2.2.1
    #Compute the homography between two sets of points

    r,c = np.shape(x1)
    A = []
    for i in range(r):
        r1 = [-x2[i,0] , -x2[i,1] , -1, 0, 0, 0, x1[i,0]*x2[i,0] , x1[i,0]*x2[i,1] , x1[i,0]]
        r2 = [ 0, 0, 0, -x2[i,0] , -x2[i,1] , -1, x1[i,1]*x2[i,0], x1[i,1]*x2[i,1]  , x1[i,1]]
        A.append(r1)
        A.append(r2)

    A = np.asarray(A)
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    
    # take last row of V.T because s val lowest there.
    ev = vh[-1,:]/vh[-1,-1]  #### Dont forget to scale idiot
    H2to1 = np.reshape(ev,(3,3))
    
    ###    Checking - 
    ###    z = np.asarray([-2,1,1]).reshape(-1,1)
    ###    print(H2to1, H2to1@z)
    
    return H2to1


def computeH_norm(x1, x2):
    #Q2.2.2
    #Compute the centroid of the points
    r,c = np.shape(x1) 
    
    #print(x1,x2)
    
    cx1 = np.mean(x1,axis=0) # 2 values - x_mean and y_mean
    cx2 = np.mean(x2,axis=0)
    
    #Shift the origin of the points to the centroid
    x1_dash = x1-cx1
    x2_dash = x2-cx2
    
    #Normalize the points so that the largest distance from the origin is equal to sqrt(2) 
    dist_x1 = np.max(np.linalg.norm(x1_dash, axis = 1)) #to find max distance
    dist_x2 = np.max(np.linalg.norm(x2_dash, axis = 1))
    
    # Scaling Factor
    sf_x1 = np.sqrt(2)/dist_x1
    sf_x2 = np.sqrt(2)/dist_x2 
    x1_dash = x1_dash*sf_x1
    x2_dash = x2_dash*sf_x2

    #Similarity transform 1
    T1 = np.asarray([[sf_x1, 0, -sf_x1*cx1[0]], [0, sf_x1, -sf_x1*cx1[1]], [0,0,1]])
    x1 = np.hstack((x1,np.ones(r).reshape(-1,1)))

    #Similarity transform 2
    T2 = np.asarray([[sf_x2, 0, -sf_x2*cx2[0]], [0, sf_x2, -sf_x2*cx2[1]], [0,0,1]])
    x2 = np.hstack((x2,np.ones(r).reshape(-1,1)))

    #Compute homography
    H2to1 = computeH(x1_dash,x2_dash)
    H2to1 = np.linalg.inv(T1) @ H2to1 @ T2
    
    ### Checking
    ### print(H2to1)
    ### z = np.asarray([-1,1,1]).reshape(-1,1)
    ### print(H2to1@z) 
    
    return H2to1


def computeH_ransac(locs1, locs2, opts):
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points

    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    inliers = [] 
    in_co = 0
    bestH2to1 = None
    for i in range(max_iters): # max_iters
        idx = np.arange(0,len(locs1))
        np.random.shuffle(idx)
        ptsel = idx[0:4]
        H2to1 = computeH_norm(locs1[ptsel], locs2[ptsel])
        locs2_dash = np.hstack((locs2,np.ones(len(locs2)).reshape(-1,1))) #homogenous form
        locs1_est = (H2to1@locs2_dash.T)[0:2,:] #column vector of estimate found
        dist = np.linalg.norm(locs1.T-locs1_est, axis = 0)
        inlier = (dist<inlier_tol).astype(int)
        inlier_count = np.sum(inlier) #total number of inlier 
        if inlier_count>in_co :
            inliers = inlier
            bestH2to1 = H2to1
            in_co = inlier_count
    
    if len(inliers) > 1 :
        index_good_pts = [i for i,j in enumerate(inliers) if j==1]
        if (len(index_good_pts)>=4): # recalculate for better values as suggested in class
            bestH2to1 = computeH_norm(locs1[index_good_pts], locs2[index_good_pts])
    else:
        bestH2to1 = np.eye(3)
        
    return bestH2to1, inliers



def compositeH(H2to1, template, img):
    
    #Create a composite image after warping the template image on top
    #of the image using the homography

    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.
    

    #Create mask of same size as template
    mask_template = np.ones((template.shape[0], template.shape[1]), dtype=np.uint8)
    mask_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    
    #Warp template by appropriate homography
    composite_img = cv2.warpPerspective(template, H2to1, (img.shape[1], img.shape[0]))
    #plt.imshow(composite_img)
    
    mask_temp_warp = cv2.warpPerspective(mask_template, H2to1, (mask_img.shape[1], mask_img.shape[0]))
    #print(mask_temp_warp.shape) ### 480 640
    #print(mask_temp_warp) ##white box on black (0)
    
    warp_not = cv2.bitwise_not(mask_temp_warp)//255
    warp_not = np.stack([warp_not,warp_not,warp_not])
    warp_not = np.transpose(warp_not, (1,2,0))
    
    background = img*warp_not
    composite_img = background + composite_img
    
    return composite_img



### for checking purposes 
'''
x1 = np.asarray([[1,1],[1,3],[3,3],[3,1]])
x2 = (x1-4) 
opts=get_opts()
m = computeH_norm(x1,x2)
print(m)
'''
'''
x1 = np.asarray([[1,1],[1,3],[3,3],[3,1]])
x2 = np.asarray([[1.366,0.366],[2.366,2.098],[4.098,1.098],[3.098,-0.634]])

m = computeH_norm(x2,x1)
print(m)

print(np.cos(np.pi/6))
'''

"""
if __name__ == "__main__":

    opts = get_opts()
    image1 = cv2.imread('../data/cv_cover.jpg')
    image2 = cv2.imread('../data/cv_desk.png')

    matches, locs1, locs2 = displayMatched(opts, image1, image2)
    locs1 = locs1[matches[:,0]]
    locs2 = locs2[matches[:,1]]
    computeH_ransac(locs1, locs2, opts)
"""