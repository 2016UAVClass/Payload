# draws matches between book and camera capture - updates when you close the image
import cv2
from matplotlib import pyplot as plt

img_object = cv2.imread('grey_book.jpg', 0)


#### 1. #### detect keypoints using surf detector and get descriptors
min_hessian = 400
cap = cv2.VideoCapture(0)
while(True):

    ret, frame = cap.read()
    
    img_scene = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create(min_hessian)

    kp_obj, des_obj = surf.detectAndCompute(img_object, None)
    kp_scene, des_scene = surf.detectAndCompute(img_scene, None)


    #### match ###3

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_obj,des_scene, k=2)

    #flann = cv2.FlannBasedMatcher(index_params,search_params)
    #matches = flann.knnMatch(des_obj,des_scene,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in xrange(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,0,255),
                       singlePointColor = None,
                       matchesMask = matchesMask,
                       flags = 2)

    img3 = cv2.drawMatchesKnn(img_object,kp_obj,img_scene,kp_scene,matches,None,**draw_params)

    plt.imshow(img3,),plt.show()


