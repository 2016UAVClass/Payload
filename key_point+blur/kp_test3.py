# draws matches between book and camera capture - updates when you close the image
import cv2
from matplotlib import pyplot as plt
import time 

import subprocess


img_object = cv2.imread('notebook-11inchesaway.jpg', 0)

#### 1. #### detect keypoints using surf detector and get descriptors
min_hessian = 400
cap = cv2.VideoCapture(0)
surf = cv2.xfeatures2d.SURF_create(min_hessian)
kp_obj, des_obj = surf.detectAndCompute(img_object, None)

def make_center(kp_list, matches_mask, matches):
    x=0
    y=0

    total=0
    for i, mask in enumerate(matches_mask):
        if 1 in mask:

            kp = kp_list[matches[i][0].trainIdx]

            x+=kp.pt[0]
            y+=kp.pt[1]
            
            total+=1
    if total==0:
        total = 1
    return (1.0*x/total, 1.0*y/total)

def call_adjust(x,y):
    difx=x-300
    dify=y-300

    speedwpm = 600
    centered_thresh=40
    if abs(difx) < centered_thresh and abs(dify) < centered_thresh:
        subprocess.call(["say", "down", '-r', str(speedwpm)])
    else:
        if abs(difx)>abs(dify):
            if difx<0:
                subprocess.call(["say", "left", '-r', str(speedwpm)])
            else:
                subprocess.call(["say", "right", '-r', str(speedwpm)])
        else:
            if dify<0:
                subprocess.call(["say", "forward", '-r', str(speedwpm)])
            else:
                subprocess.call(["say", "back", '-r', str(speedwpm)])


while(True):

    ret, frame = cap.read()
    
    img_scene = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
    good_matches = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
            
    (x,y)= make_center(kp_scene, matchesMask, matches)

    cv2.circle(img_scene,(int(x),int(y)), 4, (0,0,255), -1)


    #cv2.imshow('frame', img_scene)
    call_adjust(x,y)
    #time.sleep(.01)
    draw_params = dict(matchColor = (0,0,255),
                        singlePointColor = None,
                        matchesMask = matchesMask,
                        flags = 2)

    img3 = cv2.drawMatchesKnn(img_object,kp_obj,img_scene,kp_scene,matches,None,**draw_params)

    plt.imshow(img3,),plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
