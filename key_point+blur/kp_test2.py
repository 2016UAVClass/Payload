# draws matches between multiple book images and camera capture - updates when you close the image
import cv2
from matplotlib import pyplot as plt

img_object = cv2.imread('grey_book.jpg', 0)
obj_im_files = ['book_far2.jpg','book_far_blurry.jpg','book_skew2.jpg','book_skew1.jpg','grey_book.jpg']
min_hessian = 400
surf = cv2.xfeatures2d.SURF_create(min_hessian)
obj_desc = []
for book in obj_im_files:
    im =cv2.imread(book, 0)
    kp_obj, des_obj = surf.detectAndCompute(im, None)
    obj_desc.append(des_obj)
#### 1. #### detect keypoints using surf detector and get descriptors

cap = cv2.VideoCapture(0)
while(True):

    ret, frame = cap.read()
    
    img_scene = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    max_image = ""
    highest_conf = 0
    kp_scene, des_scene = surf.detectAndCompute(img_scene, None)
    count_list = [0]*len(obj_desc)
    for j, des in enumerate( obj_desc):

        #### match ###3

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des,des_scene, k=2)

        #flann = cv2.FlannBasedMatcher(index_params,search_params)
        #matches = flann.knnMatch(des_obj,des_scene,k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in xrange(len(matches))]

        # ratio test as per Lowe's paper
        match_count = 0
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
                match_count+=1

        if match_count > highest_conf:
            highest_conf = match_count
            max_image = obj_im_files[j]
        count_list[j] = match_count

    for num, name in enumerate(obj_im_files):
        print "|"+name+"\t:" + str(count_list[num])+"\t",
        # draw_params = dict(matchColor = (0,0,255),
        #                    singlePointColor = None,
        #                    matchesMask = matchesMask,
        #                    flags = 2)

        # img3 = cv2.drawMatchesKnn(img_object,kp_obj,img_scene,kp_scene,matches,None,**draw_params)

        # plt.imshow(img3,),plt.show()


