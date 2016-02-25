# draws matches between book and camera capture - updates when you close the image
import cv2
from matplotlib import pyplot as plt
import time 
import math
import subprocess
import os.path
import dill
import numpy as np


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

def kp_match_list(kp_obj, kp_scene, matches_mask, matches):
    obj_list = []
    scene_list = []
    for i, mask in enumerate(matches_mask):
        if 1 in mask:
            scene_list.append(kp_scene[matches[i][0].trainIdx])
            obj_list.append(kp_obj[matches[i][0].queryIdx])

    return obj_list, scene_list

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

def get_dist(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2 )
def find_furthest_kp_sets(kp_list, num_kps=5):
    kp_tup_dists = []
    kps_used = []
    for kp in kp_list:
        for kp2 in kp_list:
            tup = {'kp':kp,'kp2':kp2, 'pix_dist':get_dist(kp.pt[0], kp.pt[1],kp2.pt[0], kp2.pt[1])}
            kp_tup_dists.append(tup)

    dist_sort = sorted(kp_tup_dists, key=lambda tup: tup['pix_dist'], reverse=True)
    return dist_sort[:num_kps]

def add_measured_dist_to_kp_pairs(kp_tup_list, pixels, measured_dist):
    ratio = 1.0*measured_dist/pixels #inch per pixel
    for tupl in kp_tup_list:
        tupl['measured_dist'] =  tupl['pix_dist']*ratio

def store_kps(filename, kp_list):

    tup_list=[]
    for pair in kp_list:
        kp = pair['kp']
        kp2 = pair['kp2']
        
        temp = hashable_kp(kp)
        temp2 = hashable_kp(kp2)
        new_pair = {}
        new_pair['measured_dist'] = pair['measured_dist']
        new_pair['pix_dist'] = pair['pix_dist']
        new_pair['kp'] = temp
        new_pair['kp2'] = temp2
        tup_list.append(new_pair)
        
    with open(filename, 'w') as f:
        dill.dump(tup_list, f)

def load_kps(filename):
    tup_list = dill.load(open(filename))
    print len(tup_list)
    new_list = []
    for i, d in enumerate(tup_list):
        
        kptup=d['kp']
        kp2=d['kp2']

        temp_kp = cv2.KeyPoint(x=kptup[0][0],y=kptup[0][1],_size=kptup[1], _angle=kptup[2], _response=kptup[3], _octave=kptup[4], _class_id=kptup[5])
        temp_kp2 = cv2.KeyPoint(x=kp2[0][0],y=kp2[0][1],_size=kp2[1], _angle=kp2[2], _response=kp2[3], _octave=kp2[4], _class_id=kp2[5])

        d['kp']=temp_kp
        d['kp2']=temp_kp2
        
        new_list.append(d)

    return new_list

def hashable_kp(kp):
    return (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)


KNOWN_DISTANCE = 10.0 #distance measured to initial image

def reject_outliers(data, m=2):
    bins = []
    for thing in data:
        added = False
        for bin in bins:
            if abs(np.mean(bin) - thing) < m:
                added = True
                np.append(bin, [thing])

        if added == False:
            bins.append([thing])

    biggest_bin =[] 
    for bin in bins:
        if len(bin) > len(biggest_bin):
            biggest_bin = bin
            
    
    return biggest_bin
    
def get_obj_dist(object_list, scene_list, furthests, kp_hash):

    dist_guesses = []
    print '=========='
    used_kps= set()
    for pair in furthests:
        kp1 = pair['kp']
        kp2 = pair['kp2']
        pix_dist_scene = pair['pix_dist']

        idx1 = scene_list.index(kp1)
        idx2 =  scene_list.index(kp2)

        m_kps = (hashable_kp(object_list[idx1]),hashable_kp(object_list[idx2]))
        if m_kps in kp_hash and not m_kps[0] in used_kps and not m_kps[1] in used_kps :
            used_kps.add(m_kps[0])
            used_kps.add(m_kps[1])
            orig_kp_info = kp_hash[m_kps]

            orig_pix_dist = orig_kp_info[0]
            orig_measured_dist = orig_kp_info[1]
            F=(orig_pix_dist*KNOWN_DISTANCE)/orig_measured_dist
            dist_guess = (orig_measured_dist*F)/pix_dist_scene
            dist_guesses.append(dist_guess)

    print len(dist_guesses)
    if len(dist_guesses) >0:
        ar = np.array(dist_guesses)
        if np.std(ar)<4:
            #print ar
            ar = reject_outliers( ar, m=2)
            #print ar
            return np.mean(ar)
    
    return None

if __name__ == '__main__':
    kp_save_loc = 'notebook_kps_measurements.pickle'
    img_object = cv2.imread('10inchAway.jpg', 0)
    
    #### 1. #### detect keypoints using surf detector and get descriptors
    min_hessian = 400
    cap = cv2.VideoCapture(0)
    surf = cv2.xfeatures2d.SURF_create(min_hessian)
    kp_obj, des_obj = surf.detectAndCompute(img_object, None)

    kp_pair_measurements = None
    if os.path.exists(kp_save_loc):
        kp_pair_measurements =  load_kps(kp_save_loc)
    else:
        img3 = None
        #img_scene = cv2.cvtColor(img_object, cv2.COLOR_BGR2GRAY)

        img3 = cv2.drawKeypoints(img_object, kp_obj, img3)

        kp_pairs = find_furthest_kp_sets(kp_obj,10000)
        tup = kp_pairs[0]

        kp1 =tup['kp']
        kp2 =tup['kp2']                
        kp_obj = [kp1, kp2]
        img3 = cv2.drawKeypoints(img_object, kp_obj, img3, color=[0,255,0])

        cv2.imshow('frame', img3)
        cv2.imshow('frame', img3)

        dist = raw_input('input distance between the highlighted keypoints: ')
        add_measured_dist_to_kp_pairs(kp_pairs, tup['pix_dist'], float(dist))
        store_kps(kp_save_loc, kp_pairs)
        kp_pair_measurements = kp_pairs

        cap.release()
        cv2.destroyAllWindows()
    assert kp_pair_measurements[0]['measured_dist'] != None

    #setup distance hashmap
    kp_hash = {}
    for ds in kp_pair_measurements:
        print ds
        pair_set = (hashable_kp( ds['kp']), hashable_kp(ds['kp2']))
        kp_hash[pair_set] = (ds['pix_dist'], ds['measured_dist'])

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
            if m.distance < 0.6*n.distance:
                matchesMask[i]=[1,0]

        (x,y)= make_center(kp_scene, matchesMask, matches)

        #get the kps from the object and the scene
        object_list, scene_list = kp_match_list(kp_obj, kp_scene, matchesMask, matches)
        furthests = find_furthest_kp_sets(scene_list,50)

        dist = get_obj_dist(object_list, scene_list, furthests, kp_hash)
        
        #cv2.circle(img_scene,(int(x),int(y)), 10, (0,0,255), -1)

        #cv2.imshow('frame', img_scene)
        #call_adjust(x,y)
        #time.sleep(.01)
        #draw_params = dict(matchColor = (0,0,255),
        #                    singlePointColor = None,
        #                    matchesMask = matchesMask,
        #                    flags = 2)

        #img3 = cv2.drawMatchesKnn(img_object,kp_obj,img_scene,kp_scene,matches,None,**draw_params)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        if dist != None:
         
            print "\t\t\t\t\t\t", dist
        #    cv2.putText(img3,"%.2f" % round(dist,2),(int(x+160),int(y+110)), font, 1,(0,255,0),2,cv2.LINE_AA)
        #cv2.imshow('frame2', img3)
        #thingy = False
        #while not thingy:
        #    if cv2.waitKey(1) & 0xFF == ord('l'):
        #        thingy = True
        #plt.imshow(img3,),plt.show()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
