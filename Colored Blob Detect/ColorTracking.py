import cv2
import math
import PID
import numpy as np

cap = cv2.VideoCapture(0)

blob_detect_config = cv2.SimpleBlobDetector_Params()
blob_detect_config.filterByArea = True
blob_detect_config.minArea = 200
blob_detect_config.maxArea = 1000000

# Filter by Circularity
blob_detect_config.filterByCircularity = True
blob_detect_config.minCircularity = 0.3

# Filter by Convexity
blob_detect_config.filterByConvexity = False
blob_detect_config.minConvexity = 0.87

# Filter by Inertia
blob_detect_config.filterByInertia = False
blob_detect_config.minInertiaRatio = 0.01

blob_detect = cv2.SimpleBlobDetector_create(blob_detect_config)

controller_out = PID.PID()
controller_out.SetKp(1)
controller_out.SetKi(1)
controller_out.SetKd(1)

def nothing(x):
    print(x)
    return

cv2.namedWindow('frame')
cv2.createTrackbar('Thresh_Min', 'frame', 0, 255, nothing)

while (1):
    # Take each frame
    _, frame = cap.read()
    height, width, _ = frame.shape
    # Convert BGR to HSV
    frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    frame_cr = cv2.extractChannel(frame_ycrcb, 1)

    _, frame_cr_highlights = cv2.threshold(frame_cr,cv2.getTrackbarPos('Thresh_Min', 'frame'),255, cv2.THRESH_BINARY)
    #frame_cr_highlights = cv2.adaptiveThreshold(frame_cr,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    # Threshold the HSV image to get only blue colors
    # mask = cv2.inRange(hsv, lower_green, upper_green)

    keypoints = blob_detect.detect(frame_cr_highlights)

    im_with_keypoints = cv2.drawKeypoints(frame_cr_highlights, keypoints, np.array([]), (0, 255, 0),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if(len(keypoints) > 0):
        center = (width/2, height/2)
        target = (int(keypoints[0].pt[0]), int(keypoints[0].pt[1]))
        distance = math.sqrt((center[0] - target[0])**2 + (center[1] - target[1])**2)

        out_value = controller_out.GenOut(distance)
        controller_out.SetPrevErr(distance)
        print(out_value)

        cv2.line(im_with_keypoints, center, target, (0, 255-distance, distance), 5)

    cv2.imshow('frame', im_with_keypoints)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
