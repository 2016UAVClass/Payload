import cv2
import numpy as np

cap = cv2.VideoCapture(0)

blob_detect_config = cv2.SimpleBlobDetector_Params()
# Change thresholds
#blob_detect_config.minThreshold = 10
#blob_detect_config.maxThreshold = 200


# Filter by Area.
blob_detect_config.filterByArea = True
blob_detect_config.minArea = 200
blob_detect_config.maxArea = 1000000

# Filter by Circularity
blob_detect_config.filterByCircularity = True
blob_detect_config.minCircularity = 0.2

# Filter by Convexity
blob_detect_config.filterByConvexity = False
blob_detect_config.minConvexity = 0.87

# Filter by Inertia
blob_detect_config.filterByInertia = False
blob_detect_config.minInertiaRatio = 0.01

blob_detect = cv2.SimpleBlobDetector_create(blob_detect_config)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    frame_cr = cv2.extractChannel(frame_ycrcb,1)

    _,frame_cr_highlights = cv2.threshold(frame_cr,150,255,cv2.THRESH_BINARY_INV)
    # Threshold the HSV image to get only blue colors
    # mask = cv2.inRange(hsv, lower_green, upper_green)

    keypoints = blob_detect.detect(frame_cr_highlights)

    im_with_keypoints = cv2.drawKeypoints(frame_cr_highlights, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    cv2.imshow('frame',im_with_keypoints)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()