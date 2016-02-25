import cv2
import numpy as np

cap = cv2.VideoCapture(0)

blob_detect_config = cv2.SimpleBlobDetector_Params()
blob_detect = cv2.SimpleBlobDetector(blob_detect_config)
def regenerate_blob_detector(x):
    # Filter by Area.
    print("Called")

    blob_detect_config.filterByArea = True
    blob_detect_config.minArea = cv2.getTrackbarPos('Area_Min', 'image')
    blob_detect_config.maxArea = cv2.getTrackbarPos('Area_Max', 'image')

    # Filter by Circularity
    blob_detect_config.filterByCircularity = False
    blob_detect_config.minCircularity = 0.8

    # Filter by Convexity
    blob_detect_config.filterByConvexity = False
    blob_detect_config.minConvexity = 0.87

    # Filter by Inertia
    blob_detect_config.filterByInertia = False
    blob_detect_config.minInertiaRatio = 0.01

    print('Finished')


cv2.namedWindow('frame')
cv2.createTrackbar('Area_Max', 'frame', 0, 1000000, regenerate_blob_detector)
cv2.createTrackbar('Area_Min', 'frame', 0, 5000, regenerate_blob_detector)
regenerate_blob_detector(None)

while (1):
    global blob_detect
    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    frame_cr = cv2.extractChannel(frame_ycrcb, 1)

    _, frame_cr_highlights = cv2.threshold(frame_cr, 135, 255, cv2.THRESH_BINARY_INV)
    # Threshold the HSV image to get only blue colors
    # mask = cv2.inRange(hsv, lower_green, upper_green)

    keypoints = blob_detect.detect(frame_cr_highlights)

    print("Found " + str(len(keypoints)) + " keypoints")

    im_with_keypoints = cv2.drawKeypoints(frame_cr_highlights, keypoints, np.array([]), (0, 255, 0),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('frame', im_with_keypoints)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
