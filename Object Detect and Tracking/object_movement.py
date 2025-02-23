# USAGE
# Tracks movement of red objects
# writes out direction of moving object and offset dX dY from the center of the frame
# python object_movement.py --video object_tracking_example.mp4
# python object_movement.py -b >10
# python object_movement.py

# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

# argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
            help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
            help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "red"
# lower mask : 0-10
lower_red0 = np.array([0,50,50])
upper_red0 = np.array([10,255,255])


# upper mask : 170-180
lower_red1 = np.array([170,50,50])
upper_red1 = np.array([180,255,255])


# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=args["buffer"])
counter = 0
(dX, dY) = (0, 0)
direction = ""

# if video path was not supplied, grab the reference
# to the camera
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# otherwise grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])

# loop
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    
    # if viewing a video and did not grab next frame,
    # then reached the end of the video
    if args.get("video") and not grabbed:
        break

    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "red"
    mask0 = cv2.inRange(hsv, lower_red0, upper_red0)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    # join the masks
    mask = mask0+mask1

    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(frame,frame, mask= mask)

    # find contours in the mask and initialize the current
    # (x, y) center
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
        
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                            
        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
        # update list of tracked points
        pts.appendleft(center)

    # loop over the set of tracked points
    for i in np.arange(1, len(pts)):
        # if either of the tracked points are None, ignore
        if pts[i - 1] is None or pts[i] is None:
            continue
        
        # check to see if enough points have been accumulated in
        # the buffer
        if counter >= 10 and i == 1 and pts[-10] is not None:
            # compute the difference between the x and y
            # coordinates and re-initialize the direction
            # text variables
            dX = pts[-10][0] - pts[i][0]
            dY = pts[-10][1] - pts[i][1]
            (dirX, dirY) = ("", "")
                
            # ensure there is significant movement in the
            # x-direction
            if np.abs(dX) > 20:
                dirX = "East" if np.sign(dX) == 1 else "West"
                    
            # ensure there is significant movement in the
            # y-direction
            if np.abs(dY) > 20:
                dirY = "North" if np.sign(dY) == 1 else "South"
            
            # handle when both directions are non-empty
            if dirX != "" and dirY != "":
                direction = "{}-{}".format(dirY, dirX)
                    
            # otherwise, only one direction is non-empty
            else:
                direction = dirX if dirX != "" else dirY
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        #thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        #cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # show the movement deltas and the direction of movement on
    # the frame
    cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 0, 255), 3)
    cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 0, 255), 1)

    # show the frame to our screen and increment the frame counter
    cv2.imshow("Frame", frame)
    cv2.imshow('mask',mask)
    #cv2.imshow('res',res)
    key = cv2.waitKey(1) & 0xFF
    counter += 1
    
    # if the esc key is pressed, stop the loop
    if key == 27:
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()