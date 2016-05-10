import numpy as np
import cv2
import time 
cap = cv2.VideoCapture(0)



print "Capture 1"
time.sleep(1)
# Capture frame-by-frame
ret0, frame0 = cap.read()
time.sleep(1)
print "Capture 2"
time.sleep(1)
# Capture frame-by-frame
ret1, frame1 = cap.read()
time.sleep(1)
print "Capture 3"
time.sleep(1)
# Capture frame-by-frame
ret2, frame2 = cap.read()
time.sleep(1)
    
# Our operations on the frame come here
gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Display the resulting frame
cv2.imshow('frame0',gray0)
cv2.imshow('frame1',gray1)
cv2.imshow('frame2',gray2)

print "blur0", cv2.Laplacian(gray0, cv2.CV_64F).var()
print "blur1", cv2.Laplacian(gray1, cv2.CV_64F).var()
print "blur2", cv2.Laplacian(gray2, cv2.CV_64F).var()
# When everything done, release the capture
while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
cap.release()
cv2.destroyAllWindows()

