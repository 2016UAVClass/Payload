kp_dist_test_full.py is the file you want

if you are using a new object to be tracked crop it and name it trap_crop.png.
The first time you run this that image will be displayed with 2 keypoints highlighted.
Measure the distance between those points and enter it into the shell.

the distance from the camera to the object is currently hard coded in the line 120:
KNOWN_DISTANCE = 10.0 #distance measured to initial image

Those values will be used to assign distances between all the keypoints - the code isnt super efficient

it will then begin streaming video looking for the object displaying the distance on the image when it finds it.