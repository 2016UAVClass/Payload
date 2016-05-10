kp_test_dist_full.py finds keypoints in an image, asks you to measure the distance between
two points on the given image, in the kp_test_dist_full.py you will also need to specify the
distance from the camera to the object. it will then calculate the distance to the object if
it is confident it is visible. A version of this is written as a ros node fiducial_tracker
or something like that. I didn't save it in git, oops.

if you are using a new object to be tracked crop it and name it trap_crop.png .
The first time you run this that image will be displayed with 2 keypoints highlighted.
Measure the distance between those points and enter it into the shell.

the distance from the camera to the object is currently hard coded in the line 120:
KNOWN_DISTANCE = 10.0 #distance measured to initial image

Those values will be used to assign distances between all the keypoints - the code isnt super efficient

it will then begin streaming video looking for the object displaying the distance on the image when it finds it.