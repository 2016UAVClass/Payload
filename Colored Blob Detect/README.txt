Ask Richard (Max) if you have any questions about how to run this, since it just got way more complicated.

The program expects a raw_image published to the ROS topic named on line 27, currently set to /usb_cam/image_raw

To run with a webcam:
Set up a catkin workspace. Add the following dependencies
	cv_bridge
	usb_cam

Create a new ros package with 
	catkin_create_pkg color_tracking usb_cam cv_bridge

Add the python files to the src directory of the new package
Add the following lines to CMakeLists.txt file in your color_tracking package folder
	find_package(OpenCV)
	include_directories(${OpenCV_INCLUDE_DIRS})
Use catkin_make to build the packages and dependencies. If it fails, make sure you ran the workspace setup shell file

Start roscore
Launch the usb driver with the usb_cam-test.launch file and roslaunch
Run the python script ColorTracking.py


