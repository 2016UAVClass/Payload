#! /bin/bash
source /opt/ros/indigo/setup.bash
source ~/catkin_ws/devel/setup.bash

roscore &
sleep 2
./launch_bw.sh || echo "SOMETHING DIDNT WORK"
sleep 6
echo "started black and white"

./launch_color.sh || echo "SOMETHING DIDNT WORK"
sleep 6
echo "started color"

./camera_calibrate.sh
echo "calibrated"

./rosbag_start.sh
echo "started rosbag"


