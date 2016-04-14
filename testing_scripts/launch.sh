#! /bin/bash
source /opt/ros/indigo/setup.bash
source ~/catkin_ws/devel/setup.bash

roscore &
sleep 2
~/testing_scripts/launch_bw.sh || echo "SOMETHING DIDNT WORK"
sleep 6
echo "started black and white"

~/testing_scripts/launch_color.sh || echo "SOMETHING DIDNT WORK"
sleep 6
echo "started color"


rosrun dynamic_reconfigure dynparam set /mv_26803584 expose_us 72412

rosrun dynamic_reconfigure dynparam set /mv_26803584 b_gain 2.68

rosrun dynamic_reconfigure dynparam set /mv_30000337 agc False

rosrun dynamic_reconfigure dynparam set /mv_30000337 expose_us 10

rosrun dynamic_reconfigure dynparam set /mv_30000337 gain_db 0.0

rosrun dynamic_reconfigure dynparam set /mv_30000337 fps 24
#~/testing_scripts/camera_calibrate.sh
echo "calibrated"

~/testing_scripts/rosbag_start.sh
echo "started rosbag"


