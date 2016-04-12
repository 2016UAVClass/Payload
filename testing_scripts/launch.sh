#! /bin/bash
source /opt/ros/indigo/setup.bash
source ~/catkin_ws/devel/setup.bash

roscore &
sleep 2
./launch_bw.sh || echo "SOMETHING DIDNT WORK"
sleep 6
./launch_color.sh || echo "SOMETHING DIDNT WORK"
sleep 6

rosbag record -a -o test_pref &


