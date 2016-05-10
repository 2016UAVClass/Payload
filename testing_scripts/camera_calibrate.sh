#! /bin/bash

source /opt/ros/indigo/setup.bash
source ~/catkin_ws/devel/setup.bash

rosrun dynamic_reconfigure dynparam set /mv_26803584 expose_us 72412

rosrun dynamic_reconfigure dynparam set /mv_26803584 b_gain 2.68

rosrun dynamic_reconfigure dynparam set /mv_30000337 agc False

rosrun dynamic_reconfigure dynparam set /mv_30000337 expose_us 10

rosrun dynamic_reconfigure dynparam set /mv_30000337 gain_db 0.0

rosrun dynamic_reconfigure dynparam set /mv_30000337 fps 24
