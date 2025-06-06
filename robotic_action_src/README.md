
# Guides and steps to perform the robotic experiments for this project

<p align="center">
  <img src="https://github.com/NCSU-BAE-ARLab/Sweetpotato_transplanting/blob/main/assets/robot_action.png" width="80%" />
</p>

# Launch the robot ROS drivers
```bash
roslaunch ur_robot_driver ur5e_bringup.launch robot_ip:=192.168.1.10 kinematics_config:=/home/zzh/my_robot_calibration.yaml
```

# Launch the moveit library for robot motion planning
```bash
roslaunch ur5e_moveit_config moveit_planning_execution.launch
```

# Launch the rviz for visualization
```bash
roslaunch ur5e_moveit_config moveit_rviz.launch
```

# Launch camera node (i.e. realsense in our case)
```bash
roslaunch realsense2_camera rs_rgbd.launch align_depth:=true depth_width:=640 depth_height:=480 depth_fps:=30 color_width:=640 color_height:=480 color_fps:=30
```

# Launch the main robot code
First, go through the main_robot.py file and modify the necessory parameters for your specific environment settings.
The code is easy to follow and self-explanatory.
For running grasp planning (ROS and Pytorch together), first, navigate to your source code folder (i.e. robot_action_src)
```bash
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python3 main_robot.py
```
Make sure robot setup is already done (with URCaps)
*********************************************************************************************

