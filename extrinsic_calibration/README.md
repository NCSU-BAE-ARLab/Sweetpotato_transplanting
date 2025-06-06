### Install
Just copy this entire folder in the src folder of your ROS workspace and recompile it.

### Preparation
In the robot workspace area, put nine markers of your choice (point markers) on a flat surface.
Switch on the robot and start all basic nodes (robot, moveit, rviz, camera etc.). Please refer to robot_action_src folder of this repository.

### Step 1: store the markers in robot base frame
```bash
rosrun extrinsic_calibration store_markers_wrt_base_link
```

### Step 2 store the markers in camera frame
Please check the camera ros topics in the code file and match with the actual ros topics published by your camera node.
```bash
rosrun extrinsic_calibration store_points_wrt_kinect
```

### Step 3 Run following commands one by one to calculate the calibration parameters.
```bash
rosrun extrinsic_calibration convert_point_from_base_link_to_wrist2_link
rosrun extrinsic_calibration convert_point_from_depth_optical_frame_to_camera_link
rosrun extrinsic_calibration calculation_of_R_and_T
```

Note: UPDATE URDF FILE as per the updated calibration parameters

### For validation, you can use following commands:
```bash
rosrun extrinsic_calibration error_calculation
rosrun extrinsic_calibration visualize_marker_from_kinect
rosrun extrinsic_calibration visualize_marker_from_robot
rosrun extrinsic_calibration test_transformation
```
