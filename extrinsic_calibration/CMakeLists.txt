cmake_minimum_required(VERSION 2.8.3)
project(extrinsic_calibration)

add_compile_options(-std=c++14)
find_package(catkin REQUIRED COMPONENTS
  moveit_core
  moveit_msgs
  moveit_planners_ompl
  moveit_ros_move_group
  moveit_ros_planning_interface
  moveit_ros_visualization
  message_generation
  std_msgs
  rospy
  roscpp
  geometry_msgs
  sensor_msgs
  actionlib
  actionlib_msgs
  control_msgs
  #visualization_msgs
  pcl_ros
  image_transport
  cv_bridge
)


# Tell CMake to prioritize libraries in /usr/local/lib
#set(OpenCV_DIR /usr/local/lib/cmake/opencv4)

#link_directories(/usr/local/lib)

#find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

# Debugging: Print OpenCV version being used
message(STATUS "OpenCV version: ${OpenCV_VERSION}")



catkin_package(
#  INCLUDE_DIRS include
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

#include_directories(SYSTEM
#                    ${OpenCV_INCLUDE_DIRS}
#                    )




add_executable(store_points_wrt_kinect src/store_points_wrt_kinect.cpp)
target_link_libraries(store_points_wrt_kinect  ${OpenCV_LIBS}  ${catkin_LIBRARIES} ${PCL_LIBRARIES} opencv_core
  opencv_imgproc
  opencv_highgui)
add_dependencies(store_points_wrt_kinect extrinsic_calibration_generate_messages_cpp)



add_executable(store_markers_wrt_base_link src/store_markers_wrt_base_link.cpp)
target_link_libraries(store_markers_wrt_base_link  ${OpenCV_LIBS}   ${catkin_LIBRARIES} ${PCL_LIBRARIES} opencv_core
  opencv_imgproc
  opencv_highgui)
add_dependencies(store_markers_wrt_base_link extrinsic_calibration_generate_messages_cpp)



add_executable(convert_point_from_base_link_to_wrist2_link src/convert_point_from_base_link_to_wrist2_link.cpp)
target_link_libraries(convert_point_from_base_link_to_wrist2_link  ${OpenCV_LIBS}   ${catkin_LIBRARIES} ${PCL_LIBRARIES} opencv_core
  opencv_imgproc
  opencv_highgui)
add_dependencies(convert_point_from_base_link_to_wrist2_link extrinsic_calibration_generate_messages_cpp)



add_executable(convert_point_from_depth_optical_frame_to_camera_link src/convert_point_from_depth_optical_frame_to_camera_link.cpp)
target_link_libraries(convert_point_from_depth_optical_frame_to_camera_link  ${OpenCV_LIBS}   ${catkin_LIBRARIES} ${PCL_LIBRARIES} opencv_core
  opencv_imgproc
  opencv_highgui)
add_dependencies(convert_point_from_depth_optical_frame_to_camera_link extrinsic_calibration_generate_messages_cpp)



add_executable(calculation_of_R_and_T src/calculation_of_R_and_T.cpp)
target_link_libraries(calculation_of_R_and_T  ${OpenCV_LIBS}   ${catkin_LIBRARIES} ${PCL_LIBRARIES} opencv_core
  opencv_imgproc
  opencv_highgui)
add_dependencies(calculation_of_R_and_T extrinsic_calibration_generate_messages_cpp)



add_executable(visualize_marker_from_robot src/visualize_marker_from_robot.cpp)
target_link_libraries(visualize_marker_from_robot  ${OpenCV_LIBS}    ${catkin_LIBRARIES} ${PCL_LIBRARIES} opencv_core
  opencv_imgproc
  opencv_highgui)
add_dependencies(visualize_marker_from_robot extrinsic_calibration_generate_messages_cpp)



add_executable(visualize_marker_from_kinect src/visualize_marker_from_kinect.cpp)
target_link_libraries(visualize_marker_from_kinect  ${OpenCV_LIBS}   ${catkin_LIBRARIES} ${PCL_LIBRARIES} opencv_core
  opencv_imgproc
  opencv_highgui)
add_dependencies(visualize_marker_from_kinect extrinsic_calibration_generate_messages_cpp)



add_executable(test_transformation src/test_transformation.cpp)
target_link_libraries(test_transformation  ${OpenCV_LIBS}   ${catkin_LIBRARIES} ${PCL_LIBRARIES} opencv_core
  opencv_imgproc
  opencv_highgui)
add_dependencies(test_transformation extrinsic_calibration_generate_messages_cpp)



add_executable(error_calculation src/error_calculation.cpp)
target_link_libraries(error_calculation  ${OpenCV_LIBS}    ${catkin_LIBRARIES} ${PCL_LIBRARIES} opencv_core
  opencv_imgproc
  opencv_highgui)
add_dependencies(error_calculation extrinsic_calibration_generate_messages_cpp)



add_executable(save_cloud_and_image src/save_cloud_and_image.cpp)
target_link_libraries(save_cloud_and_image  ${OpenCV_LIBS}  ${catkin_LIBRARIES} ${PCL_LIBRARIES} opencv_core
  opencv_imgproc
  opencv_highgui)
add_dependencies(save_cloud_and_image extrinsic_calibration_generate_messages_cpp)



