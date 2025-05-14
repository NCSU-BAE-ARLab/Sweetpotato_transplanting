#include <ros/ros.h>
#include <ros/package.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/RobotState.h>
#include <fstream>
#include <iostream>



using namespace std;
using namespace Eigen;


int main(int argc, char** argv)
{
    ros::init(argc, argv, "convert_points");
    ros::AsyncSpinner spinner(1);
    spinner.start();

    std::string path = ros::package::getPath("extrinsic_calibration");
    std::string data_folder_path(path + "/data_files/");

    geometry_msgs::PointStamped kinect_point_wrt_base_link;
    geometry_msgs::PointStamped kinect_point_wrt_wrist2_link;

    tf::TransformListener listener;

    fstream myfile;

    char c;
    float a;


    myfile.open (data_folder_path + "marker_coordinates_wrt_base_link.txt", std::ios::in);
    int num_kinect_points = 0;

    while (myfile.get(c))
    {
        if(c == '\n')
            num_kinect_points = num_kinect_points + 1;
    }

    myfile.close();


    MatrixXf points_wrt_kinect(num_kinect_points, 3);


    myfile.open (data_folder_path + "marker_coordinates_wrt_base_link.txt", std::ios::in);
    int row = 0;
    int column = 0;

    do
    {

        myfile >> a;
        points_wrt_kinect(row, column) = a;
        std::cout << "a " << a << std::endl;

        column = column + 1;
        column = column % 3;
        if(column == 0)
            row = row + 1;

        if(row == num_kinect_points)
            break;

    }while (myfile.get(c));

    myfile.close();



    ofstream myfile2;

    myfile2.open (data_folder_path + "marker_coordinates_wrt_wrist2_link.txt", std::ios::out);




    for (int i=0; i< num_kinect_points; i=i+1)
    {

        kinect_point_wrt_base_link.header.frame_id = "base_link";

        kinect_point_wrt_base_link.point.x = points_wrt_kinect(i, 0);
        kinect_point_wrt_base_link.point.y = points_wrt_kinect(i, 1);
        kinect_point_wrt_base_link.point.z = points_wrt_kinect(i, 2);

        std::cout << "\nClicked_point: " << kinect_point_wrt_base_link.point.x << ", "
                  << kinect_point_wrt_base_link.point.y << ", "
                  << kinect_point_wrt_base_link.point.z << "\n";

        listener.waitForTransform( "wrist_3_link", "base_link", ros::Time(0), ros::Duration(3));

        listener.transformPoint("wrist_3_link", kinect_point_wrt_base_link, kinect_point_wrt_wrist2_link);

        myfile2 << std::setprecision(15) << kinect_point_wrt_wrist2_link.point.x << ", "
                << kinect_point_wrt_wrist2_link.point.y << ", "
                << kinect_point_wrt_wrist2_link.point.z << "\n";

        std::cout << "converted_point: " << kinect_point_wrt_wrist2_link.point.x << ", "
                  << kinect_point_wrt_wrist2_link.point.y << ", "
                  << kinect_point_wrt_wrist2_link.point.z << "\n";


    }

    myfile2.close();

    return 0;
}
