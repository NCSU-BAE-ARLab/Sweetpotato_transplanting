#include <ros/package.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <sensor_msgs/JointState.h>
#include <fstream>
#include <iostream>


#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <pcl_ros/transforms.h>

#include <signal.h>
#include <unistd.h>
#include <eigen3/Eigen/Eigen>
#include <pcl/registration/transformation_estimation_svd.h>



using namespace std;
using namespace Eigen;

int main(int argc, char **argv)
{
    fstream myfile;
    std::string path = ros::package::getPath("extrinsic_calibration");
    std::string data_folder_path(path + "/data_files/");

    char c;
    float a;


    myfile.open (data_folder_path + "kinect_points_wrt_camera_link.txt", std::ios::in);
    int num_kinect_points = 0;

    while (myfile.get(c))
    {
        if(c == '\n')
            num_kinect_points = num_kinect_points + 1;
    }

    myfile.close();


    MatrixXf points_wrt_kinect(num_kinect_points, 3);


    myfile.open (data_folder_path + "kinect_points_wrt_camera_link.txt", std::ios::in);
    int row = 0;
    int column = 0;

    do
    {

        myfile >> a;
        points_wrt_kinect(row, column) = a;

        column = column + 1;
        column = column % 3;
        if(column == 0)
            row = row + 1;

        if(row == num_kinect_points)
            break;

    }while (myfile.get(c));

    myfile.close();




    myfile.open (data_folder_path + "marker_coordinates_wrt_wrist2_link.txt", std::ios::in);
    int num_base_link_points = 0;

    while (myfile.get(c))
    {
        if(c == '\n')
            num_base_link_points = num_base_link_points + 1;
    }

    myfile.close();


    MatrixXf points_wrt_base_link(num_base_link_points, 3);


    myfile.open (data_folder_path + "marker_coordinates_wrt_wrist2_link.txt", std::ios::in);
    row = 0;
    column = 0;

    do
    {

        myfile >> a;
        points_wrt_base_link(row, column) = a;

        column = column + 1;
        column = column % 3;
        if(column == 0)
            row = row + 1;

        if(row == num_base_link_points)
            break;

    }while (myfile.get(c));

    myfile.close();


    pcl::PointCloud<pcl::PointXYZ> cloud_wrist;
    pcl::PointCloud<pcl::PointXYZ> cloud_kinect;

    for(int i=0;i<points_wrt_kinect.rows();i++)
    {
        pcl::PointXYZ kinect_point;
        kinect_point.x = points_wrt_kinect(i,0);
        kinect_point.y = points_wrt_kinect(i,1);
        kinect_point.z = points_wrt_kinect(i,2);

        pcl::PointXYZ wrist_point;
        wrist_point.x = points_wrt_base_link(i,0);
        wrist_point.y = points_wrt_base_link(i,1);
        wrist_point.z = points_wrt_base_link(i,2);

        cloud_wrist.push_back(wrist_point);
        cloud_kinect.push_back(kinect_point);
    }


    Eigen::Matrix4f pcl_tx;
    pcl::registration::TransformationEstimationSVD<pcl::PointXYZ,pcl::PointXYZ> estimate_transformation;

    estimate_transformation.estimateRigidTransformation(cloud_kinect,cloud_wrist,pcl_tx);


    Eigen::Affine3f pose_mat;
    pose_mat.matrix() = pcl_tx;
    Eigen::Quaternion< float, Eigen::AutoAlign > q(pose_mat.rotation());
    Vector3f euler = q.toRotationMatrix().eulerAngles(2, 1, 0);

//    std::cout << "r = " << euler[2]
//              << ", p = " << euler[1]
//              << ", y = " << euler[0] << std::endl;

    std::cout << "rpy " << euler[2] << " " << euler[1] << " " << euler[0] << std::endl;
    std::cout << "xyz " << pcl_tx(0, 3) << " " << pcl_tx(1, 3) << " " << pcl_tx(2, 3) << std::endl;






    myfile.open (data_folder_path + "final_R_and_T.txt", std::ios::out);
    myfile << pcl_tx ;
    myfile.close();




    return 0;
}
