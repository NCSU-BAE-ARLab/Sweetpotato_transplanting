#include <ros/ros.h>
#include <ros/package.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include<visualization_msgs/Marker.h>

#include <fstream>
#include <iostream>



cv::Mat img;
int k = 0;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_input(new pcl::PointCloud<pcl::PointXYZRGB>);

char a;


int *centroid_row_pntr;
int *centroid_col_pntr;

bool flag_img = false;
bool flag_cloud = false;
bool flag_click = false;

int number_of_points = 1;
int number_of_points_selected = 0;



void send_data(std::ostream & o, const std::vector<uchar> & v)
{
    o.write(reinterpret_cast<const char*>(v.data()), v.size());
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    if(!flag_img)
    {
        std::cout << "\nim in imageCallback";
        img = cv_bridge::toCvShare(msg, "bgr8")->image;

        std::vector<uchar> buff;
        cv::imencode(".PNG", img, buff);
        std::string path = ros::package::getPath("extrinsic_calibration");
        std::ofstream outfile (path + "/img.png");
        send_data(outfile, buff);
        outfile.close();

        k = 1;
        flag_img = 1;
        usleep(10000);
    }

}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{

    std::cout << "In call back with flag_click " << flag_click << std::endl;
    if(!flag_click)
    {


        if  ( event == cv::EVENT_LBUTTONDOWN )
        {
            std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;

            *(centroid_row_pntr + number_of_points_selected) = y;
            *(centroid_col_pntr + number_of_points_selected) = x;

            if ( std::isnan(pcl_input->at(x, y).x))
            {
                std::cout << "Nan detected " << std::endl;

            }
            else
            {
                number_of_points_selected = number_of_points_selected + 1;

                if (number_of_points_selected == number_of_points)
                    flag_click = 1;
            }
        }

    }




}

void centroid_region_cb (const sensor_msgs::PointCloud2ConstPtr& input)
{

    if(!flag_cloud)
    {
        std::cout << "\nim in centroid_region_cb";
        pcl::fromROSMsg(*input, *pcl_input);
        k = 2;
        flag_cloud = 1;
        usleep(10000);
    }

}


int main(int argc, char** argv)
{

    ros::init(argc, argv, "test_transformation");
    ros::NodeHandle nh;
    ros::AsyncSpinner spinner(4);

    ros::Subscriber sub = nh.subscribe<sensor_msgs::Image> ("camera/color/image_raw", 2, imageCallback);
    ros::Subscriber sub2 = nh.subscribe<sensor_msgs::PointCloud2> ("/camera/depth_registered/points", 1, centroid_region_cb);

    centroid_row_pntr = new int [number_of_points];
    centroid_col_pntr = new int [number_of_points];


    spinner.start();


    while(!flag_cloud | !flag_img | (pcl_input->height == 0) | (pcl_input->width == 0) )
    {
        usleep(100000);
    }



    spinner.stop();

    std::cout << "\n\nboth spinners have been stopped\n";
    std::cout <<"\nk: " << k;


    //    moveit::planning_interface::MoveGroup group("ur10_arm");

    geometry_msgs::PointStamped point_wrt_kinect;
    geometry_msgs::PointStamped point_wrt_world;
    tf::TransformListener listener;





    std::string path = ros::package::getPath("extrinsic_calibration");
    img = cv::imread(path + "/img.png");
    cv::namedWindow("view");
    cv::startWindowThread();
    cv::setMouseCallback("view", CallBackFunc, NULL);




    while(!flag_click)
    {
        cv::imshow("view", img);
        cv::waitKey(25);
    }




    point_wrt_kinect.header.frame_id = "camera_color_optical_frame";

    point_wrt_kinect.point.x = pcl_input->at(*(centroid_col_pntr + 0), *(centroid_row_pntr + 0)).x;
    point_wrt_kinect.point.y = pcl_input->at(*(centroid_col_pntr + 0), *(centroid_row_pntr + 0)).y;
    point_wrt_kinect.point.z = pcl_input->at(*(centroid_col_pntr + 0), *(centroid_row_pntr + 0)).z;

    listener.waitForTransform( "base_link", "camera_color_optical_frame", ros::Time(0), ros::Duration(3));
    listener.transformPoint("base_link", point_wrt_kinect, point_wrt_world);




    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_test_marker", 10);


    while(ros::ok())
    {

        std::cout << "See RVIZ" << std::endl;
        visualization_msgs::Marker centroid_point;
        centroid_point.header.frame_id = "base_link";
        centroid_point.header.stamp = ros::Time::now();
        centroid_point.ns = "test_points";
        centroid_point.action = visualization_msgs::Marker::ADD;
        centroid_point.pose.orientation.w  = 1.0;


        centroid_point.id = 0;



        centroid_point.type = visualization_msgs::Marker::POINTS;



        centroid_point.scale.x = 0.02;
        centroid_point.scale.y = 0.02;

        centroid_point.color.r = 1.0f;
        centroid_point.color.a = 1.0;


        for (int i=0 ;i<number_of_points; i++)
        {
            geometry_msgs::Point p_c;
            p_c = point_wrt_world.point;
            centroid_point.points.push_back(p_c);

            std::cout << "point wrt_world -> x " << point_wrt_world.point.x
                      << ",  y " << point_wrt_world.point.y
                      << ", z " << point_wrt_world.point.z << std::endl;

            std::cout << "point_wrt_kinect -> x " << point_wrt_kinect.point.x
                      << ",  y " << point_wrt_kinect.point.y
                      << ", z " << point_wrt_kinect.point.z << std::endl;

        }

        marker_pub.publish(centroid_point);

    }



    pcl::io::savePCDFileASCII ("test_pcd.pcd", *pcl_input);
    cv::imwrite("test_img.png", img);







    return 0;

}
