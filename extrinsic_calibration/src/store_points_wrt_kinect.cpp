#include <ros/ros.h>
#include <ros/package.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>

#include <pcl/kdtree/kdtree_flann.h>

#include <fstream>
#include <iostream>




char a;

int *centroid_row_pntr;
int *centroid_col_pntr;

bool flag_click = false;

int number_of_points = 9;
int number_of_points_selected = 0;





cv::Mat current_image;
void image_call_back(const sensor_msgs::ImageConstPtr &msg)
{
    cout << "Hello, World!5" << endl;
    current_image = cv_bridge::toCvShare(msg, "bgr8")->image ;
    cout << "Hello, World!6" << endl;
    return;
}


pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_point_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
void centroid_region_cb (const sensor_msgs::PointCloud2ConstPtr& msg)
{

    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*msg, pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2,*current_point_cloud);
    return;
}


void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{

    if(!flag_click)
    {

        if  ( event == cv::EVENT_LBUTTONDOWN )
        {
            std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;

            *(centroid_row_pntr + number_of_points_selected) = y;
            *(centroid_col_pntr + number_of_points_selected) = x;

            if ( std::isnan(current_point_cloud->at(x, y).x))
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



int main(int argc, char** argv)
{

    cout << "Hello, World!1" << endl;
    ros::init(argc, argv, "subscribe_image2");
    ros::NodeHandle nh;

    ros::Subscriber sub = nh.subscribe<sensor_msgs::Image> ("camera/color/image_raw", 2, image_call_back);
    ros::Subscriber sub2 = nh.subscribe<sensor_msgs::PointCloud2> ("/camera/depth_registered/points", 1, centroid_region_cb);

    centroid_row_pntr = new int [number_of_points];
    centroid_col_pntr = new int [number_of_points];

    std::string path = ros::package::getPath("extrinsic_calibration");
    std::string data_folder_path(path + "/data_files/");

    cout << "Hello, World!2" << endl;

    while (ros::ok())
    {

        while(current_image.rows == 0 or current_point_cloud->height == 0)
        {
            ros::spinOnce();
        }

        break;
    }


    cout << "Hello, World!3" << endl;

    cv::namedWindow("view");
    cv::startWindowThread();
    cv::setMouseCallback("view", CallBackFunc, NULL);

    cout << "Hello, World!4" << endl;

    while(!flag_click)
    {
        cv::imshow("view", current_image);
        cv::waitKey(25);
    }


    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud (current_point_cloud);

    std::cout << current_point_cloud->height << ", " << current_point_cloud->width << std::endl;

    pcl::PointXYZRGB searchPoint;

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    float radius = 0.02;

    std::ofstream myfile;

    myfile.open (data_folder_path + "points_wrt_kinnect.txt", std::ios::out);



    int i = 0;
    while (i < number_of_points)
    {

        pointIdxRadiusSearch.clear();
        pointRadiusSquaredDistance.clear();


        searchPoint.x = current_point_cloud->at(*(centroid_col_pntr + i), *(centroid_row_pntr + i)).x;
        searchPoint.y = current_point_cloud->at(*(centroid_col_pntr + i), *(centroid_row_pntr + i)).y;
        searchPoint.z = current_point_cloud->at(*(centroid_col_pntr + i), *(centroid_row_pntr + i)).z;
        searchPoint.r = current_point_cloud->at(*(centroid_col_pntr + i), *(centroid_row_pntr + i)).r;
        searchPoint.g = current_point_cloud->at(*(centroid_col_pntr + i), *(centroid_row_pntr + i)).g;
        searchPoint.b = current_point_cloud->at(*(centroid_col_pntr + i), *(centroid_row_pntr + i)).b;



        kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);

        for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
        {
            current_point_cloud->points[pointIdxRadiusSearch[i]].r = 0.0;
            current_point_cloud->points[pointIdxRadiusSearch[i]].g = 0.0;
            current_point_cloud->points[pointIdxRadiusSearch[i]].b = 255.0;
        }

        if (i == 0)
        {
            for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
            {
                current_point_cloud->points[pointIdxRadiusSearch[i]].r = 255.0;
                current_point_cloud->points[pointIdxRadiusSearch[i]].g = 255.0;
                current_point_cloud->points[pointIdxRadiusSearch[i]].b = 255.0;
            }
        }
        if (i == 1)
        {
            for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
            {
                current_point_cloud->points[pointIdxRadiusSearch[i]].r = 0.0;
                current_point_cloud->points[pointIdxRadiusSearch[i]].g = 255.0;
                current_point_cloud->points[pointIdxRadiusSearch[i]].b = 0.0;
            }
        }


        std::cout << "\nx: " << searchPoint.x
                  << ", y: " << searchPoint.y
                  << ", z: " << searchPoint.z << "\n";


        myfile << std::setprecision(15) << searchPoint.x << ", "
               << searchPoint.y << ", "
               << searchPoint.z << "\n";



        i = i+1;
    }




//    pcl::visualization::CloudViewer cloud_viewer("CLOUD");
//    cloud_viewer.showCloud(current_point_cloud);

    std::cout <<"\npress any key to continue\n";
    std::cin >> a;


    return 0;

}
