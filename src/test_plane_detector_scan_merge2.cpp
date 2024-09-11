#include "plane_detector/plane_detector.hpp"
#include "candidate_segment/candidate_segment.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>

#include <limits>
#include <string>
#include <vector>
#include <set>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "tic_toc.h"
#include "stdlib.h"

#include <chrono>
#include <iostream>
#include <thread>

#define PI 3.1415926

using namespace std;
using namespace cv;

const int N_SCAN = 64;
const int Horizon_SCAN = 1800;
const float ang_res_x = 360.0/float(Horizon_SCAN);
const float ang_res_y = 0.427;
const float ang_bottom = 24.9;

// const int N_SCAN = 32;
// const int Horizon_SCAN = 1800;
// const float ang_res_x = 360.0/float(Horizon_SCAN);
// const float ang_res_y = 2.61;
// const float ang_bottom = 2.487;

// const int N_SCAN = 16;
// const int Horizon_SCAN = 1800;
// const float ang_res_x = 0.2;
// const float ang_res_y = 2.0;
// const float ang_bottom = 15.0+0.1;
// const int groundScanInd = 7;

void showMergePlane2(cv::Mat &m, PlaneDetector &p);
bool checkMergePlane(DetectedPlane dp1, DetectedPlane dp_cur);
void showOnly(cv::Mat &m, PlaneDetector &p);
void checkPlane(PlaneDetector &p);

int main(int argc, char **argv) 
{

    if (argc < 2) {
        printf("Usage: %s <point cloud> [<delta> <epsilon> <gamma> <theta>]\n", argv[0]);
        return -1;
    }

    // 计算距离图像和索引格式的点云
    const std::string pcd_file(argv[1]);
    pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud(
        new pcl::PointCloud<pcl::PointXYZI>);

    if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file, *point_cloud) == -1) {
        std::cout << "Couldn't read pcd file!\n";
        return -1;
    }

    double delta = 0.5, epsilon = 0.0012, gamma = 0.15; // Default parameters
    // double delta = 0.2, epsilon = 0.0008, gamma = 0.1; // Default parameters
    int theta = 30;

    std::cout << "point_cloud->points.size(): " << point_cloud->points.size() << std::endl;

    CandidateSegment cs(0.05, 5);
    cs.setRangeImageProjParam(N_SCAN, Horizon_SCAN, ang_res_y, ang_bottom);
    cs.SegmentPointCloud(point_cloud);

    pcl::PointCloud<pcl::PointXYZI>::Ptr range_pc(
        new pcl::PointCloud<pcl::PointXYZI>);

    range_pc = cs.get_range_pc_();

    cv::Mat plane_candidate_range = cs.get_plane_candidate_range_();

    TicToc t_whole;

    PlaneDetector pd(delta, epsilon, gamma, theta);
    pd.setRawPointCloud(range_pc);

    int plane_num = pd.detectPlanes(plane_candidate_range);

    printf("whole detection time %f ms \n \n", t_whole.toc());

    // checkPlane(pd);

    showOnly(plane_candidate_range, pd);

    return 0;
}

bool checkMergePlane(DetectedPlane dp1, DetectedPlane dp_cur)
{
    double th = 5;

    // 计算cp 
    double norm = dp1.v.norm();
    Eigen::Vector3d correct_v = dp1.v / norm;
    double correct_d = dp1.d / norm;
    Eigen::Vector3d cp1 = -correct_v*correct_d;
    
    norm = dp_cur.v.norm();
    correct_v = dp_cur.v / norm;
    correct_d = dp_cur.d / norm;
    Eigen::Vector3d cp_cur = -correct_v*correct_d;

    double cosValNew = cp1.dot(cp_cur) /(cp1.norm()*cp_cur.norm()); //角度cos值
    double angleNew = acos(cosValNew) * 180 / M_PI;     //弧度角

    // std::cout << "angleNew: " << angleNew << std::endl;

    if(angleNew<(0+th) || angleNew>(180-th))
        return true;
    else
        return false;
}

void checkPlane(PlaneDetector &p)
{
    srand((unsigned)time(NULL));

    pcl::PointCloud<pcl::PointXYZI>::Ptr plane(new pcl::PointCloud<pcl::PointXYZI>);

    std::vector<std::vector<int> > detected_planes_region = p.getRegions();
    std::vector<std::vector<int> > final_merge_plane_group = p.getMergedPlanes();

    int i = (rand() % (final_merge_plane_group.size()-0+1)) + 0;
    std::vector<int> merge_plane = final_merge_plane_group[i];
    for(int j=0; j<merge_plane.size(); j++)
    {
        int detect_plane_id = merge_plane[j];
        std::vector<int> cur_region = detected_planes_region[detect_plane_id];

        for(int k=0; k<cur_region.size(); k++)
        {
            int m = cur_region[k]%Horizon_SCAN;
            int n = cur_region[k]/Horizon_SCAN;

            // 提取3d点云
            pcl::PointXYZI cur_p = p.get3DPoint2(cur_region[k]);
            plane->push_back(cur_p);
        }
    }

    // 计算均值点和协方差
    Eigen::Matrix3f cov;
    Eigen::Vector4f pc_mean;
    pcl::computeMeanAndCovarianceMatrix(*plane, cov, pc_mean);
    std::cout << "cov: " << cov << std::endl;
    Eigen::Matrix3f local_cov = cov;

    // 进行分解，计算权重
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> _es;
    _es.compute(cov);
    Eigen::Vector3f ev = _es.eigenvalues(); // 按从小到大排列

    double weight;
    if(ev.x()>0 && ev.y()>0 && ev.z()>0)
    {
        weight = (std::sqrt(ev.y())-std::sqrt(ev.x())) / std::sqrt(ev.z());
        std::cout << "weight: " << weight << std::endl;
    }

    // 重复拷贝平面两次
    pcl::PointCloud<pcl::PointXYZI>::Ptr double_plane(new pcl::PointCloud<pcl::PointXYZI>);
    *double_plane += *plane;
    *double_plane += *plane;

    std::cout << "plane->size(): " << plane->size() << " double_plane->size(): " << double_plane->size() << std::endl;

    // 计算均值点和协方差
    pcl::computeMeanAndCovarianceMatrix(*double_plane, cov, pc_mean);
    std::cout << "cov2: " << cov << std::endl;

    // 进行分解，计算权重
    _es.compute(cov);
    ev = _es.eigenvalues(); // 按从小到大排列

    if(ev.x()>0 && ev.y()>0 && ev.z()>0)
    {
        weight = (std::sqrt(ev.y())-std::sqrt(ev.x())) / std::sqrt(ev.z());
        std::cout << "weight2: " << weight << std::endl;
    }

    // 给定一个6自由度变换
    float roll = PI/6;
    float pitch = PI/4;
    float yaw = PI/3;

    double s1 = sin(roll);
    double c1 = cos(roll);
    double s2 = sin(pitch);
    double c2 = cos(pitch);
    double s3 = sin(yaw);
    double c3 = cos(yaw);

    Eigen::Matrix3f Rz;
    Rz << cos(yaw), -sin(yaw), 0,
            sin(yaw), cos(yaw), 0,
            0, 0, 1;

    Eigen::Matrix3f Ry;
    Ry << cos(pitch), 0., sin(pitch),
            0., 1., 0.,
            -sin(pitch), 0., cos(pitch);

    Eigen::Matrix3f Rx;
    Rx << 1., 0., 0.,
            0., cos(roll), -sin(roll),
            0., sin(roll), cos(roll);

    Eigen::Matrix3f R1 = Rz * Ry * Rx;
    Eigen::Vector3f euler_angles = R1.eulerAngles(2, 1, 0); 
    cout << "yaw(z) pitch(y) roll(x) = " << euler_angles.transpose() << endl;
    Eigen::Vector3f t1(1,2,3);
    Eigen::Isometry3f Trans; Trans.setIdentity();
    Trans.rotate(R1);
    Trans.pretranslate(t1);

    // 对plane进行变换，然后计算协方差
    pcl::PointCloud<pcl::PointXYZI>::Ptr landmark(
       		new pcl::PointCloud<pcl::PointXYZI>);
    pcl::transformPointCloud(*plane, *landmark, Trans.matrix());

    pcl::computeMeanAndCovarianceMatrix(*landmark, cov, pc_mean);
    std::cout << "map-cov-1: " << cov << std::endl;

    _es.compute(cov);
    ev = _es.eigenvalues(); // 按从小到大排列

    if(ev.x()>0 && ev.y()>0 && ev.z()>0)
    {
        weight = (std::sqrt(ev.y())-std::sqrt(ev.x())) / std::sqrt(ev.z());
        std::cout << "weight3: " << weight << std::endl;
    }

    // 使用变换，对局部协方差进行变换
    Eigen::Matrix3f globalcov = R1 * local_cov * R1.transpose();
    std::cout << "map-cov-2: " << globalcov << std::endl;

    _es.compute(globalcov);
    ev = _es.eigenvalues(); // 按从小到大排列

    if(ev.x()>0 && ev.y()>0 && ev.z()>0)
    {
        weight = (std::sqrt(ev.y())-std::sqrt(ev.x())) / std::sqrt(ev.z());
        std::cout << "weight4: " << weight << std::endl;
    }
}

void showOnly(cv::Mat &m, PlaneDetector &p)
{
    srand((unsigned)time(NULL)); 

    int width = m.cols; // 距离图像宽度

    // 显示容器
    pcl::PointXYZRGB rgb_point;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_pointcloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);

    // 将合并平面使用相同的颜色显示
    std::vector<std::vector<int> > detected_planes_region = p.getRegions();
    std::vector<std::vector<int> > final_merge_plane_group = p.getMergedPlanes();
    for(int i=0; i<final_merge_plane_group.size(); i++)
    {
        
        // int i = (rand() % (final_merge_plane_group.size()-0+1)) + 0;
        int r = (rand() % (255-0+1)) + 0;
        int g = (rand() % (255-0+1)) + 0;
        int b = (rand() % (255-0+1)) + 0;

        std::vector<int> merge_plane = final_merge_plane_group[i];

        for(int j=0; j<merge_plane.size(); j++)
        {
            int detect_plane_id = merge_plane[j];
            std::vector<int> cur_region = detected_planes_region[detect_plane_id];

            for(int k=0; k<cur_region.size(); k++)
            {
                int m = cur_region[k]%width;
                int n = cur_region[k]/width;

                // 提取3d点云
                pcl::PointXYZI cur_p = p.get3DPoint2(cur_region[k]);
                rgb_point.x = cur_p.x;
                rgb_point.y = cur_p.y;
                rgb_point.z = cur_p.z;
                rgb_point.r = r;
                rgb_point.g = g;
                rgb_point.b = b; 

                rgb_pointcloud->push_back(rgb_point);
            }
        }
    }

    std::cout << "rgb_pointcloud->size(): " << rgb_pointcloud->size() << std::endl;

    // pcl 显示所有检测到的平面
    pcl::visualization::PCLVisualizer::Ptr visualizer(
        new pcl::visualization::PCLVisualizer("Plane Detection Visualizer2"));
    visualizer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>
        rgb_color_handler(rgb_pointcloud);
    visualizer->addPointCloud<pcl::PointXYZRGB>(rgb_pointcloud, rgb_color_handler,
                                                "RGB PointCloud");
    visualizer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "RGB PointCloud");
    visualizer->addCoordinateSystem(5.0);                                                                  

    while (!visualizer->wasStopped()) {
        visualizer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}
