#ifndef CANDIDATE_SEGMENT__
#define CANDIDATE_SEGMENT__

#include <vector>
#include <random>
#include <queue>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Eigenvalues>
// #include <boost/array.hpp>
// #include <boost/thread.hpp>
#include <exception>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class CandidateSegment
{

public:
    CandidateSegment(double candidate_edge_ratio = 0.05, double disjoint_th = 1.5, int max_continous_point_number = 15);
    
    void setRangeImageProjParam(int N_SCAN, int Horizon_SCAN, float ang_res_y, float ang_bottom);
    int SegmentPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pcPtr);

    pcl::PointCloud<pcl::PointXYZI>::Ptr get_res_edge_pc_();
    pcl::PointCloud<pcl::PointXYZI>::Ptr get_res_plane_pc_();
    pcl::PointCloud<pcl::PointXYZI>::Ptr get_res_disjoint_pc_();
    pcl::PointCloud<pcl::PointXYZI>::Ptr get_res_edge_cand_pc_();

protected:

    void compute_range_image_();
    void compute_cloud_smoothness_();
    void generate_candidate_edge_();
    void non_maximum_suppression(); 
    void get_feature_range_image();

    void set_disjoint_mark(int i); 
    float smooth(int m);
    float continous_distance(int i);

    bool initialized_;

    // 输入点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr raw_point_cloud_; 
    pcl::PointCloud<pcl::PointXYZI>::Ptr range_point_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr tight_point_cloud_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr edge_point_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr disjoint_point_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr plane_point_cloud_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr edge_cand_point_cloud_;

    // 点云分割容器
    cv::Mat range_image_, plane_feature_candidate_, edge_feature_candidate_;

    // 点云光滑度
    std::vector<int> scanStartInd_; // 距离图像点云的扫描线首尾索引
    std::vector<int> scanEndInd_;

    std::vector<float> cloudCurvature_; // 紧凑点云的光滑度
    std::vector<smoothness_t> cloud_smoothness_; // 基于光滑度进行排序，从小到大
    std::vector<float> cloud_edge_smoothness_; // 95% + 5*c_mid

    std::vector<int> disjoint_index_;
    // 距离图像投影参数
    int N_SCAN_;
    int Horizon_SCAN_;
    float ang_res_x_;
    float ang_res_y_;
    float ang_bottom_;

    // 提取候选边缘时的阈值
    double candidate_edge_ratio_;
    double disjoint_th_;
    int max_continous_point_number_;
    
};

#endif