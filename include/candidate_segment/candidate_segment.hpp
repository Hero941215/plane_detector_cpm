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
    CandidateSegment(double candidate_edge_ratio = 0.05, double disjoint_th = 1.5);

    void setRangeImageProjParam(int N_SCAN, int Horizon_SCAN, float ang_res_y, float ang_bottom);
    int SegmentPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pcPtr);

    // 计算距离图像，每个像素存储的是相对时间戳
    Eigen::MatrixXd ComputeRangeImageWithTS(pcl::PointCloud<pcl::PointXYZI>::Ptr pcPtr);

    // 紧凑的检测结果
    pcl::PointCloud<pcl::PointXYZI>::Ptr get_res_edge_pc_();
    pcl::PointCloud<pcl::PointXYZI>::Ptr get_res_plane_pc_();
    pcl::PointCloud<pcl::PointXYZI>::Ptr get_res_disjoint_pc_();
    pcl::PointCloud<pcl::PointXYZI>::Ptr get_res_edge_cand_pc_();
    pcl::PointCloud<pcl::PointXYZI>::Ptr get_res_small_disjoint_pc_();

    // 转换成距离图像后， 对应的点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr get_range_pc_();
    cv::Mat get_plane_candidate_range_();
    cv::Mat get_edge_candidate_range_();

protected:

    void reset();
    void compute_range_image_();
    void compute_cloud_smoothness_();
    void disjoint_segment_();
    void generate_candidate_edge_();
    void non_maximum_suppression(); 
    void get_feature_range_image();

    float smooth(int m);
    float continous_distance(int i);

    bool initialized_;

    // 输入点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr corrected_point_cloud_; 
    pcl::PointCloud<pcl::PointXYZI>::Ptr range_point_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr tight_point_cloud_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr edge_point_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr disjoint_point_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr plane_point_cloud_;

    // 不参与平面候选的点云组
    pcl::PointCloud<pcl::PointXYZI>::Ptr edge_cand_point_cloud_; // 95/5*cm候选边缘点
    pcl::PointCloud<pcl::PointXYZI>::Ptr small_disjoint_point_cloud_; //中间点少于9个的小分段点云

    // 点云分割容器
    cv::Mat range_image_, plane_feature_candidate_, edge_feature_candidate_;

    // 点云光滑度
    std::vector<int> scanStartInd_; // 距离图像点云的扫描线首尾索引
    std::vector<int> scanEndInd_;

    std::vector<float> cloudCurvature_; // 紧凑点云的光滑度
    std::vector<smoothness_t> cloud_smoothness_; // 基于光滑度进行排序，从小到大
    std::vector<float> cloud_edge_smoothness_; // 95% + 5*c_mid

    std::vector<std::vector<int> > disjoint_point_index_;
    std::vector<std::vector<std::pair<int, int> > > effective_continous_segment_;
    std::vector<std::pair<int, int> > small_continous_segment_;   // 中间点少于9个的小分段组，可以理解为噪声组

    // 距离图像投影参数
    int N_SCAN_;
    int Horizon_SCAN_;
    float ang_res_x_;
    float ang_res_y_;
    float ang_bottom_;

    // 提取候选边缘时的阈值
    double candidate_edge_ratio_;
    double disjoint_th_;
};


#endif