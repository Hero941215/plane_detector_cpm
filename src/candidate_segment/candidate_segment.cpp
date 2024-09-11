#include "candidate_segment/candidate_segment.hpp"
#include "tic_toc.h"

CandidateSegment::CandidateSegment(double candidate_edge_ratio, double disjoint_th):
                candidate_edge_ratio_(candidate_edge_ratio), disjoint_th_(disjoint_th)
{
    corrected_point_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    range_point_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    tight_point_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());

    edge_point_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    disjoint_point_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    plane_point_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());

    edge_cand_point_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    small_disjoint_point_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());

    std::fill(cloud_edge_smoothness_.begin(), cloud_edge_smoothness_.end(), 0);
}

pcl::PointCloud<pcl::PointXYZI>::Ptr CandidateSegment::get_res_small_disjoint_pc_()
{
    return small_disjoint_point_cloud_;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr CandidateSegment::get_res_edge_cand_pc_()
{
    return edge_cand_point_cloud_;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr CandidateSegment::get_res_edge_pc_()
{
    return edge_point_cloud_;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr CandidateSegment::get_res_disjoint_pc_()
{
    return disjoint_point_cloud_;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr CandidateSegment::get_res_plane_pc_()
{
    return plane_point_cloud_;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr CandidateSegment::get_range_pc_()
{
    return range_point_cloud_;
}

cv::Mat CandidateSegment::get_plane_candidate_range_()
{
    return plane_feature_candidate_;
}

cv::Mat CandidateSegment::get_edge_candidate_range_()
{
    return edge_feature_candidate_;
}

int CandidateSegment::SegmentPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pcPtr)
{
    if (!initialized_) {
        // The camera parameters have to be set before detecting planes
        return 0;
    }

    corrected_point_cloud_ = pcPtr;

    reset();

    compute_range_image_();

    compute_cloud_smoothness_();

    disjoint_segment_();

    generate_candidate_edge_();

    non_maximum_suppression();

    get_feature_range_image();

}

// void CandidateSegment::reset()
// {
//     range_image_ = cv::Mat::zeros(N_SCAN_, Horizon_SCAN_, CV_32FC1);
//     edge_feature_candidate_ = cv::Mat::zeros(N_SCAN_, Horizon_SCAN_, CV_32FC1);
//     plane_feature_candidate_ = cv::Mat::zeros(N_SCAN_, Horizon_SCAN_, CV_32FC1);

//     range_point_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
//     range_point_cloud_->points.resize(N_SCAN_*Horizon_SCAN_);
//     tight_point_cloud_->clear();

//     std::fill(scanStartInd_.begin(), scanStartInd_.end(), 0);
//     std::fill(scanEndInd_.begin(), scanEndInd_.end(), 0);

//     disjoint_point_index_.clear();
//     disjoint_point_cloud_->clear();
//     effective_continous_segment_.clear();

//     edge_point_cloud_->clear();
//     plane_point_cloud_->clear();

//     std::fill(cloudCurvature_.begin(), cloudCurvature_.end(), 0);
//     // cloud_smoothness_.clear();
//     std::vector<smoothness_t> cloud_smoothness;
//     cloud_smoothness.resize(N_SCAN_*Horizon_SCAN_);
//     cloud_smoothness.swap(cloud_smoothness_);
// }

void CandidateSegment::reset()
{
    range_image_ = cv::Mat::zeros(N_SCAN_, Horizon_SCAN_, CV_32FC1); // 初始化为全0矩阵
    edge_feature_candidate_ = cv::Mat::zeros(N_SCAN_, Horizon_SCAN_, CV_32FC1);
    plane_feature_candidate_ = cv::Mat::zeros(N_SCAN_, Horizon_SCAN_, CV_32FC1);

    range_point_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    range_point_cloud_->points.resize(N_SCAN_*Horizon_SCAN_);
    tight_point_cloud_->clear();

    std::fill(scanStartInd_.begin(), scanStartInd_.end(), 0);
    std::fill(scanEndInd_.begin(), scanEndInd_.end(), 0);

    disjoint_point_cloud_->clear();

    std::vector<std::vector<int> > disjoint_point_index;
    disjoint_point_index.resize(N_SCAN_);
    disjoint_point_index.swap(disjoint_point_index_);
    
    std::vector<std::vector<std::pair<int, int> > > effective_continous_segment;
    effective_continous_segment.resize(N_SCAN_);
    effective_continous_segment.swap(effective_continous_segment_);

    small_continous_segment_.clear();
    
    edge_point_cloud_->clear();
    plane_point_cloud_->clear();

    edge_cand_point_cloud_->clear();
    small_disjoint_point_cloud_->clear();

    std::fill(cloudCurvature_.begin(), cloudCurvature_.end(), 0);
    // cloud_smoothness_.clear();
    std::vector<smoothness_t> cloud_smoothness;
    cloud_smoothness.resize(N_SCAN_*Horizon_SCAN_);
    cloud_smoothness.swap(cloud_smoothness_);
    std::fill(cloud_edge_smoothness_.begin(), cloud_edge_smoothness_.end(), 0);
}

Eigen::MatrixXd CandidateSegment::ComputeRangeImageWithTS(pcl::PointCloud<pcl::PointXYZI>::Ptr pcPtr)
{

    // 找到首尾角度
    int pc_size = pcPtr->points.size();
	float fstartOrientation = -atan2(pcPtr->points[0].y, pcPtr->points[0].x);
	float fendOrientation = -atan2(pcPtr->points[pcPtr->points.size() - 1].y,
                                                     pcPtr->points[pcPtr->points.size() - 1].x) + 2 * M_PI;

    if (fendOrientation - fstartOrientation > 3 * M_PI)
    {
        fendOrientation -= 2 * M_PI;
    }
    else if (fendOrientation - fstartOrientation < M_PI)
    {
        fendOrientation += 2 * M_PI;
    }

    float verticalAngle, horizonAngle, range_value;
    size_t rowIdn, columnIdn, index, cloudSize; 
    pcl::PointXYZI thisPoint;

    range_image_ = cv::Mat::zeros(N_SCAN_, Horizon_SCAN_, CV_32FC1); // 存储深度
    Eigen::MatrixXd RelTS = Eigen::MatrixXd::Zero(N_SCAN_, Horizon_SCAN_);  // 存储相对时间戳
    range_point_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    range_point_cloud_->points.resize(N_SCAN_*Horizon_SCAN_);
    
    float min_rel_time = 1000;
    float min_ori = 1000;
    float max_rel_time = -1000;
    float max_ori = -1000;
    int out_range_num = 0;
    int ok_num = 0;

    std::vector<int> check_ts; check_ts.resize(N_SCAN_);

    bool halfPassed = false;
    int raw_point_id = 0;
    int i=0;
    for (const auto& point : pcPtr->points) {
        thisPoint.x = point.x;
        thisPoint.y = point.y;
        thisPoint.z = point.z;
        thisPoint.intensity = raw_point_id;
        raw_point_id++;

        // 计算相对时间戳
        float ori = -atan2(thisPoint.y, thisPoint.x);
        if (!halfPassed)
        { 
            if (ori < fstartOrientation - M_PI / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > fstartOrientation + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }

            if (ori - fstartOrientation > M_PI)
            {
                halfPassed = true;
            }
        }
        else
        {
            ori += 2 * M_PI;
            if (ori < fendOrientation - M_PI * 3 / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > fendOrientation + M_PI / 2)
            {
                ori -= 2 * M_PI;
            }
        }

        float relTime = (ori - fstartOrientation) / (fendOrientation - fstartOrientation);
        ok_num++; 
        
        if(relTime<min_rel_time)
        {
            min_ori = ori;
            min_rel_time = relTime;
        }

        if(relTime>max_rel_time)
        {
            max_ori = ori;
            max_rel_time = relTime;
        }

        // 计算竖直方向上的角度（雷达的第几线）
        verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
            
        rowIdn = (verticalAngle + ang_bottom_) / ang_res_y_;
        if(raw_point_id==0)
            std::cout << rowIdn << std::endl;

        if (rowIdn < 0 || rowIdn >= N_SCAN_)
        {
            // if(relTime<0)
            // {
            //     std::cout << rowIdn << std::endl;
            // }
            continue;
        }
        // else
        // {
        //     if(i%15==0)
        //         std::cout << rowIdn << std::endl;
        // }
            
        // if(relTime<0)
        //     std::cout << "i: " << i << " relTime: " << relTime << std::endl;

        horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

        columnIdn = -round((horizonAngle-90.0)/ang_res_x_) + Horizon_SCAN_/2;
        if (columnIdn >= Horizon_SCAN_)
            columnIdn -= Horizon_SCAN_;

        if (columnIdn < 0 || columnIdn >= Horizon_SCAN_)
            continue;

        if(rowIdn==63 && i%15==0)
            std::cout << columnIdn << " " << relTime << std::endl;

        RelTS(rowIdn, columnIdn) = relTime;

        range_value = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
        range_image_.at<float>(rowIdn, columnIdn) = range_value;

        index = columnIdn  + rowIdn * Horizon_SCAN_;
        range_point_cloud_->points[index] = thisPoint;

        if(relTime<0)
        {
            out_range_num++;
            check_ts[rowIdn] = 1;
        }

        i++;
    }

    for(int i=0; i<64; i++)
    {
        if(check_ts[i])
            std::cout << "RelTS: " << i << " " << RelTS.row(15) << std::endl << std::endl << std::endl << std::endl;
    }
    

    std::cout << "ok_num: " << ok_num << " out_range_num: " << out_range_num
              << " ratio: " << 1.0*out_range_num/ok_num << std::endl;
              
    std::cout << "min_rel_time: " << min_rel_time << " min_ori: " << min_ori << std::endl;
    std::cout << "max_rel_time: " << max_rel_time << " max_ori: " << max_ori << std::endl;

    return RelTS;
}

void CandidateSegment::compute_range_image_()
{
    
    float verticalAngle, horizonAngle, range_value;
    size_t rowIdn, columnIdn, index, cloudSize; 
    pcl::PointXYZI thisPoint;

    // 距离图像及对应的离散点云
    int raw_point_id = 0;
    for (const auto& point : corrected_point_cloud_->points) {
        thisPoint.x = point.x;
        thisPoint.y = point.y;
        thisPoint.z = point.z;
        thisPoint.intensity = raw_point_id;
        raw_point_id++;

        // 计算竖直方向上的角度（雷达的第几线）
        verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
            
        rowIdn = (verticalAngle + ang_bottom_) / ang_res_y_;
        if (rowIdn < 0 || rowIdn >= N_SCAN_)
            continue;

        horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

        columnIdn = -round((horizonAngle-90.0)/ang_res_x_) + Horizon_SCAN_/2;
        if (columnIdn >= Horizon_SCAN_)
            columnIdn -= Horizon_SCAN_;

        if (columnIdn < 0 || columnIdn >= Horizon_SCAN_)
            continue;

        range_value = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
        range_image_.at<float>(rowIdn, columnIdn) = range_value;

        // thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

        index = columnIdn  + rowIdn * Horizon_SCAN_;
        range_point_cloud_->points[index] = thisPoint;
    }

    // 紧凑点云和扫描线在紧凑点云中的首尾索引
    int sizeOfSegCloud = 0;
    for (size_t i = 0; i < N_SCAN_; ++i) 
    {
        scanStartInd_[i] = sizeOfSegCloud + 5; // 一条扫描线的第6个点

        for (size_t j = 0; j < Horizon_SCAN_; ++j) 
        {
            if(range_image_.at<float>(i,j) > 0.1) // 说明是有效深度值
            {
                index = j  + i * Horizon_SCAN_;
                thisPoint = range_point_cloud_->points[index];
                thisPoint.intensity = index;  // 存储一维索引
                tight_point_cloud_->push_back(thisPoint);
                ++sizeOfSegCloud;
            }

        }

        scanEndInd_[i] = sizeOfSegCloud-1 - 5; // 一条扫描线的倒数第6个点
    }

}

// 按扫描线计算非连续点，保留有效连续段
void CandidateSegment::disjoint_segment_()
{
    
    for (int i = 0; i < N_SCAN_; i++) 
    {
        int sp = scanStartInd_[i];
        int ep = scanEndInd_[i];

        if (sp >= ep)
            continue;

        disjoint_point_index_[i].push_back(sp);

        for (int j = sp+1; j < ep; j++) // 要比最后一个计算光滑度的点少一个进行遍历
        {
            if(continous_distance(j)>disjoint_th_) // 不连续的点
            {
                disjoint_point_index_[i].push_back(j);
                disjoint_point_cloud_->push_back(tight_point_cloud_->points[j]);
            }
        }

        disjoint_point_index_[i].push_back(ep);
    }

    int small_dis_segment_num = 0;
    int effective_seg_num = 0;
    for (int i = 0; i < N_SCAN_; i++) 
    {

        if(disjoint_point_index_[i].size()!=0)
        {
            for(int j=0; j<disjoint_point_index_[i].size()-1; j++)
            {
                int pA = disjoint_point_index_[i][j];
                int pB = disjoint_point_index_[i][j+1];

                // std::cout << "pA: " << pA << " pB: " << pB << std::endl;

                int continous_point_num = pB-pA-1;
                if(continous_point_num>=9) // 与使用左右5个点计算光滑度相关
                {
                    effective_seg_num++;
                    effective_continous_segment_[i].push_back(std::make_pair(pA, pB));
                }
                else
                {
                    if(continous_point_num>=3) // 含点组
                        small_dis_segment_num++;
                    
                    small_continous_segment_.push_back(std::make_pair(pA, pB)); // 无论是否夹点，都会被放进去，有的点会被重复放入
                }
                    
            }
        }
    }
    std::cout << "effective_continous_segment_.size(): " << effective_seg_num << std::endl;
    std::cout << "small_dis_segment_num: " << small_dis_segment_num << std::endl;

     // 小分段点云，粗糙度很大，也直接移除掉
    pcl::PointXYZI thisPoint;
    int real_small_point_num = 0;
    for(int i=0; i<small_continous_segment_.size(); i++)
    {
        int sp = small_continous_segment_[i].first;
        int ep = small_continous_segment_[i].second;

        real_small_point_num += (ep - sp -1);

        for(int j=sp; j<=ep; j++)
        {
            thisPoint = tight_point_cloud_->points[j];
            small_disjoint_point_cloud_->push_back(thisPoint);
        }
    }
    std::cout << "real_small_point_num: " << real_small_point_num << std::endl;
}

// 对于每个有效连续分段， 对其中的有效点进行排序， 并提取候选边缘

// void CandidateSegment::generate_candidate_edge_()
// {
//     float min_smooth_th = 0.5; // 我们会将光滑度大于0.5的所有点都视为边缘集合点

//     for (int i = 0; i < N_SCAN_; i++) 
//     {

//         std::vector<std::pair<int, int> > continous_segment = effective_continous_segment_[i];
//         for(int j=0; j<continous_segment.size(); j++)
//         {
//             int sp = continous_segment[j].first + 5;
//             int ep = continous_segment[j].second - 5;

//             std::sort(cloud_smoothness_.begin()+sp, cloud_smoothness_.begin()+ep+1, by_value());

//             int mid_index = sp + (ep - sp) / 2;
//             float mid_smooth = cloud_smoothness_[mid_index].value;

//             int percent_index = sp + (1-candidate_edge_ratio_)*(ep - sp);
//             float percent_smooth = cloud_smoothness_[percent_index].value;

            
//             float smooth_th1 = std::min(percent_smooth, 5*mid_smooth);
//             float smooth_th2 = std::min(smooth_th1, min_smooth_th);

//             // std::cout << " smooth_th2: " <<  smooth_th2 << std::endl;

//             int k=ep;
//             while(cloud_smoothness_[k].value>smooth_th2)
//             {
//                 cloud_edge_smoothness_[cloud_smoothness_[k].ind] = cloud_smoothness_[k].value;
//                 k--;
//             }
//         }

//     }

//     // 存储候选边缘点云
//     pcl::PointXYZI thisPoint;
//     int check = 0;
//     int cloudSize = tight_point_cloud_->points.size();
//     for(int i=0; i<cloudSize; i++)
//     {
//         if(cloud_edge_smoothness_[i]>0.0)
//         {
//             if(cloud_edge_smoothness_[i]>0.05)
//             {
//                 thisPoint = tight_point_cloud_->points[i];
//                 edge_cand_point_cloud_->push_back(thisPoint);
//             }
//             else
//             {
//                 check++;
//             }
//         }
//     }

//     std::cout << "check: " << check << " edge_cand_point_cloud_->points.size(): " << edge_cand_point_cloud_->points.size() << std::endl;
// }

void CandidateSegment::generate_candidate_edge_()
{

    pcl::PointXYZI thisPoint;
    for (int i = 0; i < N_SCAN_; i++) 
    {

        std::vector<std::pair<int, int> > continous_segment = effective_continous_segment_[i];
        for(int j=0; j<continous_segment.size(); j++)
        {
            int sp = continous_segment[j].first + 5;
            int ep = continous_segment[j].second - 5;

            // 将间断点周围的光滑度计算有问题的点加入 edge_cand_point_cloud_
            for(int k=continous_segment[j].first; k<sp; k++)
            {
                thisPoint = tight_point_cloud_->points[k];
                edge_cand_point_cloud_->push_back(thisPoint);
            }

            for(int k=continous_segment[j].second; k>ep; k--)
            {
                thisPoint = tight_point_cloud_->points[k];
                edge_cand_point_cloud_->push_back(thisPoint);
            }

            std::sort(cloud_smoothness_.begin()+sp, cloud_smoothness_.begin()+ep+1, by_value());

            int mid_index = sp + (ep - sp) / 2;
            float mid_smooth = cloud_smoothness_[mid_index].value;

            int percent_index = sp + (1-candidate_edge_ratio_)*(ep - sp);
            float percent_smooth = cloud_smoothness_[percent_index].value;

            // std::cout << " 5*mid_smooth: " <<  5*mid_smooth << " percent_smooth: " << percent_smooth << std::endl;

            if(percent_smooth < 5*mid_smooth) // 更平滑的决定了更多的候选点
            {
                for(int k=percent_index; k<=ep; k++)
                {
                    cloud_edge_smoothness_[cloud_smoothness_[k].ind] = cloud_smoothness_[k].value;
                }
            }
            else
            {
                int k=ep;
                float smooth_th = 5*mid_smooth;
                while(cloud_smoothness_[k].value>smooth_th)
                {
                    cloud_edge_smoothness_[cloud_smoothness_[k].ind] = cloud_smoothness_[k].value;
                    k--;
                }
            }

        }

    }

    // 存储候选边缘点云，因为是大粗糙度，因此，要从平面候选中移除
    
    int check = 0;
    int cloudSize = tight_point_cloud_->points.size();
    for(int i=0; i<cloudSize; i++)
    {
        if(cloud_edge_smoothness_[i]>0.0)
        {
            if(cloud_edge_smoothness_[i]>0.1) // 这里移除的点，应该足够不可靠，否则，会导致平面也连不上
            {
                thisPoint = tight_point_cloud_->points[i];
                edge_cand_point_cloud_->push_back(thisPoint);
            }
            else
            {
                check++;
            }
        }
    }

    std::cout << "check: " << check << " edge_cand_point_cloud_->points.size(): " << edge_cand_point_cloud_->points.size() << std::endl;
    
}

void CandidateSegment::non_maximum_suppression() 
{
    for (int i = 0; i < N_SCAN_; i++) 
    {
        std::vector<std::pair<int, int> > continous_segment = effective_continous_segment_[i];
        for(int j=0; j<continous_segment.size(); j++)
        {
            int sp = continous_segment[j].first + 5;
            int ep = continous_segment[j].second - 5;

            float best_smoothess = -1.0; int best_id = -1; // 连续边缘中的最大粗糙度和对应的索引
            for(int k=sp; k<=ep; k++)
            {
                if(cloud_edge_smoothness_[k]>0.0) // 当前粗糙度大于零
			    {
                    if(best_smoothess>0.0) 
                    {
                        if(cloud_edge_smoothness_[k] > best_smoothess)
						{
							 best_smoothess = cloud_edge_smoothness_[k];
							 best_id = k;
							 cloud_edge_smoothness_[k-1] =-1.0; 
						}
						else 
						{
                        
							cloud_edge_smoothness_[k] =-1.0; 
							
						}
                    }
                    else // 没有上一个粗糙度，初始化。。
                    {
                        best_smoothess = cloud_edge_smoothness_[k];
				        best_id = k;
                    }
                }
                else
                {
                    best_smoothess = -1;
                    best_id = -1;
                }
            }

        }
    }

    // 直接将 effective_continous_segment_ 对应的非连续点存储为 cloud_edge_smoothness_[k]==1
    // 注意，首尾点是扫描线的起止， 因此不保留
    // for (int i = 0; i < N_SCAN_; i++) 
    // {
    //     std::vector<std::pair<int, int> > continous_segment = effective_continous_segment_[i];
    //     for(int j=0; j<continous_segment.size(); j++)
    //     {
    //         int sp = continous_segment[j].first;
    //         int ep = continous_segment[j].second;
    //         // 第一段的第一个点不保留
    //         if(j==0)
    //         {
    //             cloud_edge_smoothness_[ep] = 1.0;
    //             continue;
    //         }

    //         if(j==(continous_segment.size()-1))
    //         {
    //             cloud_edge_smoothness_[sp] = 1.0;
    //             continue;
    //         }

    //         cloud_edge_smoothness_[sp] = 1.0;
    //         cloud_edge_smoothness_[ep] = 1.0;
    //     }
    // }

}

void CandidateSegment::get_feature_range_image()
{

    pcl::PointXYZI thisPoint;
    int cloudSize = tight_point_cloud_->points.size();
    int index, wid_index, hei_index;
    for(int i=0; i<cloudSize; i++)
    {
        thisPoint = tight_point_cloud_->points[i];
        index = (int)thisPoint.intensity;
        wid_index = index % Horizon_SCAN_;
        hei_index = index / Horizon_SCAN_;
        if(cloud_edge_smoothness_[i]>0.1)
        {
            edge_feature_candidate_.at<float>(hei_index, wid_index) = 1.0;
            edge_point_cloud_->push_back(thisPoint);
        }
        else
        {
            plane_feature_candidate_.at<float>(hei_index, wid_index) = 1.0; // 初步的
            // plane_point_cloud_->push_back(thisPoint);
        }
    }

    std::cout << "edge_point_cloud_.size(): " << edge_point_cloud_->size() << std::endl;
    std::cout << "plane_point_cloud_.size(): " << plane_point_cloud_->size() << std::endl;

    // 使用 edge_cand_point_cloud_ + small_disjoint_point_cloud_ 将不可靠的平面点移除掉
    for(int i=0; i<edge_cand_point_cloud_->points.size(); i++)
    {
        thisPoint = edge_cand_point_cloud_->points[i];
        index = (int)thisPoint.intensity;
        wid_index = index % Horizon_SCAN_;
        hei_index = index / Horizon_SCAN_;

        plane_feature_candidate_.at<float>(hei_index, wid_index) = 0.0; // 初步的
    }

    // 基于筛选后的plane_feature_candidate_得到plane_point_cloud_点云
    // for(int i=0; i<small_disjoint_point_cloud_->points.size(); i++)
    // {
    //     thisPoint = small_disjoint_point_cloud_->points[i];
    //     index = (int)thisPoint.intensity;
    //     wid_index = index % Horizon_SCAN_;
    //     hei_index = index / Horizon_SCAN_;

    //     plane_feature_candidate_.at<float>(hei_index, wid_index) = 0.0; // 初步的
    // }

    for (size_t i = 0; i < Horizon_SCAN_; i++) {
        for (size_t j = 0; j < N_SCAN_; j++) {
            if(plane_feature_candidate_.at<float>(j, i)>0)
            {
                index = i  + j * Horizon_SCAN_;
                thisPoint = range_point_cloud_->points[index];
                plane_point_cloud_->push_back(thisPoint);
            }
        }
  }

}

void CandidateSegment::compute_cloud_smoothness_()
{
    int cloudSize = tight_point_cloud_->points.size();
    for (int i = 5; i < cloudSize - 5; i++) 
    {
        cloudCurvature_[i] = smooth(i);

        cloud_smoothness_[i].value = cloudCurvature_[i];
        cloud_smoothness_[i].ind = i;
    }
}

void CandidateSegment::setRangeImageProjParam(int N_SCAN, int Horizon_SCAN, float ang_res_y, float ang_bottom)
{
    N_SCAN_ = N_SCAN;
    Horizon_SCAN_ = Horizon_SCAN;
    ang_res_x_ = 360.0/float(Horizon_SCAN);
    ang_res_y_ = ang_res_y;
    ang_bottom_ = ang_bottom;

    // 初始化存储容器
    range_image_ = cv::Mat::zeros(N_SCAN, Horizon_SCAN, CV_32FC1);
    edge_feature_candidate_ = cv::Mat::zeros(N_SCAN, Horizon_SCAN, CV_32FC1);
    plane_feature_candidate_ = cv::Mat::zeros(N_SCAN, Horizon_SCAN, CV_32FC1);
    range_point_cloud_->points.resize(N_SCAN*Horizon_SCAN);

    scanStartInd_.resize(N_SCAN);
    scanEndInd_.resize(N_SCAN);
    disjoint_point_index_.resize(N_SCAN);
    effective_continous_segment_.resize(N_SCAN);

    cloud_smoothness_.resize(N_SCAN*Horizon_SCAN);
    cloud_edge_smoothness_.resize(N_SCAN*Horizon_SCAN);
    cloudCurvature_.resize(N_SCAN*Horizon_SCAN);

    initialized_ = true;

}

// LOAM
float CandidateSegment::smooth(int m)
{
    float diffX = tight_point_cloud_->points[m - 5].x + tight_point_cloud_->points[m - 4].x 
                + tight_point_cloud_->points[m - 3].x + tight_point_cloud_->points[m - 2].x 
                + tight_point_cloud_->points[m - 1].x - 10 * tight_point_cloud_->points[m].x 
                + tight_point_cloud_->points[m + 1].x + tight_point_cloud_->points[m + 2].x 
                + tight_point_cloud_->points[m + 3].x + tight_point_cloud_->points[m + 4].x 
                + tight_point_cloud_->points[m + 5].x;
    float diffY = tight_point_cloud_->points[m - 5].y + tight_point_cloud_->points[m - 4].y 
                + tight_point_cloud_->points[m - 3].y + tight_point_cloud_->points[m - 2].y 
                + tight_point_cloud_->points[m - 1].y - 10 * tight_point_cloud_->points[m].y 
                + tight_point_cloud_->points[m + 1].y + tight_point_cloud_->points[m + 2].y 
                + tight_point_cloud_->points[m + 3].y + tight_point_cloud_->points[m + 4].y 
                + tight_point_cloud_->points[m + 5].y;
    float diffZ = tight_point_cloud_->points[m - 5].z + tight_point_cloud_->points[m - 4].z 
                + tight_point_cloud_->points[m - 3].z + tight_point_cloud_->points[m - 2].z 
                + tight_point_cloud_->points[m - 1].z - 10 * tight_point_cloud_->points[m].z 
                + tight_point_cloud_->points[m + 1].z + tight_point_cloud_->points[m + 2].z 
                + tight_point_cloud_->points[m + 3].z + tight_point_cloud_->points[m + 4].z 
                + tight_point_cloud_->points[m + 5].z;
                
    return diffX * diffX + diffY * diffY + diffZ * diffZ;
}


// FEVO-LOAM: Feature Extraction and Vertical Optimized Lidar Odometry and Mapping
float CandidateSegment::continous_distance(int i)
{
    pcl::PointXYZI pA = tight_point_cloud_->points[i-1];
    pcl::PointXYZI pB = tight_point_cloud_->points[i];
    pcl::PointXYZI pC = tight_point_cloud_->points[i+1];

    float diffx_forward = (pA.x -pB.x);
    float diffy_forward = (pA.y -pB.y);
    float diffz_forward = (pA.z -pB.z);
    float diff_forward = std::sqrt(diffx_forward * diffx_forward + diffy_forward * diffy_forward + diffz_forward * diffz_forward);

    float diffx_backward = (pC.x -pB.x);
    float diffy_backward = (pC.y -pB.y);
    float diffz_backward = (pC.z -pB.z);
    float diff_backward = std::sqrt(diffx_backward * diffx_backward + diffy_backward * diffy_backward + diffz_backward * diffz_backward);

    float disjoint_param = std::max(diff_backward/diff_forward, diff_forward/diff_backward);

    // std::cout << "disjoint_param: " << disjoint_param << std::endl;

    if(std::max(diff_forward, diff_backward)>0.5)
        return disjoint_param;
    else
        return -1;

}