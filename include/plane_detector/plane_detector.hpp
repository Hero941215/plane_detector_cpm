#ifndef PLANE_DETECTOR__
#define PLANE_DETECTOR__

#include <vector>
#include <random>
#include <queue>
#include "detected_plane.hpp"
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Eigenvalues>
// #include <boost/array.hpp>
// #include <boost/thread.hpp>
#include <exception>
#include <opencv2/opencv.hpp>
#include <iostream>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

#define __MIN_RANGE__ 0.3
#define __MAX_RANGE__ 200.0

struct UnknownDepthException:public std::exception {
  const char *what() const throw()
  {
    return "Unknown Depth Exception";
  }
  
};

enum PixelStatus {
  UNPROCESSED = -3, IN_R_PRIMA, IN_QUEUE
};

//! @class PlaneDetector
//! @brief Implements a fast plane detector on RGB-D images as presented in Poppinga IROS 2008
class PlaneDetector {
public:
  //! Recommended constructor
  PlaneDetector(double delta = 1.0, double epsilon = 1.0, double gamma = 10.0, int theta = 1000, double _std_dev = 0.03);
  
  //! @brief Detects the planes inside the image
  int detectPlanes(const cv::Mat &depth);

  void RegionGrow();

  void MergeWithCP();
  
  inline std::vector<DetectedPlane> getPlanes() const {
    return _detected_planes;
  }

  inline std::vector<std::vector<int> > getMergedPlanes() const {
    return _final_merge_plane_group;
  }

  inline std::vector<std::vector<int> > getRegions() const
  {
    return _detected_planes_region;
  }

  void resetStatus();
  
  inline int getPos(int i, int j) const {
    return i + j * _width;
  }
  
  pcl::PointXYZI get3DPoint2(int index);
  
  void setRawPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pcPtr);

  inline bool isInitialized() {return _initialized;}
  
  inline void setFloatImage(bool new_value) {_float_image = new_value;}
  
protected:
  // Parameters from the paper Poppinga: 
  double _delta; // Maximum neighbor distance (0.2 is OK)
  double _epsilon; // Maximum MSE (1e-4)
  double _gamma;  // Maximum distance to plane
  int _theta; // Minimum number of points of the segmented plane

  std::vector<std::vector<int> > _detected_planes_region;
  std::vector <DetectedPlane> _detected_planes; // Vector de IDs
  std::vector <int> _detected_ids; // Contiene todos los IDs que son considerados planos

  std::vector<std::vector<int> > _final_merge_plane_group; // 经过cp搜索和法向量检测的合并面id


  std::vector<int> _status_vec; // Relaciona los puntos de la imagen con una region
  std::queue<int> _q;
  // std::priority_queue <int, std::vector<int>, std::less<int> > _q;
  int _available_pixels;
  bool _initialized;
  double _std_dev;
  cv::Mat _image;

  std::vector<int> _curr_region; // Saves the region in coordinates --> i + j * _height
  DetectedPlane _curr_plane;
  
  // Internal stuff for update Matrices
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> _es;
  DetectedPlane _p;

  int _curr_region_id;
  
  // Attributes of the image
  int _width, _height;
  bool _float_image;
  
  std::vector<Eigen::Vector3d> _color;
  
  // 直接载入3d点
  pcl::PointCloud<pcl::PointXYZI>::Ptr _fullCloud; 

  int getRandomPixel(int region = (int)UNPROCESSED) const;
  //! @brief Gets the unprocessed nearest neighbor of the image
  int getNearestNeighbor(int index) const;
  //! @brief Gets the amount of available pixels in image
  int availablePixels() const;
  
  // For random numbers
  static std::default_random_engine generator;
  
  //! @brief Adds pixels to region and adds its nearest neighbors to the queue (modifying status_vec)
  void addPixelToRegion(int index, int _curr_region_id);
  
  // Eigen::Vector3d get3DPoint(int i, int j) const;

  Eigen::Vector3d get3DPoint(int index) const;
  //! @brief Gets the depth of a depth image
  //! @return THe depth value or -1.0 if the data is not valid
  double getDepth(int i, int j) const;
  
  //! @brief updates s_g, m_k, p_k and MSE
  //! @retval true The conditions of step 8 of the matrix (see [1]) are met --> the matrices are updated
  bool updateMatrices(const Eigen::Vector3d &v);

  bool checkMergePlane(DetectedPlane dp1, DetectedPlane dp_cur);
  
};

std::default_random_engine PlaneDetector::generator;

PlaneDetector::PlaneDetector(double delta, double epsilon, double gamma, int theta, double _std_dev):
_delta(delta), _epsilon(epsilon), _gamma(gamma), _theta(theta),_initialized(false),_std_dev(_std_dev),
_float_image(1)
{
  _fullCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
}
  
int PlaneDetector::detectPlanes(const cv::Mat& image)
{
  // 设置相机内参
  if (!_initialized) {
    // The camera parameters have to be set before detecting planes
    return 0;
  }
  
  // First: get parameters of the image (MUST BE AT THE FIRST)
  _width = image.cols;
  _height = image.rows;
  this->_image = image; // TODO: será lento?
  
  // All internal status to zero or proper values
  // 重置状态变量， 特别是 _status_vec
  resetStatus();
  // _q.empty();
  
  RegionGrow();

  MergeWithCP();
  
  return _detected_planes.size();
}

// 检测法向量和偏置
bool PlaneDetector::checkMergePlane(DetectedPlane dp1, DetectedPlane dp_cur)
{
    double th = 4;

    // 计算cp 
    double norm = dp1.v.norm();
    Eigen::Vector3d correct_v = dp1.v / norm;
    double correct_d = dp1.d / norm;
    Eigen::Vector3d cp1 = -correct_v*correct_d;
    
    double norm2 = dp_cur.v.norm();
    Eigen::Vector3d correct_v2 = dp_cur.v / norm2;
    double correct_d2 = dp_cur.d / norm2;
    Eigen::Vector3d cp_cur = -correct_v2*correct_d2;

    double cosValNew = cp1.dot(cp_cur) /(cp1.norm()*cp_cur.norm()); //角度cos值
    double angleNew = acos(cosValNew) * 180 / M_PI;     //弧度角

    // std::cout << "angleNew: " << angleNew << std::endl;

    if(angleNew<(0+th) || angleNew>(180-th))
    {
        // 进一步检测偏置
        if(angleNew<(0+th))
        {
            if(std::fabs(correct_d2-correct_d)<0.15)
                return true;
            else
                return false;
        }
        else
        {
            if(std::fabs(correct_d2+correct_d)<0.15)
                return true;
            else
                return false;   
        }

    }
    else
        return false;
}

void PlaneDetector::MergeWithCP()
{

    _final_merge_plane_group.clear();

    // 基于最近点的检测平面容器和kd-tree容器 - pcl-ds
    pcl::PointCloud<pcl::PointXYZI>::Ptr cp_detect_plane(
      new pcl::PointCloud<pcl::PointXYZI>);
    
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreePlane(new pcl::KdTreeFLANN<pcl::PointXYZI>);

    pcl::PointXYZI cp_plane;
    for(int i=0; i<_detected_planes.size(); i++)
    {
        // 计算面特征法向量模长
        double norm = _detected_planes[i].v.norm();
        Eigen::Vector3d correct_v = _detected_planes[i].v / norm;
        double correct_d = _detected_planes[i].d / norm;
        Eigen::Vector3d cp = -correct_v*correct_d;
        cp_plane.x = cp[0];
        cp_plane.y = cp[1];
        cp_plane.z = cp[2];
        cp_plane.intensity = i;
        cp_detect_plane->push_back(cp_plane);

        // std::cout << "cp: " << cp.transpose() << std::endl;
    }

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    kdtreePlane->setInputCloud(cp_detect_plane);
    std::vector<std::pair<bool,int> > plane_check(_detected_planes.size(), std::make_pair(false, -1)); // 是否已经被合并过， 对应的group id, 从 0 开始
    std::vector<std::vector<int> > merge_plane_group; // 可以合并的平面的id / 也包含独立平面
    for(int i=0; i<plane_check.size(); i++)
    {
        // 如果平面尚未被检测或者合并
        if(plane_check[i].first==false)
        {

            cp_plane = cp_detect_plane->points[i];
            kdtreePlane->radiusSearch(cp_plane, 0.2, pointIdxRadiusSearch, pointRadiusSquaredDistance); // 会包含输入kd-tree的最近点

            std::set<int> merged_plane_group_id;
            // 检查当前平面是否可以与已合并平面继续合并， 并且统计可以与那些合并平面合并
            for(int j=0; j<pointIdxRadiusSearch.size(); j++)
            {
                if(plane_check[pointIdxRadiusSearch[j]].second>=0)
                {
                    merged_plane_group_id.insert(plane_check[pointIdxRadiusSearch[j]].second);
                }
            }
            
            if(merged_plane_group_id.empty()) // 创建一个新的合并平面
            {
                std::vector<int> new_plane; 
                for(int j=0; j<pointIdxRadiusSearch.size(); j++)
                {
                    cp_plane = cp_detect_plane->points[pointIdxRadiusSearch[j]];
                    new_plane.push_back((int)cp_plane.intensity );
                    plane_check[(int)cp_plane.intensity ].first = true;
                    plane_check[(int)cp_plane.intensity ].second = merge_plane_group.size();
                }
                merge_plane_group.push_back(new_plane);
            }
            else // merged_plane_group_id非空， 则发现新的检测平面需要合并到已有的平面中
            {

                std::vector<int> merged_plane_group_id_;
                merged_plane_group_id_.assign(merged_plane_group_id.begin(), merged_plane_group_id.end());

                if(merged_plane_group_id.size()>1) 
                {
                    sort(merged_plane_group_id_.begin(),merged_plane_group_id_.end());   

                    // 遍历merged_plane_group_id_进行合并操作
                    for(int j=merged_plane_group_id_.size()-1; j>0; j--)
                    {
                        // 合并到最老平面
                        merge_plane_group[merged_plane_group_id_[0]].insert(merge_plane_group[merged_plane_group_id_[0]].end(),
                                      merge_plane_group[merged_plane_group_id_[j]].begin(), merge_plane_group[merged_plane_group_id_[j]].end());

                        std::vector<int> merge_plane = merge_plane_group[merged_plane_group_id_[j]];
                        for(int k=0; k<merge_plane.size(); k++)
                        {
                            plane_check[merge_plane[k]].second = merged_plane_group_id_[0];
                        }

                        // 把他后面的合并组，按顺序提前一位，同时， 修正对应的plane_check
                        for(int k=merged_plane_group_id_[j]; k<merge_plane_group.size()-1; k++)
                        {
                            std::vector<int> merge_plane2 = merge_plane_group[k+1];
                            merge_plane_group[k] = merge_plane2;
                            for(int l=0; l<merge_plane2.size(); l++)
                            {
                                plane_check[merge_plane2[l]].second -= 1;
                            }
                        }
                        merge_plane_group.resize(merge_plane_group.size()-1);
                    }

                    // 将pointIdxRadiusSearch中的其他平面添加到最老的merged_plane_group中
                    for(int j=0; j<pointIdxRadiusSearch.size(); j++)
                    {
                        if(plane_check[pointIdxRadiusSearch[j]].second<0)
                        {
                            cp_plane = cp_detect_plane->points[pointIdxRadiusSearch[j]];
                            merge_plane_group[merged_plane_group_id_[0]].push_back((int)cp_plane.intensity );
                            plane_check[(int)cp_plane.intensity ].first = true;
                            plane_check[(int)cp_plane.intensity ].second = merged_plane_group_id_[0];
                        }
                    }
                   
                }
                else // 只有一个已合并平面
                {
                    for(int j=0; j<pointIdxRadiusSearch.size(); j++)
                    {
                        if(plane_check[pointIdxRadiusSearch[j]].second<0)
                        {
                            cp_plane = cp_detect_plane->points[pointIdxRadiusSearch[j]];
                            merge_plane_group[merged_plane_group_id_[0]].push_back((int)cp_plane.intensity );
                            plane_check[(int)cp_plane.intensity ].first = true;
                            plane_check[(int)cp_plane.intensity ].second = merged_plane_group_id_[0];
                        }

                    }
                }       
            }


        }
    }

    for(int i=0; i<merge_plane_group.size(); i++)
    {
        if(merge_plane_group[i].size()==1) // 对应于绝大多数情况, 只有一条线
        {
            _final_merge_plane_group.push_back(merge_plane_group[i]); 
        }
        else
        {
            std::vector<int> final_merge_plane;
            while(merge_plane_group[i].size()!=0)
            {

                // std::cout << "merge_plane_group[i].size(): " << merge_plane_group[i].size() << std::endl;
                final_merge_plane.clear();

                // 提取当前容器中的第一个面的参数
                int detect_plane_fir_id = merge_plane_group[i][0];
                DetectedPlane dp1 = _detected_planes[detect_plane_fir_id];

                final_merge_plane.push_back(detect_plane_fir_id);

                if(merge_plane_group[i].size()>1)
                {
                    int save = 0;
                    int cur_size = merge_plane_group[i].size();
                    // 和所有后面的线进行比较，如果足够近则放入则放入final_merge_plane， 并且从merge_plane_group中剔除掉 
                    for(int j=1; j<cur_size; j++)
                    {

                        int detect_plane_cur_id = merge_plane_group[i][j];
                        DetectedPlane dp = _detected_planes[detect_plane_cur_id];
                        if(checkMergePlane(dp1, dp))
                        {
                            // std::cout << "real merge! " << std::endl;
                            final_merge_plane.push_back(detect_plane_cur_id);
                            continue;
                        }

                        if(j==save)
                            save++;
                        else
                        {
                            int tmp = merge_plane_group[i][j];
                            merge_plane_group[i][j] = merge_plane_group[i][save];
                            merge_plane_group[i][save] = tmp;
                            save++;
                        } 
                    }
                    merge_plane_group[i].resize(save);
                }
                else // 
                {
                    merge_plane_group[i].resize(0);
                }

                _final_merge_plane_group.push_back(final_merge_plane);
                
            }
        }
    }

    std::cout << "_final_merge_plane_group.size(): " << _final_merge_plane_group.size() << std::endl;


}

void PlaneDetector::RegionGrow()
{
    // int _curr_region_id = 0;
    
    // 统计有深度的像素块数量
    // _available_pixels = availablePixels();

    // 搜索平面
    while (_available_pixels > _theta * 1.1) 
    {
      //     std::cout << "Available pixels: " << _available_pixels << std::endl;
      _curr_region.clear();
      // Initialization of the algorithm--> random point and nearest neighbor

      // 随机找到一个没有查询过的像素点
      int candidate = getRandomPixel();

      // 找到 window of size 9 和当前点最接近的一个点
      int nearest = getNearestNeighbor(candidate);
      
      if (nearest > 0) 
      {
        _status_vec[nearest] = _curr_region_id;

        // 通过candidate和nearest扩充最初的点集合
        addPixelToRegion(candidate, _curr_region_id);
        addPixelToRegion(nearest, _curr_region_id);

        // Initialize matrices and vectors
        Eigen::Vector3d r_1 = get3DPoint(candidate);
        Eigen::Vector3d r_2 = get3DPoint(nearest);
        _curr_plane.s_g = r_1 + r_2;
        _curr_plane.m_k = (r_1 - _curr_plane.s_g * 0.5)*(r_1 - _curr_plane.s_g * 0.5).transpose();
        _curr_plane.m_k += (r_2 - _curr_plane.s_g * 0.5)*(r_2 - _curr_plane.s_g * 0.5).transpose();
        _curr_plane.p_k = r_1 * r_1.transpose() + r_2*r_2.transpose();
        _curr_plane.n_points = 2;
        
        while (!_q.empty()) // 遍历平面搜索队列
        {
          int new_point = _q.front();
          // int new_point = _q.top();
          _q.pop();
          Eigen::Vector3d v = get3DPoint(new_point);
          if (updateMatrices(v)) { // 判断当前点是否属于平面
            addPixelToRegion(new_point, _curr_region_id);
          } else {
            _available_pixels--;
          }
        }
        
        // The queue has been emptied --> clear possible QUEUE status and add the region to the detected planes if condition of step 12 (Algorithm 1)
        if (_curr_region.size() > _theta) {

           int point_num = _curr_region.size();
          _es.compute(_curr_plane.m_k/(point_num-1));
          Eigen::Vector3d eigenvalue = _es.eigenvalues();
          double ratio = eigenvalue.x() / eigenvalue.y();

          if(ratio<0.1)
          {
            _curr_plane.makeDPositive();
            // _curr_plane.calculateCovariance(_std_dev); // 计算协方差其实没有啥用处
    //         std::cout << "Detected plane: " << _curr_plane.toString() << std::endl;
            _detected_planes.push_back(_curr_plane);
            _detected_ids.push_back(_curr_region_id);
            _detected_planes_region.push_back(_curr_region);

            _es.compute(_curr_plane.m_k);

          // Eigen::Vector3d eigenvalue = _es.eigenvalues();

          // std::cout << "eigen: " << eigenvalue.transpose() << std::endl << std::endl;
          }

        }
        _curr_region_id++; // The next plane will have a new identificator
      } else {
        // No nearest neighbor available --> discard (to R_PRIMA)
        _status_vec[candidate] = (int)IN_R_PRIMA;
        _available_pixels--;
      }
    }
}

void PlaneDetector::setRawPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pcPtr)
{
    _fullCloud = pcPtr;
    _initialized = true;
}

void PlaneDetector::resetStatus()
{
  if (_status_vec.size() != _width*_height) {
    _status_vec.resize(_width*_height);
  }
 
  for (size_t i = 0; i < _width; i++) {
    for (size_t j = 0; j < _height; j++) {
      if (getDepth(i, j) > 0.0) { // 有效深度
        _status_vec[getPos(i,j)] = (int)UNPROCESSED; // -3
      } else {
        _status_vec[getPos(i,j)] = (int)IN_R_PRIMA;  // -2 
      }
    }
  }
  
  _p.init();
  _curr_plane.init();
  
  _detected_planes_region.clear();
  _detected_planes.clear();
  _detected_ids.clear();

  _final_merge_plane_group.clear();

  _curr_region_id = 0;
   
  _available_pixels = availablePixels();
}

int PlaneDetector::getRandomPixel(int region) const
{
  std::uniform_int_distribution<int> distribution(0, _height * _width - 1);
  int ret_val;
  
  do {
    ret_val = distribution(generator);
  } while (_status_vec[ret_val] != region);
  
  return ret_val;
}

//! Gets the unprocessed nearest neighbor of a pixel in a window of size 9
int PlaneDetector::getNearestNeighbor(int index) const
{
  int near = -1;
  int aux;
  float min_dist = 1e10;
  
  // 基于1维索引获得对应雷达坐标系下的3d点
  // 从3*3改成5*5？
  // 3*3 对应 -1 -> 2
  // 5*5 对应 -2 -> 3
  Eigen::Vector3d v = get3DPoint(index);
  Eigen::Vector3d v_;
  for (int i = -2; i < 3; i++) {
    for (int j = -2; j < 3; j++) {
      if (i != 0 || j != 0) {
        aux = index + i + j * _width;
        
        if (aux > 0 && aux < _height * _width) { // Check bounds
          if (_status_vec[aux] == (int)UNPROCESSED) {
            v_ = get3DPoint(aux);
            double dist = (v - v_).norm();
            if (dist < min_dist) {
              near = aux;
              min_dist = dist;
            }
          }
        }
      }
    }
  }
  return near;
}

// 计算所有的可用深度数量
int PlaneDetector::availablePixels() const
{
  bool ret_val = false;
  int cont = 0;
  
  //TODO: Uncomment
  for (int i = 0; i < _width * _height /*&& cont < _theta*/; i++) {
    if (_status_vec[i] == (int)UNPROCESSED) 
      cont++;
  }
  
  return cont;
}

// 基于2d图像索引获取对应的深度值
double PlaneDetector::getDepth(int i, int j) const 
{
  double ret_val = -1.0;
  
  if (i >= _image.cols || j >= _image.rows)
    return -1.0;
  // std::cout << "Data: " << cvbDepth->image.at<float>(i,j) << " ";
  if(_float_image)
  {
    ret_val = _image.at<float>(j, i); /// CAUTION!!!! the indices in cv::Mat are row, cols (elsewhere cols, rows)
  } 
  else
  {
    ret_val = _image.at<u_int16_t>(j, i)*0.001;
  }
  if(ret_val < __MIN_RANGE__ || ret_val > __MAX_RANGE__)
  {
    ret_val = -1.0;
  }
  return ret_val;
}

// 直接从点云容器中获得index
Eigen::Vector3d PlaneDetector::get3DPoint(int index) const
{
  pcl::PointXYZI p_pcl = _fullCloud->points[index];
  Eigen::Vector3d p_eigen(p_pcl.x, p_pcl.y, p_pcl.z);
  return p_eigen;
}

pcl::PointXYZI PlaneDetector::get3DPoint2(int index)
{
   return _fullCloud->points[index];

}
  
void PlaneDetector::addPixelToRegion(int index, int _curr_region_id)
{
  // 平面搜索容器_curr_region中添加索引
  _status_vec[index] = _curr_region_id;
  _curr_region.push_back(index);
  Eigen::Vector3d v = get3DPoint(index);
  _available_pixels--;
  
  // 以当前点为中心，看看还有没有新的点可以插入
  int neighbor = getNearestNeighbor(index);
  while (neighbor >= 0) {
    Eigen::Vector3d v_2 = get3DPoint(neighbor);
    if ( (v-v_2).norm() < _delta) { // First check --> the neighbor is sufficiently near to the point
      _status_vec[neighbor] = (int)IN_QUEUE;
      _q.push(neighbor);
      neighbor = getNearestNeighbor(index);
    } else {
      neighbor = -1;
    }
  }
}

bool PlaneDetector::updateMatrices(const Eigen::Vector3d& v)
{
  size_t k = _curr_region.size();
  double div_1 = 1.0 / (double)(k + 1);
  double div = 1.0 / (double)(k);

  // 临时更新
  // 所有点的和
  _p.s_g = _curr_plane.s_g + v;

  // 更新协方差矩阵
  _p.m_k = _curr_plane.m_k + v*v.transpose() - (_p.s_g * div_1)*_p.s_g.transpose() + (_curr_plane.s_g * div) * _curr_plane.s_g.transpose();

  // p*pT的累积
  _p.p_k = _curr_plane.p_k + v*v.transpose();
  
  // Calculate d and n (n is the eigenvector related to the lowest eigenvalue)
  _es.compute(_p.m_k*div);
  double min_eigenval = 1e10;
  int min_i; 
  for (int i = 0; i < 3; i++) {
    double curr_eigen = fabs(_es.eigenvalues()(i));

    // std::cout << "_es.eigenvalues()(i)" << _es.eigenvalues()(i) << std::endl;

    if (curr_eigen < min_eigenval) {
      min_eigenval = curr_eigen;
      min_i = i;
    }
    
  }

  _p.v = _es.eigenvectors().col(min_i);
  _p.v  = _p.v / _p.v.norm();

  _p.d = div_1 * _p.v.dot(_p.s_g); // 计算出来的第四维度平面参数， 后面都用 -d
  
  // std::cout << "_p.d: " << _p.d << std::endl;

  // Update the MSE (Eq 3)
  _p.mse = div_1 * _p.v.transpose() * _p.p_k * _p.v - 2 * _p.v.dot(_p.s_g) * _p.d * div_1 + _p.d * _p.d;
  _p.n_points = k + 1;
  
  
  // Check if the new plane meets the constraint of algorithm 1 step 8. If so, update the values of the matrices of the class
  if (_p.mse < _epsilon && _p.distance(v) < _gamma) 
  {
    // Meets the constraints --> Actualize the plane
    _p.r_g = _p.s_g * div_1;
    _curr_plane = _p;
    
    return true;
  }
  return false;
}

#endif
