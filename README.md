# plane_detector_cpm

**Implementation of the fast plane detector as described in the paper "Fast  Plane  Detection  and  Polygonalization  in  noisy  3D  Range  Images", of Poppinga et al. (IROS 2008)  https://ieeexplore.ieee.org/abstract/document/4650729**

## 0. Features
Extract a set of structural plane points from range image using region growing algorithm. In order to address common occlusion problem, this algorithm uses a planar set based on the closest point representation to construct a kd-tree, and merges adjacent planar point sets that are less than the distance threshold.

## 1. Prerequisites
### 1.1 **Ubuntu**
**Ubuntu >= 18.04**

ROS    >= Melodic. [ROS Installation](http://wiki.ros.org/ROS/Installation)

### 1.2. **PCL && Eigen**
PCL    >= 1.8,   Follow [PCL Installation](http://www.pointclouds.org/downloads/linux.html).

## 2. Build

Clone the repository and catkin_make:

```
    cd ~/$A_ROS_DIR$/src
    git clone https://github.com/Hero941215/lego-loam_ndt
    cd lego-loam_ndt
    git submodule update --init
    cd ../..
    catkin_make
    source devel/setup.bash
```

## 3. Run
### 3.1. **run lego-loam**

roslaunch lego_loam run.launch

### 3.2. **run ndt_localizer**

roslaunch ndt_localizer ndt_localizer.launch

#### Config map loader
Move your map pcd file (.pcd) to the map folder inside this project (`ndt_localizer/map`), change the pcd_path in `map_loader.launch` to you pcd path, for example:

```xml
<arg name="pcd_path"  default="$(find ndt_localizer)/map/kaist02.pcd"/>
```

## 4. Acknowledgments

Thanks for [LeGo-LOAM](https://github.com/RobustFieldAutonomyLab/LeGO-LOAM), [ndt_localizer](https://github.com/AbangLZU/ndt_localizer).


