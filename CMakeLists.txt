cmake_minimum_required(VERSION 2.8.3)
project(plane_detector)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)
set(CMAKE_CXX_FLAGS "-std=c++17 -Wfatal-errors")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

add_definitions(-DBOOST_ERROR_CODE_HEADER_ONLY) 
 
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  compressed_depth_image_transport
  cv_bridge
  dynamic_reconfigure
  image_transport
  message_filters
  rosbag
  roscpp
  rospy
  sensor_msgs
  tf
  visualization_msgs
)

#find_package(Eigen3 REQUIRED)
set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
find_package(OpenCV 3.4.1 REQUIRED)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED)
find_package(PCL REQUIRED QUIET)

find_package(OpenMP REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

SET(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_BUILD_TYPE Release)

#Dynamic reconfigure stuff
# generate_dynamic_reconfigure_options(cfg/Detector.cfg)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS compressed_depth_image_transport dynamic_reconfigure  image_transport message_filters message_runtime rosbag   roscpp   rospy   sensor_msgs   std_msgs   tf visualization_msgs
#  LIBRARIES ${PROJECT_NAME}
  DEPENDS OpenCV PCL
)

include_directories(
  ${catkin_INCLUDE_DIRS}, 
  ${EIGEN3_INCLUDE_DIR} , 
  ${OpenCV_INCLUDE_DIRS},
  ${PCL_INCLUDE_DIRS}, 
 ${Boost_INCLUDE_DIRS}, include
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(
# include
#  ${catkin_INCLUDE_DIRS}
# )


# 候选线段和平面分割模块
add_library(candidate_segment SHARED 
src/candidate_segment/candidate_segment.cpp)

# 不依赖ros的独立运行模块， 下面的平面检测模块. 检测面 和 合并面
add_executable(test_plane_detector_scan_merge2 src/test_plane_detector_scan_merge2.cpp)
target_link_libraries(test_plane_detector_scan_merge2 ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${Boost_LIBRARIES} candidate_segment)


## Install stuff
install(DIRECTORY include/${PROJECT_NAME}
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.hpp"
  PATTERN ".svn" EXCLUDE
)

