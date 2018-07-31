SET(PROJECT_NAME open_quadtree_mapping)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)
PROJECT(${PROJECT_NAME})
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")
SET(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo

find_package(CUDA REQUIRED)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -use_fast_math)

# Specify compute capability
list(APPEND CUDA_NVCC_FLAGS -arch=sm_61)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include(/home/wang/software/opencv/opencv-3.2.0.bin/OpenCVConfig.cmake)
find_package(Eigen REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)

find_package(catkin REQUIRED COMPONENTS
  roscpp roslib cmake_modules cv_bridge image_transport pcl_ros sensor_msgs geometry_msgs)
catkin_package(
    DEPENDS OpenCV Eigen Boost
    CATKIN_DEPENDS roscpp roslib image_transport pcl_ros
    INCLUDE_DIRS include
    #LIBRARIES efficient_map efficient_map_cuda
)


include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

# Set link libraries
list(APPEND LINK_LIBS
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
)

include_directories(${catkin_INCLUDE_DIRS})
list(APPEND LINK_LIBS ${catkin_LIBRARIES})

cuda_add_executable(open_quadtree_mapping_node
    src/check_cuda_device.cu
    src/seed_matrix.cu
    src/depthmap.cpp
    src/depthmap_node.cpp
    src/publisher.cpp
    src/main_ros.cpp
    )
target_link_libraries(open_quadtree_mapping_node
${catkin_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
