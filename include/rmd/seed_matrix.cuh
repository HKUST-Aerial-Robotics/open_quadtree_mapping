// This file is part of REMODE - REgularized MOnocular Depth Estimation.
//
// Copyright (C) 2014 Matia Pizzoli <matia dot pizzoli at gmail dot com>
// Robotics and Perception Group, University of Zurich, Switzerland
// http://rpg.ifi.uzh.ch
//
// REMODE is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// REMODE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <iostream>
// #include <future>
#include <quadmap/device_image.cuh>
#include <quadmap/camera_model/pinhole_camera.cuh>
#include <quadmap/se3.cuh>
#include <quadmap/match_parameter.cuh>
#include <quadmap/pixel_cost.cuh>
#include <quadmap/frameelement.cuh>
#include <quadmap/DepthSeed.cuh>

#include <opencv2/opencv.hpp>
// #include <opencv2/gpu/gpu.hpp>//for opencv2
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

namespace quadmap
{
class SeedMatrix
{
public:
  SeedMatrix(
      const size_t &width,
      const size_t &height,
      const PinholeCamera &cam);
  ~SeedMatrix();

  void set_remap(cv::Mat _remap_1, cv::Mat _remap_2);
  void set_semi2dense_ratio(int _semi2dense_ratio){semi2dense_ratio = _semi2dense_ratio;}
  bool input_raw(
    cv::Mat raw_mat,
    const SE3<float> T_curr_world
  );

  void get_result(cv::Mat &depth, cv::Mat &debug, cv::Mat &reference);
private:
  bool add_frames(
    cv::cuda::GpuMat &input_image,
    const SE3<float> T_curr_world);
  void add_income_image(
    const cv::cuda::GpuMat &input_image,
    const SE3<float> T_world);
  void download_output();

  //for semidebse
  void set_income_as_keyframe();
  void create_new_keyframe();
  void create_new_keyframe_async();
  void initial_keyframe();
  void update_keyframe();
  void project_to_current_frame();
  bool need_switchkeyframe();

  //for full dense 
  void extract_depth();
  bool need_add_reference();
  void add_reference();

  //for depth fusion
  void fuse_output_depth();

  cudaStream_t swict_semidense_stream1;
  cudaStream_t swict_semidense_stream2;
  cudaStream_t swict_semidense_stream3;
  size_t width;
  size_t height;
  int frame_index;
  int semi2dense_ratio;
  bool initialized;
  MatchParameter match_parameter;

  DeviceImage<float> depth_output;
  DeviceImage<float> debug_image;

  //income image
  DeviceImage<float> pre_income_image;
  DeviceImage<float> income_image;
  DeviceImage<float2> income_gradient;
  SE3<float> income_transform;
  cv::Mat income_undistort;

  //semidense
  DeviceImage<DepthSeed> keyframe_semidense;
  DepthSeed* semidense_hostptr;
  DepthSeed* semidense_new_hostptr;
  float2 *income_gradient_hostptr;
  DeviceImage<float> keyframe_image;
  DeviceImage<float2> keyframe_gradient;
  DeviceImage<float2> semidense_on_income;
  SE3<float> keyframe_transform;
  cv::Mat keyframeMat;

  //for depth extraction
  DeviceImage<int> pixel_age_table;
  DeviceImage<float4> depth_fuse_seeds;
  SE3<float> this_fuse_worldpose;

  int current_frames;
  std::deque<FrameElement> framelist_host;

  //used for gpu remap
  cv::cuda::GpuMat remap_1, remap_2;
  cv::cuda::GpuMat input_image;
  cv::cuda::GpuMat input_float;
  cv::cuda::GpuMat undistorted_image;

  //result
  cv::Mat cv_output;
  cv::Mat cv_debug;

  //camera model
  PinholeCamera camera;
};

}