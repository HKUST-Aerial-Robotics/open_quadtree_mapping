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

#include <memory>
#include <Eigen/Eigen>
#include <opencv2/imgproc/imgproc.hpp>

#include <quadmap/seed_matrix.cuh>
#include <quadmap/se3.cuh>
#include <mutex>
#include <iostream>
namespace quadmap
{

class Depthmap
{
public:
  Depthmap(
      size_t width,
      size_t height,
      float fx,
      float cx,
      float fy,
      float cy,
      cv::Mat remap_1,
      cv::Mat remap_2,
      int semi2dense_ratio
      );

  bool add_frames(  const cv::Mat &img_curr,
                    const SE3<float> &T_curr_world);

  const cv::Mat_<float> getDepthmap() const;
  const cv::Mat_<float> getDebugmap() const;
  const cv::Mat getReferenceImage() const;

  float getFx() const
  { return fx_; }

  float getFy() const
  { return fy_; }

  float getCx() const
  { return cx_; }

  float getCy() const
  { return cy_; }

  std::mutex & getRefImgMutex()
  { return ref_img_mutex_; }

  size_t getWidth() const
  { return width_; }

  size_t getHeight() const
  { return height_; }
  
  SE3<float> getT_world_ref() const
  { return T_world_ref; }

private:
  SeedMatrix seeds_;
  size_t width_;
  size_t height_;
  float fx_, fy_, cx_, cy_;

  std::mutex ref_img_mutex_;

  SE3<float> T_world_ref;
  cv::Mat depth_out;
  cv::Mat reference_out;
  cv::Mat debug_out;
};

}