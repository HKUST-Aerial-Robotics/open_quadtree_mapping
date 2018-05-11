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

#include <quadmap/publisher.h>

#include <quadmap/seed_matrix.cuh>

#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <Eigen/Eigen>

quadmap::Publisher::Publisher(ros::NodeHandle &nh,
                          std::shared_ptr<quadmap::Depthmap> depthmap)
  : nh_(nh)
  , pc(new PointCloud)
{
  // for save image
  save_index = 0;
  std::cout << " initial the publisher ! " << std::endl;
  save_path = std::string("/home/wang/Desktop/test/pure_sgm/");

  depthmap_ = depthmap;
  colored_.create(depthmap->getHeight(), depthmap_->getWidth(), CV_8UC3);
  pub_pc = nh_.advertise<PointCloud>("pointcloud", 1);


  pub_color_depth = nh_.advertise<sensor_msgs::Image>("color_depth", 1);
  pub_depth       = nh_.advertise<sensor_msgs::Image>("depth", 1);
  pub_debug       = nh_.advertise<sensor_msgs::Image>("debug", 1);
  pub_reference   = nh_.advertise<sensor_msgs::Image>("reference", 1);;
}

void quadmap::Publisher::publishDebugmap(ros::Time msg_time)
{
  cv::Mat debug_mat;
  cv_bridge::CvImage cv_debug;
  debug_mat = depthmap_->getDebugmap();


  double min;
  double max;
  cv::minMaxIdx(debug_mat, &min, &max);
  max = 10;
  min = 0;
  cv::Mat adjMap;
  debug_mat.convertTo(adjMap, CV_8UC1, 255 / (max-min), -min);
  cv::Mat falseColorsMap;
  cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);

  cv::Mat reference_mat;
  reference_mat = depthmap_->getReferenceImage();
  cv::Mat black_image;
  cv::cvtColor(reference_mat, black_image, cv::COLOR_GRAY2BGR);
  // cv::addWeighted( black_image, 0.5, falseColorsMap, 0.5, 0.0, falseColorsMap);
  
  cv_debug.header.frame_id = "debug_mat";
  cv_debug.encoding = sensor_msgs::image_encodings::BGR8;
  cv_debug.image = falseColorsMap;
  cv_debug.header.stamp = msg_time;
  pub_debug.publish(cv_debug.toImageMsg());
}

void quadmap::Publisher::publishDepthmap(ros::Time msg_time)
{
  cv::Mat depthmap_mat;
  cv::Mat reference_mat;
  cv_bridge::CvImage cv_image, cv_image_colored, cv_image_reference;
  cv_image.header.frame_id = "depthmap";
  cv_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  depthmap_mat = depthmap_->getDepthmap();
  reference_mat = depthmap_->getReferenceImage();
  cv_image.image = depthmap_mat;

  cv_image_reference.header.frame_id = "reference_mat";
  cv_image_reference.encoding = sensor_msgs::image_encodings::MONO8;
  cv_image_reference.image = reference_mat;

  //color code the map
  double min;
  double max;
  cv::minMaxIdx(depthmap_mat, &min, &max);
  cv::Mat adjMap;
  std::cout << "depthimage min max of the depth: " << min << " , " << max << std::endl;
  min = 0.5; max = 10;
  // min = 0; max = 65;
  // min = 10; max = 30;
  depthmap_mat.convertTo(adjMap,CV_8UC1, 255/(max-min), -min);
  cv::Mat falseColorsMap;
  cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);
  cv::Mat mask;
  cv::inRange(falseColorsMap, cv::Scalar(0, 0, 255), cv::Scalar(0, 0, 255), mask);
  cv::Mat black_image = cv::Mat::zeros(falseColorsMap.size(), CV_8UC3);
  // cv::cvtColor(reference_mat, black_image, cv::COLOR_GRAY2BGR);
  // cv::addWeighted( black_image, 0.5, falseColorsMap, 0.5, 0.0, falseColorsMap);
  black_image.copyTo(falseColorsMap, mask);
  cv_image_colored.header.frame_id = "depthmap";
  cv_image_colored.encoding = sensor_msgs::image_encodings::BGR8;
  cv_image_colored.image = falseColorsMap;

  if(nh_.ok())
  {
    cv_image.header.stamp = msg_time;
    cv_image_colored.header.stamp = msg_time;
    pub_depth.publish(cv_image.toImageMsg());
    pub_color_depth.publish(cv_image_colored.toImageMsg());

    cv_image_reference.header.stamp = msg_time;
    pub_reference.publish(cv_image_reference.toImageMsg());
    std::cout << "INFO: publishing depth map" << std::endl;
  }
}

void quadmap::Publisher::publishPointCloud(ros::Time msg_time)
{
  {
    std::lock_guard<std::mutex> lock(depthmap_->getRefImgMutex());

    const cv::Mat depth = depthmap_->getDepthmap();
    // const cv::Mat depth = depthmap_->getDebugmap();

    const cv::Mat ref_img = depthmap_->getReferenceImage();
    const quadmap::SE3<float> T_world_ref = depthmap_->getT_world_ref();

    const float fx = depthmap_->getFx();
    const float fy = depthmap_->getFy();
    const float cx = depthmap_->getCx();
    const float cy = depthmap_->getCy();
    pc->clear();

    for(int y=0; y<depth.rows; ++y)
    {
      for(int x=0; x<depth.cols; ++x)
      {
        float depth_value = depth.at<float>(y, x);
        if(depth_value < 0.1)
          continue;
        const float3 f = normalize( make_float3((x-cx)/fx, (y-cy)/fy, 1.0f) );
        const float3 xyz = T_world_ref * ( f * depth_value );

        PointType p;
        p.x = xyz.x;
        p.y = xyz.y;
        p.z = xyz.z;
        const uint8_t intensity = ref_img.at<uint8_t>(y, x);
        p.intensity = intensity;
        pc->push_back(p);
      }
    }

    //  ///*for debug*/
    // for(int y=0; y<depth.rows; ++y)
    // {
    //   for(int x=0; x<depth.cols; ++x)
    //   {
    //     float depth_value = depth.at<float>(y, x);
    //     if(depth_value < 0.1)
    //       continue;
    //     const float3 f = make_float3((x-cx)/fx, (y-cy)/fy, 1.0f);
    //     const float3 xyz = T_world_ref * ( f * depth_value );

    //     PointType p;
    //     p.x = x / 100.0;
    //     p.y = depth_value;
    //     p.z = - y / 100.0;
    //     const uint8_t intensity = ref_img.at<uint8_t>(y, x);
    //     p.intensity = intensity;
    //     pc->push_back(p);
    //   }
    // }

  }
  if (!pc->empty())
  {
    if(nh_.ok())
    {
      pcl_conversions::toPCL(msg_time, pc->header.stamp);
      pc->header.frame_id = "/world";
      pub_pc.publish(pc);
    }
  }
}

void quadmap::Publisher::publishDepthmapAndPointCloud(ros::Time msg_time)
{
  publishDepthmap(msg_time);
  publishDebugmap(msg_time);
  publishPointCloud(msg_time);
}
