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

#include <vector_types.h>
#include <cuda_toolkit/helper_math.h>

namespace quadmap
{
  
struct PinholeCamera
{
  __host__
  PinholeCamera()
    : fx(0.0f), fy(0.0f), cx(0.0f), cy(0.0f)
  {
  }

  __host__ __device__
  PinholeCamera(const PinholeCamera& camera_ref)
  :fx(camera_ref.fx), fy(camera_ref.fy), cx(camera_ref.cx), cy(camera_ref.cy)
  {
  }


  __host__ __device__
  PinholeCamera(float fx, float fy,
                float cx, float cy)
    : fx(fx), fy(fy), cx(cx), cy(cy)
  { }

  __host__ __device__ __forceinline__
  float3 cam2world(const float2 & uv) const
  {
    return make_float3((uv.x - cx)/fx,
                       (uv.y - cy)/fy,
                       1.0f);
  }

  __host__ __device__ __forceinline__
  float2 world2cam(const float3 & xyz) const
  {
    return make_float2(fx*xyz.x / xyz.z + cx,
                       fy*xyz.y / xyz.z + cy);
  }

  __host__ __device__ __forceinline__
  float3 world2cam_f3(const float3 & xyz) const
  {
    return make_float3(fx*xyz.x / xyz.z + cx,
                       fy*xyz.y / xyz.z + cy,
                       xyz.z);
  }

  __host__ __device__ __forceinline__
  float getOnePixAngle() const
  {
    return atan2f(1.0f, 2.0f*fx)*2.0f;
  }

    __host__ __device__ __forceinline__
  float getf() const
  {
    return (fx + fy) / 2;
  }

  float fx, fy;
  float cx, cy;
};

}