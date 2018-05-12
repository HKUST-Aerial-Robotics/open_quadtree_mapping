// #pragma once
#include <vector>
#include <vector_types.h>
#include <cuda_toolkit/helper_math.h>
#include <quadmap/device_image.cuh>
#include <quadmap/texture_memory.cuh>
#include <quadmap/match_parameter.cuh>
#include <quadmap/camera_model/pinhole_camera.cuh>

namespace quadmap
{

#define initial_a 10
#define initial_b 10
#define initial_variance 2500

__device__ __forceinline__ float normpdf(const float &x, const float &mu, const float &sigma_sq)
{
  return (expf(-(x-mu)*(x-mu) / (2.0f*sigma_sq))) * rsqrtf(2.0f*M_PI*sigma_sq);
}
__device__ __forceinline__ bool is_goodpoint(const float4 &point_info)
{
  return (point_info.x /(point_info.x + point_info.y) > 0.60);
}
__device__ __forceinline__ bool is_badpoint(const float4 &point_info)
{
  return (point_info.x < 0.001) || (point_info.x /(point_info.x + point_info.y) < 0.45);
}

__global__ void high_gradient_filter
(DeviceImage<float> *depth_output_devptr,
  DeviceImage<float> *filtered_depth_devptr)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int width = depth_output_devptr->width;
  const int height = depth_output_devptr->height;

  if(x >= width - 1 || y >= height - 1 || x == 0 || y == 0)
    return;

  float gradient_x = depth_output_devptr->atXY(x+1,y)-depth_output_devptr->atXY(x-1,y);
  float gradient_y = depth_output_devptr->atXY(x,y+1)-depth_output_devptr->atXY(x,y-1);
  if(gradient_x*gradient_x + gradient_y*gradient_y > 0.01)
    filtered_depth_devptr->atXY(x,y) = -1;
}

__global__ void fuse_transform(
  DeviceImage<float4> *pre_seeds_devptr,
  DeviceImage<int> *transform_table_devptr,
  SE3<float> last_to_cur,
  PinholeCamera camera)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int width = pre_seeds_devptr->width;
  const int height = pre_seeds_devptr->height;

  if(x >= width || y >= height)
    return;
  
  const int index = x + y * width;

  const float3 dir = normalize(camera.cam2world(make_float2(x, y)));

  float4 pixel_info = pre_seeds_devptr->atXY(x,y);

  if ( is_badpoint(pixel_info) )
    return;

  float3 projected = last_to_cur * (dir * pixel_info.z);
  float new_depth = length(projected);
  if(new_depth <= 0.5)
    return;

  pixel_info.z = new_depth;
  pixel_info.w += new_depth * 0.01;
  // pixel_info.y *= 1.001;

  const float2 project_point = camera.world2cam(projected);
  const int projecte_x = project_point.x + 0.5;
  const int projecte_y = project_point.y + 0.5;

  //projected out of the image
  if(projecte_x >= width || projecte_x < 0 || projecte_y >= height || projecte_y < 0)
    return;

  //check color diff
  // float origin_color = tex2D(pre_image_tex, x + 0.5, y + 0.5);
  // float trans_color = tex2D(income_image_tex, projecte_x + 0.5, projecte_y + 0.5);
  // if( fabs(origin_color-trans_color) > 30.0 )
  //   return;

  int *check_ptr = &(transform_table_devptr->atXY(projecte_x, projecte_y));
  int expect_i = 0;
  int actual_i;
  bool finish_job = false;
  int max_loop = 5;
  while(!finish_job && max_loop > 0)
  {
    max_loop--;
    actual_i = atomicCAS(check_ptr, expect_i, index);
    if(actual_i != expect_i)
    {
      int now_x = actual_i % width;
      int now_y = actual_i / width;
      float now_d = (pre_seeds_devptr->atXY(now_x, now_y)).z;
      if(now_d < new_depth)
        finish_job = true;
    }
    else
    {
      finish_job = true;
    }
    expect_i = actual_i;
  }

  pre_seeds_devptr->atXY(x,y) = pixel_info;
}

__global__ void hole_filling(DeviceImage<int> *transform_table_devptr)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int width = transform_table_devptr->width;
  const int height = transform_table_devptr->height;

  if(x >= width - 1 || y >= height - 1 || x <= 1 || y <= 1)
    return;

  const int transform_i = transform_table_devptr->atXY(x,y);

  if(transform_i == 0)
    return;

  for(int i = -1; i <= 1; i++)
  {
    for(int j = -1; j <= 1; j++)
    {
      int *neighbor = &(transform_table_devptr->atXY(x + j, y + i));
      atomicCAS(neighbor, 0, transform_i);
    }
  }
}

__global__ void fuse_currentmap(
  DeviceImage<int> *transform_table_devptr,
  DeviceImage<float> *depth_output_devptr,
  DeviceImage<float4> *former_depth_devptr,
  DeviceImage<float4> *new_depth_devptr)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;

  const int width = transform_table_devptr->width;
  const int height = transform_table_devptr->height;

  if(x >= width || y >= height)
    return;

  float depth_estimate = depth_output_devptr->atXY(x,y);
  // float uncertianity = depth_estimate * depth_estimate * 0.01;
  // float uncertianity = 1.0;
  float uncertianity = fmaxf(0.5,depth_estimate*0.2);
  uncertianity *= uncertianity;
  if(depth_estimate <= 0.0f)
    uncertianity = 1e9;

  int pre_position = transform_table_devptr->atXY(x,y);
  float4 pixel_info;
  if(pre_position > 0)
    pixel_info = former_depth_devptr->atXY(pre_position%width, pre_position/width);
  else
  { 
    pixel_info = make_float4(initial_a, initial_b, depth_estimate, initial_variance);
  }

  if( (depth_estimate - pixel_info.z)*(depth_estimate - pixel_info.z) > uncertianity + pixel_info.w)
     pixel_info = make_float4(initial_a, initial_b, depth_estimate, initial_variance);

  //orieigin info
  float a = pixel_info.x;
  float b = pixel_info.y;
  float miu = pixel_info.z;
  float sigma_sq = pixel_info.w;

  float new_sq = uncertianity * sigma_sq / (uncertianity + sigma_sq);
  float new_miu = (depth_estimate * sigma_sq + miu * uncertianity) / (uncertianity + sigma_sq);
  float c1 = (a / (a+b)) * normpdf(depth_estimate, miu, uncertianity + sigma_sq);
  float c2 = (b / (a+b)) * 1 / 50.0f;

  const float norm_const = c1 + c2;
  c1 = c1 / norm_const;
  c2 = c2 / norm_const;
  const float f = c1 * ((a + 1.0f) / (a + b + 1.0f)) + c2 *(a / (a + b + 1.0f));
  const float e = c1 * (( (a + 1.0f)*(a + 2.0f)) / ((a + b + 1.0f) * (a + b + 2.0f))) +
                  c2 *(a*(a + 1.0f) / ((a + b + 1.0f) * (a + b + 2.0f)));

  const float mu_prime = c1 * new_miu + c2 * miu;
  const float sigma_prime = c1 * (new_sq + new_miu * new_miu) + c2 * (sigma_sq + miu * miu) - mu_prime * mu_prime;
  const float a_prime = ( e - f ) / ( f - e/f );
  const float b_prime = a_prime * ( 1.0f - f ) / f;
  const float4 updated = make_float4(a_prime, b_prime, mu_prime, sigma_prime);

  __syncthreads();

  if(is_goodpoint(pixel_info))
    depth_output_devptr->atXY(x,y) = mu_prime;
  else
    depth_output_devptr->atXY(x,y) = -1.0f;

  new_depth_devptr->atXY(x,y) = updated;
}

}//namespace