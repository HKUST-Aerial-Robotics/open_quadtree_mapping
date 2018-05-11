// #pragma once
#include <vector>
#include <vector_types.h>
#include <cuda_toolkit/helper_math.h>
#include <quadmap/device_image.cuh>

namespace quadmap
{

//kernal
__global__ void quadtree_image_kernal(DeviceImage<int> *quadtree_devptr);
__global__ void quadtree_depth_kernal(DeviceImage<float> *prior_depth_devptr, DeviceImage<int> *quadtree_devptr);
//according to the image and depth to select the point;
//if the depth is available, select the point according to the depth image, level = max(depth_level, image_level)
//if the depth is not available, level = max(4, image_level)

__global__ void quadtree_image_kernal(DeviceImage<int> *quadtree_devptr)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int width = quadtree_devptr->width;
  const int height = quadtree_devptr->height;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;

  __shared__ float pyramid_intensity[16][16];
  __shared__ int pyramid_num[16][16];
  __shared__ bool approve[16][16];

  if(x >= width || y >= height)
    return;

  const float my_intensity = tex2D(income_image_tex, x + 0.5f, y + 0.5f);
  int pyramid_level = 0;
  float average_color = my_intensity;
  pyramid_intensity[local_x][local_y] = my_intensity;
  pyramid_num[local_x][local_y] = 1;

  //go to find the level
  for(int i = 1; i <= 4; i++)
  {
    int level_x = local_x - local_x % (1 << i);
    int level_y = local_y - local_y % (1 << i);
    int num_pixels = (1 << i) * (1 << i);
    bool I_AM_LAST_NODE = (local_x % (1 << (i - 1)) == 0) && (local_y % (1 << (i - 1)) == 0);

    if(I_AM_LAST_NODE && (level_x != local_x || level_y != local_y))
    {
      atomicAdd(&(pyramid_intensity[level_x][level_y]), pyramid_intensity[local_x][local_y]);
      atomicAdd(&(pyramid_num[level_x][level_y]), pyramid_num[local_x][local_y]);
    }
    approve[level_x][level_y] = true;
    __syncthreads();

    if(pyramid_num[level_x][level_y] != num_pixels)
      break;

    average_color = pyramid_intensity[level_x][level_y] / float(num_pixels);
    if( fabs(my_intensity - average_color) > 10.0)
    {
      approve[level_x][level_y] = false;
    }
    __syncthreads();

    if(approve[level_x][level_y])
      pyramid_level = i;
    else
    {
      pyramid_num[level_x][level_y] = 0;
      break;
    }
    __syncthreads();
  }
  pyramid_level = pyramid_level < 2 ? 2 : pyramid_level;
  quadtree_devptr->atXY(x, y) = pyramid_level;
  // quadtree_devptr->atXY(x, y) = 2;
}

__global__ void quadtree_depth_kernal(DeviceImage<float> *prior_depth_devptr, DeviceImage<int> *quadtree_devptr)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int width = quadtree_devptr->width;
  const int height = quadtree_devptr->height;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;

  __shared__ float pyramid_invdepth[16][16];
  __shared__ int pyramid_num[16][16];
  __shared__ bool approve[16][16];
  __shared__ bool all_vaild;
  all_vaild = true;
  __syncthreads();

  if(x >= width || y >= height)
    return;

  const float my_depth = prior_depth_devptr->atXY(x,y);
  const bool vaild_depth = (my_depth > 0.0f);
  float my_invdepth;
  if(vaild_depth)
    my_invdepth = 1.0f / my_depth;
  else
    my_invdepth = 0.0f;
  int pyramid_level = 0;
  float average_invdepth = my_invdepth;
  pyramid_invdepth[local_x][local_y] = my_invdepth;
  pyramid_num[local_x][local_y] = 1;

  if(! vaild_depth)
    all_vaild = false;
  __syncthreads();

  //go to find the level
  for(int i = 1; i <= 4 && all_vaild; i++)
  {
    int level_x = local_x - local_x % (1 << i);
    int level_y = local_y - local_y % (1 << i);
    int num_pixels = (1 << i) * (1 << i);
    bool I_AM_LAST_NODE = (local_x % (1 << (i - 1)) == 0) && (local_y % (1 << (i - 1)) == 0);

    if(I_AM_LAST_NODE && (level_x != local_x || level_y != local_y))
    {
      atomicAdd(&(pyramid_invdepth[level_x][level_y]), pyramid_invdepth[local_x][local_y]);
      atomicAdd(&(pyramid_num[level_x][level_y]), pyramid_num[local_x][local_y]);
    }
    approve[level_x][level_y] = true;
    __syncthreads();

    if(pyramid_num[level_x][level_y] != num_pixels)
      break;

    average_invdepth = pyramid_invdepth[level_x][level_y] / float(num_pixels);
    if( fabs(my_invdepth - average_invdepth) > 0.01f )
    {
      approve[level_x][level_y] = false;
    }
    __syncthreads();

    if(approve[level_x][level_y])
      pyramid_level = i;
    else
    {
      pyramid_num[level_x][level_y] = 0;
      break;
    }
    __syncthreads();
  }

  int color_level = quadtree_devptr->atXY(x, y);
  int level = max(color_level, pyramid_level);

  quadtree_devptr->atXY(x, y) = level;
}

}//namespace
