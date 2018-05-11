#ifndef DEPTH_SEED_CUH
#define DEPTH_SEED_CUH

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <vector_types.h>
#include <cuda_toolkit/helper_math.h>
#include <stdio.h>
#include <quadmap/stereo_parameter.cuh>

// #define is_vaild(DepthSeed) DepthSeed.track_info.x > 0
// #define blacklist(DepthSeed) DepthSeed.track_info.y
// #define vaild_counter(DepthSeed) DepthSeed.track_info.z

// #define idepth(DepthSeed) DepthSeed.depth_info.x
// #define idepth_var(DepthSeed) DepthSeed.depth_info.y
// #define idepth_smooth(DepthSeed) DepthSeed.depth_info.z
// #define idepth_var_smooth(DepthSeed) DepthSeed.depth_info.w

struct DepthSeed
{
  __device__ __host__ __forceinline__
  void initialize()
  {
    track_info = make_float3(1, 0, 20);
    depth_info = make_float4(IDEPTH_INITIAL, VARIANCE_MAX, IDEPTH_INITIAL, VARIANCE_MAX);
  }

  __device__ __host__ __forceinline__
  void initialize(const float &idepth, const float &variance, const float &vaild_counter)
  {
    track_info = make_float3(1, 0, vaild_counter);
    depth_info = make_float4(idepth, variance, IDEPTH_INITIAL, VARIANCE_MAX);
  }
  __device__ __host__ __forceinline__
  void set_invaild()
  {
    track_info.x = -1;
  }
  __device__ __host__ __forceinline__
  bool is_vaild()
  {
    return track_info.x > 0;
  }
  __device__ __host__ __forceinline__
  float& idepth()
  {
    return depth_info.x;
  }
  __device__ __host__ __forceinline__
  float& variance()
  {
    return depth_info.y;
  }
  __device__ __host__ __forceinline__
  float& smooth_idepth()
  {
    return depth_info.z;
  }
  __device__ __host__ __forceinline__
  float& smooth_variance()
  {
    return depth_info.w;
  }
  __device__ __host__ __forceinline__
  float& vaild_counter()
  {
    return track_info.z;
  }  
  __device__ __host__ __forceinline__
  float& blacklist()
  {
    return track_info.y;
  }
  __device__ __host__ __forceinline__
  void set_smooth(const float &smooth_idepth, const float &smooth_variance)
  {
    depth_info.z = smooth_idepth;
    depth_info.w = smooth_variance;
  }
  float3 track_info;
  float4 depth_info;
};

#endif // DEPTH_SEED_CUH
