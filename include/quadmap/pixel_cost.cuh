#ifndef PIXEL_COST_CUH
#define PIXEL_COST_CUH

#define DEPTH_NUM 64
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <vector_types.h>
#include <cuda_toolkit/helper_math.h>
#include <stdio.h>

struct PIXEL_COST
{
  __host__ __device__ __forceinline__
  void add_cost(const int &depth_id, const float &now_cost)
  {
    cost[depth_id] += now_cost;
  }
  __device__ __forceinline__
  void set_cost(const int &depth_id, const float &now_cost)
  {
    cost[depth_id] = now_cost;
  }
  
  __device__ __forceinline__
  float* cost_ptr(const int &depth_id)
  {
    return &cost[depth_id];
  }

    __host__ __device__ __forceinline__
  float get_cost(const int &depth_id)
  {
    return cost[depth_id];
  }

  float cost[DEPTH_NUM];
};


#endif // PIXEL_COST_CUH
