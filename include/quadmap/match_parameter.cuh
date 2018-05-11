#ifndef MATCH_PARAMETER_CUH
#define MATCH_PARAMETER_CUH

#include <vector_types.h>
#include <cuda_toolkit/helper_math.h>
#include <quadmap/se3.cuh>
#include <quadmap/camera_model/pinhole_camera.cuh>
#include <quadmap/pixel_cost.cuh>
#include <quadmap/frameelement.cuh>

namespace quadmap
{

struct MatchParameter
{
  __host__
  MatchParameter():
  is_dev_allocated(false),
  current_frames(0)
  {
  }

  __host__
  void setDevData()
  {
    if(!is_dev_allocated)
    {
      // Allocate device memory
      const cudaError err = cudaMalloc(&dev_ptr, sizeof(*this));
      if(err != cudaSuccess)
        throw CudaException("MatchParameter, cannot allocate device memory to store parameters.", err);
      else
      {
        is_dev_allocated = true;
      }
    }

    // Copy data to device memory
    const cudaError err2 = cudaMemcpy(dev_ptr, this, sizeof(*this), cudaMemcpyHostToDevice);
    if(err2 != cudaSuccess)
      throw CudaException("MatchParameter, cannot copy image parameters to device memory.", err2);
  }

  PinholeCamera camera_model;

  // /*current_frames stroes vaild number in framelist_dev*/ //
  // /*framelist_dev stroes ref devptr and SE3 income_to_ref in framelist_dev*/ //
  int current_frames;
  FrameElement framelist_dev[KEYFRAME_NUM];

  MatchParameter *dev_ptr;
  bool is_dev_allocated;
};

} // namespace quadmap

#endif