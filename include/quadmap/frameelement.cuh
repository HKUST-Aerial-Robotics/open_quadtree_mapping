#ifndef FrameElement_CUH
#define FrameElement_CUH
#include <vector_types.h>
#include <cuda_toolkit/helper_math.h>
#include <stdio.h>
#include <quadmap/se3.cuh>
#include <quadmap/device_image.cuh>

#define KEYFRAME_NUM 60

namespace quadmap
{

struct FrameElement
{
  DeviceImage<float> *frame_ptr;
  SE3<float> transform;
  FrameElement()
  {
    frame_ptr = NULL;
  }
};

}
#endif // FrameElement_CUH
