#include <cuda_toolkit/helper_math.h>
#include <quadmap/device_image.cuh>
#include <quadmap/texture_memory.cuh>
#include <quadmap/match_parameter.cuh>
#include <quadmap/pixel_cost.cuh>
#include <ctime>

namespace quadmap
{
//function declear here!
void global_upsample(DeviceImage<float> &sparse_depth, DeviceImage<float> &depth);
void local_upsample(DeviceImage<float> &sparse_image, DeviceImage<float> &dense_image);
__global__ void build_weight_row(DeviceImage<float> *row_weight_devptr);
__global__ void build_weight_col(DeviceImage<float> *col_weight_devptr);
__global__ void smooth_row(DeviceImage<float> *sparse_devptr, DeviceImage<float> *row_weight_devptr, DeviceImage<float3> *temp_devptr, DeviceImage<float2> *smooth_devptr);
__global__ void smooth_col(DeviceImage<float2> *row_smooth_devptr, DeviceImage<float> *col_weight_devptr, DeviceImage<float3> *temp_devptr, DeviceImage<float> *smooth_devptr);
__global__ void depth_interpolate(  DeviceImage<float> *featuredepth_devptr,
                                    DeviceImage<float> *depth_devptr);
//function define here!
void global_upsample(DeviceImage<float> &sparse_depth, DeviceImage<float> &depth)
{
	const int width = sparse_depth.width;
	const int height = sparse_depth.height;

	/*build weight*/
	DeviceImage<float> row_weight(width+1, height);
	DeviceImage<float> col_weight(width, height+1);
	row_weight.zero();
	col_weight.zero();
	dim3 weight_block;
	dim3 weight_grid;
	weight_block.x = 16;
	weight_block.y = 16;
	weight_grid.x = (width + weight_block.x - 1) / weight_block.x;
	weight_grid.y = (height + weight_block.y - 1) / weight_block.y;
	build_weight_row<<<weight_grid, weight_block>>>(row_weight.dev_ptr);
	build_weight_col<<<weight_grid, weight_block>>>(col_weight.dev_ptr);
	cudaDeviceSynchronize();

	/*smooth*/
	DeviceImage<float3> temp_data(width,height);
	DeviceImage<float2> smooth_temp_data(width,height);
	dim3 smooth_row_block;
	dim3 smooth_row_grid;
	smooth_row_block.x = 32;
	smooth_row_grid.x = (height + smooth_row_block.x - 1) / smooth_row_block.x;
	smooth_row<<<smooth_row_grid, smooth_row_block>>>(sparse_depth.dev_ptr, row_weight.dev_ptr, temp_data.dev_ptr, smooth_temp_data.dev_ptr);
	cudaDeviceSynchronize();


	dim3 smooth_col_block;
	dim3 smooth_col_grid;
	smooth_col_block.x = 8;
	smooth_col_grid.x = (width + smooth_col_block.x - 1) / smooth_col_block.x;
	smooth_col<<<smooth_col_grid, smooth_col_block>>>(smooth_temp_data.dev_ptr, col_weight.dev_ptr, temp_data.dev_ptr, depth.dev_ptr);
	cudaDeviceSynchronize();
}
__global__ void build_weight_row(DeviceImage<float> *row_weight_devptr)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int width = row_weight_devptr->width;
	const int height = row_weight_devptr->height;
	if(x >= width || y >= height || x == 0)
		return;
	float my_value = tex2D(income_image_tex, x + 0.5, y + 0.5);
	float left_value = tex2D(income_image_tex, x - 1 + 0.5, y + 0.5);
	float diff_color = my_value - left_value;
	#if use_fabs_distence
	float weight = expf(-fabs(diff_color)/upsample_sigma);
	#else
	float weight = expf(-diff_color*diff_color/upsample_sigma);
	#endif
	row_weight_devptr->atXY(x,y) = weight;
}
__global__ void build_weight_col(DeviceImage<float> *col_weight_devptr)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int width = col_weight_devptr->width;
	const int height = col_weight_devptr->height;
	if(x >= width || y >= height || y == 0)
		return;
	float my_value = tex2D(income_image_tex, x + 0.5, y + 0.5);
	float up_value = tex2D(income_image_tex, x + 0.5, y - 1 + 0.5);
	float diff_color = my_value - up_value;
	#if use_fabs_distence
	float weight = expf(-fabs(diff_color)/upsample_sigma);
	#else
	float weight = expf(-diff_color*diff_color/upsample_sigma);
	#endif
	col_weight_devptr->atXY(x,y) = weight;
}
__global__ void smooth_row(DeviceImage<float> *sparse_devptr, DeviceImage<float> *row_weight_devptr, DeviceImage<float3> *temp_devptr, DeviceImage<float2> *smooth_devptr)
{
	const int my_row = threadIdx.x + blockIdx.x * blockDim.x;
	const int width = sparse_devptr->width;
	const int height = sparse_devptr->height;
	if(my_row >= height)
		return;

	float weight[17];
	float depth[16];
	float3 info[16];
	float2 smoothed[16];
	float former_c = 0.0;
	float former_depth = 0.0;
	float former_indicator = 0.0;
	weight[16] = 0.0;

	int loop_times = width/16;
	int weight_begin_x = 1;
	int depth_begin_x = 0;

	//now forwards
	for(int this_loop = 0; this_loop < loop_times-1; this_loop++)
	{
		weight[0] = weight[16];
		memcpy(weight+1, row_weight_devptr->ptr_atXY(weight_begin_x, my_row), 16*sizeof(float));
		memcpy(depth, sparse_devptr->ptr_atXY(depth_begin_x, my_row), 16*sizeof(float));
		float this_indicator;
		for(int i = 0; i < 16; i++)
		{
			this_indicator = depth[i] > 0 ? 1 : 0; 
			float a_i = - upsample_lambda * weight[i];
			float b_i = 1 + upsample_lambda*(weight[i] + weight[i+1]);
			float c_i = - upsample_lambda * weight[i+1];
			float c_hat = c_i / (b_i - former_c * a_i);
			float depth_hat = (depth[i] - former_depth * a_i) / (b_i - former_c * a_i);
			float indicator_hat = (this_indicator - former_indicator * a_i) / (b_i - former_c * a_i);

			former_c = c_hat;
			former_depth = depth_hat;
			former_indicator = indicator_hat;
			info[i] = make_float3(c_hat, depth_hat, indicator_hat);
		}
		//write into the memory
		memcpy(temp_devptr->ptr_atXY(depth_begin_x, my_row), info, 16*sizeof(float3));

		weight_begin_x += 16;
		depth_begin_x += 16;
	}

	//last loop
	{
		weight[0] = weight[16];
		memcpy(weight+1, row_weight_devptr->ptr_atXY(weight_begin_x, my_row), 16*sizeof(float));
		memcpy(depth, sparse_devptr->ptr_atXY(depth_begin_x, my_row), 16*sizeof(float));
		float this_indicator;
		for(int i = 0; i < 16; i++)
		{
			this_indicator = depth[i] > 0 ? 1 : 0; 
			float a_i = - upsample_lambda * weight[i];
			float b_i = 1 + upsample_lambda*(weight[i] + weight[i+1]);
			float c_i = - upsample_lambda * weight[i+1];
			float c_hat = c_i / (b_i - former_c * a_i);
			float depth_hat = (depth[i] - former_depth * a_i) / (b_i - former_c * a_i);
			float indicator_hat = (this_indicator - former_indicator * a_i) / (b_i - former_c * a_i);

			former_c = c_hat;
			former_depth = depth_hat;
			former_indicator = indicator_hat;
			info[i] = make_float3(c_hat, depth_hat, indicator_hat);
		}
	}

	//now backwards
	former_depth = 0.0;
	former_indicator = 0.0;

	//first loop
	{
		for(int i = 15; i >= 0; i--)
		{
			smoothed[i].x = info[i].y - info[i].x * former_depth;
			smoothed[i].y = info[i].z - info[i].x * former_indicator;
			former_depth = smoothed[i].x;
			former_indicator = smoothed[i].y;
		}
		memcpy(smooth_devptr->ptr_atXY(depth_begin_x, my_row), smoothed, 16*sizeof(float2));
		depth_begin_x -= 16;
	}

	for(int this_loop = 0; this_loop < loop_times - 1; this_loop++)
	{
		memcpy(info, temp_devptr->ptr_atXY(depth_begin_x, my_row), 16*sizeof(float3));
		for(int i = 15; i >= 0; i--)
		{
			smoothed[i].x = info[i].y - info[i].x * former_depth;
			smoothed[i].y = info[i].z - info[i].x * former_indicator;
			former_depth = smoothed[i].x;
			former_indicator = smoothed[i].y;
		}
		memcpy(smooth_devptr->ptr_atXY(depth_begin_x, my_row), smoothed, 16*sizeof(float2));
		depth_begin_x -= 16;
	}
}
__global__ void smooth_col(DeviceImage<float2> *row_smooth_devptr, DeviceImage<float> *col_weight_devptr, DeviceImage<float3> *temp_devptr, DeviceImage<float> *smooth_devptr)
{
	const int my_col = threadIdx.x + blockDim.x * blockIdx.x;
	const int local_col = threadIdx.x;
	const int width = row_smooth_devptr->width;
	const int height = row_smooth_devptr->height;
	if(my_col >= width)
		return;

	__shared__ float weight[8][8];
	__shared__ float2 smooth[8][8];
	__shared__ float depth[8][8];
	__shared__ float3 info[8][8];

	float former_c = 0.0;
	float former_depth = 0.0;
	float former_indicator = 0.0;
	float former_weight = 0.0;
	int loop_times = height / 8;

	//loop
	int load_begin_x = blockDim.x * blockIdx.x;
	int load_begin_y = threadIdx.x;
	for(int i = 0; i < loop_times - 1; i++)
	{
		//load data
		memcpy(weight[local_col], col_weight_devptr->ptr_atXY(load_begin_x, load_begin_y + 1), 8*sizeof(float));
		memcpy(smooth[local_col], row_smooth_devptr->ptr_atXY(load_begin_x, load_begin_y), 8*sizeof(float2));
		__syncthreads();
		for(int i = 0; i < 8; i++)
		{
			float a_i = - upsample_lambda * former_weight;
			float b_i = 1 + upsample_lambda * (former_weight + weight[i][local_col]);
			float c_i = - upsample_lambda * weight[i][local_col];
			float c_hat = c_i / (b_i - former_c * a_i);
			float depth_hat = (smooth[i][local_col].x - former_depth * a_i) / (b_i - former_c * a_i);
			float indicator_hat = (smooth[i][local_col].y - former_indicator * a_i) / (b_i - former_c * a_i);

			former_c = c_hat;
			former_depth = depth_hat;
			former_indicator = indicator_hat;
			former_weight = weight[i][local_col];
			info[i][local_col] = make_float3(c_hat, depth_hat, indicator_hat);
		}
		__syncthreads();
		memcpy(temp_devptr->ptr_atXY(load_begin_x, load_begin_y), info[local_col], 8*sizeof(float3));
		load_begin_y += 8;
	}
	//last loop
	{
		//load data
		memcpy(weight[local_col], col_weight_devptr->ptr_atXY(load_begin_x, load_begin_y + 1), 8*sizeof(float));
		memcpy(smooth[local_col], row_smooth_devptr->ptr_atXY(load_begin_x, load_begin_y), 8*sizeof(float2));
		__syncthreads();
		for(int i = 0; i < 8; i++)
		{
			float a_i = - upsample_lambda * former_weight;
			float b_i = 1 + upsample_lambda*(former_weight + weight[i][local_col]);
			float c_i = - upsample_lambda * weight[i][local_col];
			float c_hat = c_i / (b_i - former_c * a_i);
			float depth_hat = (smooth[i][local_col].x - former_depth * a_i) / (b_i - former_c * a_i);
			float indicator_hat = (smooth[i][local_col].y - former_indicator * a_i) / (b_i - former_c * a_i);

			former_c = c_hat;
			former_depth = depth_hat;
			former_indicator = indicator_hat;
			former_weight = weight[i][local_col];
			info[i][local_col] = make_float3(c_hat, depth_hat, indicator_hat);
		}
	}

	//back wards
	former_depth = 0.0;
	former_indicator = 0.0;
	float smooth_depth;
	float smooth_indicator;
	{
		for(int i = 7; i >= 0; i--)
		{
			smooth_depth = info[i][local_col].y - info[i][local_col].x * former_depth;
			smooth_indicator = info[i][local_col].z - info[i][local_col].x * former_indicator;
			depth[i][local_col] = smooth_depth / smooth_indicator;
			former_depth = smooth_depth;
			former_indicator = smooth_indicator;
		}
		__syncthreads();
		memcpy(smooth_devptr->ptr_atXY(load_begin_x, load_begin_y), depth[local_col], 8*sizeof(float));
		load_begin_y -= 8;
	}
	for(int this_loop = 0; this_loop < loop_times - 1; this_loop++)
	{
		memcpy(info[local_col], temp_devptr->ptr_atXY(load_begin_x, load_begin_y), 8*sizeof(float3));
		for(int i = 7; i >= 0; i--)
		{
			smooth_depth = info[i][local_col].y - info[i][local_col].x * former_depth;
			smooth_indicator = info[i][local_col].z - info[i][local_col].x * former_indicator;
			depth[i][local_col] = smooth_depth / smooth_indicator;
			former_depth = smooth_depth;
			former_indicator = smooth_indicator;
		}
		__syncthreads();
		memcpy(smooth_devptr->ptr_atXY(load_begin_x, load_begin_y), depth[local_col], 8*sizeof(float));
		load_begin_y -= 8;
	}
}

void local_upsample(DeviceImage<float> &sparse_image, DeviceImage<float> &dense_image)
{
	int width = dense_image.width;
	int height = dense_image.height;
	dim3 interpolate_block;
	dim3 interpolate_grid;
	interpolate_block.x = 16;
	interpolate_block.y = 16;
	interpolate_grid.x = (width + interpolate_block.x - 1) / interpolate_block.x;
	interpolate_grid.y = (height + interpolate_block.y - 1) / interpolate_block.y;
	depth_interpolate<<<interpolate_grid, interpolate_block>>>(sparse_image.dev_ptr, dense_image.dev_ptr);
}

__global__ void depth_interpolate(  DeviceImage<float> *featuredepth_devptr,
                                    DeviceImage<float> *depth_devptr)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int image_width = depth_devptr->width;
  const int image_height = depth_devptr->height;

  if(x >= image_width || y >= image_height)
    return;

  const float this_intensity = tex2D(income_image_tex, x + 0.5f, y + 0.5f);
  const int keypoint_step = 4;

  float this_feature_weight;
  float this_feature_depth;
  float weight_sum = 0.0f;
  float sparital_factor = 100.0f;
  float intensity_factor = 50.0f;

  float interpolated_depth = 0.0f;

  for(int i = 0; i < 7; i ++)
  {
    for(int j = 0; j < 7; j++)
    {
      int check_x = (x / 4 + i - 3) * 4;
      int check_y = (y / 4 + j - 3) * 4;
      if(check_x < 0 || check_x >= image_width || check_y < 0 || check_y >= image_height)
      {
        this_feature_weight = 0.0f;
        this_feature_depth = 0.0f;
      }
      else
      {
      	__syncthreads();
        this_feature_depth = featuredepth_devptr->atXY(check_x, check_y);
        if(this_feature_depth <= 0.0)
        	continue;
        float intensity_diff = this_intensity - tex2D(income_image_tex, check_x + 0.5f, check_y + 0.5f);
        float2 sparital_diff = make_float2(x - check_x, y - check_y);
        float sparital_weight = sparital_diff.x * sparital_diff.x + sparital_diff.y * sparital_diff.y;
        float intensity_weight = intensity_diff * intensity_diff;
        this_feature_weight = expf( - sparital_weight / sparital_factor - intensity_weight / intensity_factor);
      }

      interpolated_depth += this_feature_weight * this_feature_depth;
      weight_sum += this_feature_weight;
    }
  }

  //normalize the weight
  depth_devptr->atXY(x,y) = interpolated_depth / weight_sum;
}
}