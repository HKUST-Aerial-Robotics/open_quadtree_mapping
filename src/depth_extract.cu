#include <quadmap/device_image.cuh>
#include <quadmap/stereo_parameter.cuh>
#include <ctime>

namespace quadmap
{
__global__ void prior_to_cost(
	DeviceImage<float2> *depth_prior_devptr,
	DeviceImage<PIXEL_COST> *cost_devptr,
	DeviceImage<int> *num_devptr);
__global__ void image_to_cost(
	MatchParameter *match_parameter_devptr,
	DeviceImage<int> *age_table_devptr,
	DeviceImage<PIXEL_COST> *cost_devptr,
	DeviceImage<int> *num_devptr);
__global__ void normalize_the_cost(
	DeviceImage<PIXEL_COST> *cost_devptr,
	DeviceImage<int> *num_devptr);
__global__ void upsample_naive(
	DeviceImage<float> *coarse_depth_devptr,
	DeviceImage<float> *full_dense_devptr);

__global__ void prior_to_cost(
	DeviceImage<float2> *depth_prior_devptr,
	DeviceImage<PIXEL_COST> *cost_devptr,
	DeviceImage<int> *num_devptr)
{
	const int x = blockIdx.x;
	const int y = blockIdx.y;
	const int depth_id = threadIdx.x;
	const int width = depth_prior_devptr->width;
	const int height = depth_prior_devptr->height;

	if (x >= width - 3 || y >= height - 3 || x <= 2 || y <= 2)
		return;

	float2 prior = depth_prior_devptr->atXY(x,y);
	if(prior.x <= 0 || prior.y <= 0)
		return;

	const float my_invdepth = STEP_INV_DEPTH * depth_id + MIN_INV_DEPTH;
	float cost = PRIOR_COST_SCALE * (my_invdepth-prior.x) * (my_invdepth-prior.x) / prior.y;
	// float cost = PRIOR_COST_SCALE * (my_depth-prior.x) * (my_depth-prior.x);
	cost = cost < TRUNCATE_COST ? cost : TRUNCATE_COST;
	atomicAdd(cost_devptr->atXY(x/4,y/4).cost_ptr(depth_id), cost);
	atomicAdd(num_devptr->ptr_atXY(x/4,y/4),1);
}
 
// __global__ void image_to_cost(
// 	MatchParameter *match_parameter_devptr,
// 	DeviceImage<int> *age_table_devptr,
// 	DeviceImage<PIXEL_COST> *cost_devptr,
// 	DeviceImage<int> *num_devptr)
// {
// 	const int x = blockIdx.x*4;
// 	const int y = blockIdx.y*4;
// 	const int depth_id = threadIdx.x;
// 	const int frame_id = threadIdx.y;
// 	const int width = age_table_devptr->width;
// 	const int height = age_table_devptr->height;
// 	const int frame_num = match_parameter_devptr->current_frames;
// 	const int my_quadsize = 1 << tex2D(quadtree_tex, x, y);

// 	if (x >= width -1 || y >= height - 1 || x <= 0 || y <= 0 || x % my_quadsize != 0 || y % my_quadsize != 0 )
// 		return;

// 	int this_age = age_table_devptr->atXY(x,y);

// 	if(this_age >= frame_num)
// 		this_age = frame_num - 1;
// 	if(this_age < 5)
// 		this_age = 5;

// 	__shared__ float cost[DEPTH_NUM];
// 	__shared__ int aggregate_num[DEPTH_NUM];
// 	if(frame_id == 0)
// 	{
// 		cost[depth_id] = 0;
// 		aggregate_num[depth_id] = 0;
// 	}
// 	const int my_frame_id = (float) this_age / 5.0 * (float) frame_id;
// 	__syncthreads();
// 	//read memory
// 	PinholeCamera camera = match_parameter_devptr->camera_model;
// 	FrameElement my_reference = match_parameter_devptr->framelist_dev[my_frame_id];
// 	float my_patch[3][3];
// 	for(int j = 0; j < 3; j++)
// 	{
// 		for(int i = 0; i < 3; i++)
// 		{
// 			my_patch[i][j] = tex2D(income_image_tex, x + i - 1.0 + 0.5, y + j - 1.0 + 0.5);
// 		}
// 	}

// 	//calculate
// 	const SE3<float> income_to_ref = my_reference.transform;
// 	float3 my_dir = normalize(camera.cam2world(make_float2(x, y)));
// 	float my_depth = 1.0 / (STEP_INV_DEPTH * depth_id + MIN_INV_DEPTH);
// 	float2 project_point = camera.world2cam(income_to_ref*(my_dir*my_depth));
// 	int point_x = project_point.x + 0.5;
// 	int point_y = project_point.y + 0.5;
// 	float u2u = income_to_ref.data(0,0);
// 	float u2v = income_to_ref.data(1,0);
// 	float v2u = income_to_ref.data(0,1);
// 	float v2v = income_to_ref.data(1,1);

// 	if( point_x >= 2 && point_x < width - 2 && point_y >= 2 && point_y < height - 2)
// 	{
// 		float my_cost = 0.0;
// 		for(int j = 0; j < 3; j++)
// 		{
// 			for(int i = 0; i < 3; i++)
// 			{
// 				int check_x = project_point.x + u2u * i + v2u * j - 1.0 + 0.5;
// 				int check_y = project_point.y + u2v * i + v2v * j - 1.0 + 0.5;
// 				my_cost += fabs(my_patch[i][j] - my_reference.frame_ptr->atXY(check_x, check_y));
// 			}
// 		}
// 		atomicAdd(cost + depth_id, my_cost);
// 		atomicAdd(aggregate_num + depth_id, 1);
// 	}
// 	__syncthreads();
// 	if(frame_id == 0)
// 	{
// 		float my_depth_cost = cost[depth_id];
// 		if(aggregate_num[depth_id] > 0)
// 			my_depth_cost = my_depth_cost / (float)aggregate_num[depth_id] / 255.0f;
// 		else
// 			my_depth_cost = 100;
// 		atomicAdd(cost_devptr->atXY(x/4,y/4).cost_ptr(depth_id), my_depth_cost);
// 		atomicAdd(num_devptr->ptr_atXY(x/4,y/4),1);
// 	}
// }

__global__ void image_to_cost(
	MatchParameter *match_parameter_devptr,
	DeviceImage<int> *age_table_devptr,
	DeviceImage<PIXEL_COST> *cost_devptr,
	DeviceImage<int> *num_devptr)
{
	const int x = blockIdx.x*4;
	const int y = blockIdx.y*4;
	const int depth_id = threadIdx.x;
	const int frame_id = threadIdx.y;
	const int width = age_table_devptr->width;
	const int height = age_table_devptr->height;
	const int frame_num = match_parameter_devptr->current_frames;
	const int my_quadsize = 1 << tex2D(quadtree_tex, x, y);

	if (x >= width -1 || y >= height - 1 || x <= 0 || y <= 0 || x % my_quadsize != 0 || y % my_quadsize != 0 )
		return;

	int this_age = age_table_devptr->atXY(x,y);

	if(this_age >= frame_num)
		this_age = frame_num - 1;
	if(this_age < 10)
		this_age = 9;

	__shared__ float cost[DEPTH_NUM][10];
	__shared__ float aggregate_num[DEPTH_NUM][10];
	const int my_frame_id = (float) this_age / 10.0 * (float) frame_id;

	//read memory
	PinholeCamera camera = match_parameter_devptr->camera_model;
	FrameElement my_reference = match_parameter_devptr->framelist_dev[my_frame_id];
	float my_patch[3][3];
	for(int j = 0; j < 3; j++)
	{
		for(int i = 0; i < 3; i++)
		{
			my_patch[i][j] = tex2D(income_image_tex, x + i - 1.0 + 0.5, y + j - 1.0 + 0.5);
		}
	}

	//calculate
	const SE3<float> income_to_ref = my_reference.transform;
	float3 my_dir = normalize(camera.cam2world(make_float2(x, y)));
	float my_depth = 1.0 / (STEP_INV_DEPTH * depth_id + MIN_INV_DEPTH);
	float2 project_point = camera.world2cam(income_to_ref*(my_dir*my_depth));
	int point_x = project_point.x + 0.5;
	int point_y = project_point.y + 0.5;
	float u2u = income_to_ref.data(0,0);
	float u2v = income_to_ref.data(1,0);
	float v2u = income_to_ref.data(0,1);
	float v2v = income_to_ref.data(1,1);

	if( point_x >= 2 && point_x < width - 2 && point_y >= 2 && point_y < height - 2)
	{
		float my_cost = 0.0;
		for(int j = -1; j <= 1; j++)
		{
			for(int i = -1; i <= 1; i++)
			{
				int check_x = project_point.x + u2u * i + v2u * j + 0.5;
				int check_y = project_point.y + u2v * i + v2v * j + 0.5;
				my_cost += fabs(my_patch[i+1][j+1] - my_reference.frame_ptr->atXY(check_x, check_y));
			}
		}
		cost[depth_id][frame_id] = my_cost;
		aggregate_num[depth_id][frame_id] = 1;
	}
	else
	{
		cost[depth_id][frame_id] = 0;
		aggregate_num[depth_id][frame_id] = 0;
	}
	__syncthreads();
	for(int r = 8; r > 0; r /= 2)
	{
		if(frame_id + r < blockDim.y && frame_id < r)
		{
		  cost[depth_id][frame_id] += cost[depth_id][frame_id+r];
		  aggregate_num[depth_id][frame_id] += aggregate_num[depth_id][frame_id+r];
		}
		__syncthreads();
	}

	if(frame_id == 0)
	{
		float my_depth_cost = cost[depth_id][0];
		if(aggregate_num[depth_id][0] > 0)
			my_depth_cost = my_depth_cost / (float)aggregate_num[depth_id][0] / 255.0f;
		else
			my_depth_cost = 100;
		atomicAdd(cost_devptr->atXY(x/4,y/4).cost_ptr(depth_id), my_depth_cost);
		atomicAdd(num_devptr->ptr_atXY(x/4,y/4),1);
	}
}

__global__ void normalize_the_cost(
	DeviceImage<PIXEL_COST> *cost_devptr,
	DeviceImage<int> *num_devptr)
{
	const int x = blockIdx.x;
	const int y = blockIdx.y;
	const int depth_id = threadIdx.x;
	const int add_num = num_devptr->atXY(x,y);
	if(add_num<=0)
		return;
	float mycost = cost_devptr->atXY(x,y).get_cost(depth_id);
	mycost /= (float)add_num;
	cost_devptr->atXY(x,y).set_cost(depth_id, mycost);
}

__global__ void naive_extract(
	DeviceImage<PIXEL_COST> *cost_devptr,
	DeviceImage<float> *coarse_depth_devptr)
{
	const int x = blockIdx.x;
	const int y = blockIdx.y;

	const int depth_id = threadIdx.x;
	__shared__ float cost[DEPTH_NUM];
	__shared__ float min_cost[DEPTH_NUM];
	__shared__ int min_id[DEPTH_NUM];
	cost[depth_id] = cost_devptr->atXY(x,y).get_cost(depth_id);
	min_cost[depth_id] = cost[depth_id];
	min_id[depth_id] = depth_id;
	__syncthreads();

	for(int i = DEPTH_NUM / 2; i > 0; i /= 2)
	{
		if( depth_id < i && min_cost[depth_id+i] < min_cost[depth_id])
		{
			min_cost[depth_id] = min_cost[depth_id+i];
			min_id[depth_id] = min_id[depth_id+i];
		}
		__syncthreads();
	}

	if(depth_id == 0)
	{
		float disparity;
		if(min_id[0] == 0 || min_id[0] == DEPTH_NUM - 1)
			disparity = min_id[0];
		else
		{
			float cost_pre = cost[min_id[0] - 1];
			float cost_post = cost[min_id[0] + 1];
			float a = cost_pre - 2.0f * min_cost[0] + cost_post;
			float b = - cost_pre + cost_post;
			disparity = (float) min_id[0] - b / (2.0f * a);
		}
		coarse_depth_devptr->atXY(x,y) = 1.0 / (STEP_INV_DEPTH * disparity + MIN_INV_DEPTH);
	}
}

__global__ void upsample_naive(
	DeviceImage<float> *coarse_depth_devptr,
	DeviceImage<float> *full_dense_devptr)
{
	const int x = threadIdx.x + blockDim.x * blockIdx.x;
	const int y = threadIdx.y + blockDim.y * blockIdx.y;
	const int width = full_dense_devptr->width;
	const int height = full_dense_devptr->height;
	if(x >= width || y >= height)
		return;
	const int my_quadsize = 1 << tex2D(quadtree_tex, x, y);
	full_dense_devptr->atXY(x,y) = coarse_depth_devptr->atXY(x/my_quadsize*my_quadsize, y/my_quadsize*my_quadsize);
}
	

}