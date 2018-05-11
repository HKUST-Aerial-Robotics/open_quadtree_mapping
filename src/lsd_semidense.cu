#include <quadmap/device_image.cuh>
#include <quadmap/stereo_parameter.cuh>
#include <ctime>

//for camera model
#define fx(camera_para) camera_para.x
#define cx(camera_para) camera_para.y
#define fy(camera_para) camera_para.z
#define cy(camera_para) camera_para.w

namespace quadmap
{
__global__ void initialize_keyframe_kernel(
	DeviceImage<DepthSeed> *new_keyframe_devptr);
__global__ void initialize_keyframe_kernel(
	DeviceImage<DepthSeed> *new_keyframe_devptr,
	DeviceImage<int> *transtable_devptr,
	DeviceImage<float3> *new_info_devptr);
__global__ void propogate_keyframe_kernel(
	DeviceImage<DepthSeed> *old_keyframe_devptr,
	float4 camera_para, SE3<float> old_to_new,
	DeviceImage<int> *transtable_devptr,
	DeviceImage<float3> *new_info_devptr);
__global__ void regulizeDepth_kernel(
	DeviceImage<DepthSeed> *keyframe_devptr,
	bool removeOcclusions);
__global__ void regulizeDepth_FillHoles_kernel(
	DeviceImage<DepthSeed> *keyframe_devptr);
__global__ void update_keyframe_kernel(
	DeviceImage<DepthSeed> *keyframe_devptr,
	float4 camera_para,
	SE3<float> key_to_income,
	DeviceImage<float> *debug_devptr);
__global__ void depth_project_kernel(
	DeviceImage<DepthSeed> *keyframe_devptr,
	float4 camera_para,
	SE3<float> key_to_income,
	DeviceImage<float> *depth);
__global__ void depth_project_kernel(
	DeviceImage<DepthSeed> *keyframe_devptr,
	float4 camera_para,
	SE3<float> key_to_income,
	DeviceImage<float2> *depth);
__device__ __forceinline__ float search_point(
  	const int &x,
  	const int &y,
  	const int &width,
  	const int &height,
  	const float2 &epipolar_line,
  	const float &gradient_max,
  	const float &my_gradient,
  	const float &min_idep,
  	const float &max_idep,
  	const float4 &camera_para,
  	const SE3<float> &key_to_income,
  	float &result_idep,
  	float &result_var,
  	float &result_eplength);
__device__ __forceinline__ float subpixle_interpolate(
	const float &pre_cost,
	const float &cost,
	const float &post_cost);

__global__ void initialize_keyframe_kernel(DeviceImage<DepthSeed> *new_keyframe_devptr)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int width = new_keyframe_devptr->width;
	const int height = new_keyframe_devptr->height;

	if (x >= width - 3 || y >= height - 3 || x <= 2 || y <= 2)
		return;

	float2 this_gradient = tex2D(income_gradient_tex, x + 0.5, y + 0.5);
	float gradient_mag_2 = dot(this_gradient, this_gradient);

	if (gradient_mag_2 < MIN_GRAIDIENT * MIN_GRAIDIENT)
		return;

	DepthSeed initialized_seed;
	initialized_seed.initialize();
	new_keyframe_devptr->atXY(x,y) = initialized_seed;
}

__global__ void initialize_keyframe_kernel(DeviceImage<DepthSeed> *new_keyframe_devptr, DeviceImage<int> *transtable_devptr, DeviceImage<float3> *new_info_devptr)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int width = new_keyframe_devptr->width;
	const int height = new_keyframe_devptr->height;

	if (x >= width - 3 || y >= height - 3 || x <= 2 || y <= 2)
		return;

	int tranform_index = transtable_devptr->atXY(x,y);
	if(tranform_index <= 0)
		return;

	float3 initial_info = new_info_devptr->atXY(tranform_index%width, tranform_index/width);
	DepthSeed initialized_seed;
	initialized_seed.initialize(initial_info.x, initial_info.y, initial_info.z);

	new_keyframe_devptr->atXY(x,y) = initialized_seed;
}

__global__ void propogate_keyframe_kernel(
	DeviceImage<DepthSeed> *old_keyframe_devptr,
	float4 camera_para, SE3<float> old_to_new,
	DeviceImage<int> *transtable_devptr,
	DeviceImage<float3> *new_info_devptr)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int width = old_keyframe_devptr->width;
	const int height = old_keyframe_devptr->height;

	if (x >= width - 3 || y >= height - 3 || x <= 2 || y <= 2)
		return;

	DepthSeed old_seed = old_keyframe_devptr->atXY(x,y);
	if( ! old_seed.is_vaild() )
		return;

	//project to the new frame

	float3 old_point = make_float3((x - cx(camera_para))/fx(camera_para), (y - cy(camera_para))/fy(camera_para), 1.0);
	old_point /= old_seed.smooth_idepth();
	float3 new_point = old_to_new * old_point;

	float new_idepth = 1.0f / new_point.z;
	int new_x = new_point.x * new_idepth * fx(camera_para) + cx(camera_para) + 0.5;
	int new_y = new_point.y * new_idepth * fy(camera_para) + cy(camera_para) + 0.5;

	if(new_x <= 2 || new_x >= width - 3 || new_y <= 2 || new_y >= height - 3)
		return;

	//check gradient
	float2 new_gradient = tex2D(income_gradient_tex, new_x + 0.5, new_y + 0.5);
	float new_gradient_2 = dot(new_gradient, new_gradient);
	if( new_gradient_2 < MIN_GRAIDIENT * MIN_GRAIDIENT)
		return;

	float old_color = tex2D(keyframe_image_tex, x+0.5, y+0.5);
	float new_color = tex2D(income_image_tex, new_x+0.5, new_y+0.5);
	float diff_color = old_color - new_color;
	if(diff_color * diff_color > (MAX_DIFF_CONSTANT + MAX_DIFF_GRAD_MULT*new_gradient_2))
		return;

	//new info
	float idepth_ratio_4 = new_idepth / old_seed.smooth_idepth();
	idepth_ratio_4 *= idepth_ratio_4;
	idepth_ratio_4 *= idepth_ratio_4;
	float new_var = idepth_ratio_4*old_seed.smooth_variance();

	//try to add
	new_info_devptr->atXY(x,y) = make_float3(new_idepth, new_var, old_seed.vaild_counter());
	int *check_ptr = &(transtable_devptr->atXY(new_x, new_y));
	int expect_i = 0;
	int actual_i;
	int max_try = 5;
	bool finish_job = false;
	int my_index = y * width + x;
	while (!finish_job && max_try > 0)
	{
		max_try--;
		actual_i = atomicCAS(check_ptr, expect_i, my_index);
		if (actual_i != expect_i)
		{
			int other_x = actual_i % width;
			int other_y = actual_i / width;
			float other_idepth = (new_info_devptr->atXY(other_x, other_y)).x;
			if (other_idepth > new_idepth)
				finish_job = true;
		}
		else
		{
			finish_job = true;
		}
		expect_i = actual_i;
	}
}

__global__ void depth_project_kernel(
	DeviceImage<DepthSeed> *keyframe_devptr,
	float4 camera_para,
	SE3<float> key_to_income,
	DeviceImage<float> *depth)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int width = keyframe_devptr->width;
	const int height = keyframe_devptr->height;

	if (x >= width - 3 || y >= height - 3 || x <= 2 || y <= 2)
		return;

	DepthSeed keyseed = keyframe_devptr->atXY(x,y);
	if( ! keyseed.is_vaild() )
		return;

	if( keyseed.vaild_counter() < 20 )
		return;

	// depth->atXY(x,y) = 1.0f/ keyseed.smooth_idepth();
	// return;

	///*project to the new frame*///
	float3 key_point = make_float3((x - cx(camera_para))/fx(camera_para), (y - cy(camera_para))/fy(camera_para), 1.0);
	key_point /= keyseed.smooth_idepth();
	float3 new_point = key_to_income * key_point;

	float new_idepth = 1.0f / new_point.z;
	int new_x = new_point.x * new_idepth * fx(camera_para) + cx(camera_para) + 0.5;
	int new_y = new_point.y * new_idepth * fy(camera_para) + cy(camera_para) + 0.5;

	if(new_x <= 2 || new_x >= width - 3 || new_y <= 2 || new_y >= height - 3)
		return;

	// ///*check gradient*///
	float2 new_gradient = tex2D(income_gradient_tex, new_x + 0.5, new_y + 0.5);
	float new_gradient_2 = dot(new_gradient, new_gradient);
	if( new_gradient_2 < MIN_GRAIDIENT * MIN_GRAIDIENT)
		return;

	float old_color = tex2D(keyframe_image_tex, x+0.5, y+0.5);
	float new_color = tex2D(income_image_tex, new_x+0.5, new_y+0.5);
	float diff_color = old_color - new_color;
	if(diff_color * diff_color > (MAX_DIFF_CONSTANT + MAX_DIFF_GRAD_MULT*new_gradient_2))
		return;

	//try to add
	//ignore the occultion
	float new_depth = length(new_point);
	if(new_depth!=new_depth || keyseed.smooth_variance() > 0.02)
		return;

	depth->atXY(new_x,new_y) = new_depth;
	// depth->atXY(x,y) = 1.0/keyseed.smooth_idepth();
}

__global__ void depth_project_kernel(
	DeviceImage<DepthSeed> *keyframe_devptr,
	float4 camera_para,
	SE3<float> key_to_income,
	DeviceImage<float2> *semidense_render)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int width = keyframe_devptr->width;
	const int height = keyframe_devptr->height;

	if (x >= width - 3 || y >= height - 3 || x <= 2 || y <= 2)
		return;

	DepthSeed keyseed = keyframe_devptr->atXY(x,y);
	if( ! keyseed.is_vaild() )
		return;

	if( keyseed.vaild_counter() < 10 )
		return;

	///*project to the new frame*///
	float3 key_point = make_float3((x - cx(camera_para))/fx(camera_para), (y - cy(camera_para))/fy(camera_para), 1.0);
	key_point /= keyseed.smooth_idepth();
	float3 new_point = key_to_income * key_point;

	float new_idepth = 1.0f / new_point.z;
	int new_x = new_point.x * new_idepth * fx(camera_para) + cx(camera_para) + 0.5;
	int new_y = new_point.y * new_idepth * fy(camera_para) + cy(camera_para) + 0.5;

	if(new_x <= 2 || new_x >= width - 3 || new_y <= 2 || new_y >= height - 3)
		return;

	// ///*check gradient*///
	float2 new_gradient = tex2D(income_gradient_tex, new_x + 0.5, new_y + 0.5);
	float new_gradient_2 = dot(new_gradient, new_gradient);
	if( new_gradient_2 < MIN_GRAIDIENT * MIN_GRAIDIENT)
		return;

	float old_color = tex2D(keyframe_image_tex, x+0.5, y+0.5);
	float new_color = tex2D(income_image_tex, new_x+0.5, new_y+0.5);
	float diff_color = old_color - new_color;
	if(diff_color * diff_color > (MAX_DIFF_CONSTANT + MAX_DIFF_GRAD_MULT*new_gradient_2))
		return;

	//try to add
	float new_depth = length(new_point);
	float variance_scale = (new_depth / new_point.z);
	float new_variance = variance_scale * variance_scale * keyseed.smooth_variance();
	if(isfinite(new_depth) && isfinite(new_variance))
		return;
	new_variance = new_variance < 0.01 ? 0.01 : new_variance;
	semidense_render->atXY(new_x,new_y) = make_float2(1.0f/new_depth, new_variance);
}

__global__ void regulizeDepth_kernel(DeviceImage<DepthSeed> *keyframe_devptr, bool removeOcclusions)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int width = keyframe_devptr->width;
	const int height = keyframe_devptr->height;

	if (x >= width - 3 || y >= height - 3 || x <= 2 || y <= 2)
		return;

	DepthSeed depthseed = keyframe_devptr->atXY(x,y);
	if(!depthseed.is_vaild())
		return;

	float sum_idepth = 0.0;
	float sum_ivariance = 0.0;
	float sum_vaild = 0.0;
	int occlusion = 0;
	int not_occlusion = 0;
	for(int j = -2; j <= 2; j++)
	{
		for(int i = -2; i <= 2; i++)
		{
			DepthSeed other_seed = keyframe_devptr->atXY(x+i,y+j);
			if( ! other_seed.is_vaild() )
				continue;

			float diff_idepth = depthseed.idepth() - other_seed.idepth();
			if(diff_idepth * diff_idepth > depthseed.variance() + other_seed.variance())
			{
				if(depthseed.idepth() < other_seed.idepth())
					occlusion++;
				continue;
			}

			sum_vaild += other_seed.vaild_counter();
			not_occlusion++;

			float spatial_weight = REG_DIST_VAR * (i*i+j*j);
			float ivariance = 1.0f / (other_seed.variance() + spatial_weight);
			sum_idepth += ivariance * other_seed.idepth();
			sum_ivariance += ivariance;
		}
	}

	if(sum_vaild < VAL_SUM_MIN_FOR_KEEP)
		depthseed.set_invaild();
	if(removeOcclusions && occlusion > not_occlusion)
		depthseed.set_invaild();

	float smooth_idepth = sum_idepth / sum_ivariance;
	float smooth_variance = 1.0f / sum_ivariance;
	depthseed.set_smooth(smooth_idepth, smooth_variance);
	keyframe_devptr->atXY(x,y) = depthseed;
}
__global__ void regulizeDepth_FillHoles_kernel(DeviceImage<DepthSeed> *keyframe_devptr)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int width = keyframe_devptr->width;
	const int height = keyframe_devptr->height;

	if (x >= width - 3 || y >= height - 3 || x <= 2 || y <= 2)
		return;

	float2 this_gradient = tex2D(income_gradient_tex, x + 0.5, y + 0.5);
	float gradient_mag_2 = dot(this_gradient, this_gradient);

	if (gradient_mag_2 < MIN_GRAIDIENT * MIN_GRAIDIENT)
		return;

	DepthSeed depthseed = keyframe_devptr->atXY(x,y);
	if(depthseed.is_vaild())
		return;

	float sum_idepth = 0.0;
	float sum_ivariance = 0.0;
	float sum_vaild = 0.0;
	for(int j = -2; j<= 2; j++)
	{
		for(int i = -2; i <= 2; i++)
		{
			DepthSeed other_seed = keyframe_devptr->atXY(x+i,y+j);
			if( other_seed.is_vaild() )
			{
				sum_vaild += other_seed.vaild_counter();
				float weigth = 1.0f / other_seed.variance();
				sum_idepth += other_seed.idepth() * weigth;
				sum_ivariance += weigth;
			}
		}
	}

	bool create = (sum_vaild > VAL_SUM_MIN_FOR_UNBLACKLIST) || (depthseed.blacklist() >= MIN_BLACKLIST && sum_vaild > VAL_SUM_MIN_FOR_CREATE);
	if(create)
	{
		depthseed.initialize(sum_idepth/sum_ivariance, 1.0f/sum_ivariance, 0.0);
		keyframe_devptr->atXY(x,y) = depthseed;
	}
}

__global__ void update_keyframe_kernel(DeviceImage<DepthSeed> *keyframe_devptr, float4 camera_para, SE3<float> key_to_income, DeviceImage<float> *debug_devptr)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int width = keyframe_devptr->width;
	const int height = keyframe_devptr->height;

	if (x >= width - 3 || y >= height - 3 || x <= 2 || y <= 2)
		return;

	float2 this_gradient = tex2D(keyframe_gradient_tex, x + 0.5, y + 0.5);
	float gradient_max = length(this_gradient);

	if (gradient_max < MIN_GRAIDIENT)
		return;

	DepthSeed depthseed = keyframe_devptr->atXY(x,y);
	if(depthseed.blacklist() < MIN_BLACKLIST)
		return;

	bool first_observe = !(depthseed.is_vaild());

	//get epipolarline
	SE3<float> income_to_key = key_to_income.inv();
	float2 epiline;
	float gradient_alone_epipolar;
	{
		float3 income_to_key_t = income_to_key.getTranslation();
		epiline.x = - fx(camera_para) * income_to_key_t.x + income_to_key_t.z * (x - cx(camera_para));
		epiline.y = - fy(camera_para) * income_to_key_t.y + income_to_key_t.z * (y - cy(camera_para));
		float epi_length = length(epiline);
		if(epi_length < 1.0)
		{
			if(update_debug) debug_devptr->atXY(x,y) = 1;
			return;
		}
		//normalize epipolar
		epiline = epiline / epi_length;
		gradient_alone_epipolar = fabs(dot(epiline, this_gradient));

		if(gradient_alone_epipolar < 2.0)
		{
			if(update_debug) debug_devptr->atXY(x,y) = 2;
			return;
		}
		if(gradient_alone_epipolar / gradient_max < 0.3)
		{
			if(update_debug) debug_devptr->atXY(x,y) = 3;
			return;
		}
	}

	float search_error, result_idep, result_var, result_eplength;
	float search_min_idep = MIN_INV_DEPTH;
	float search_max_idep = MAX_INV_DEPTH;
	if(!first_observe)
	{
		float search_min_idep = depthseed.smooth_idepth() - 2.0 * sqrtf(depthseed.smooth_variance());
		float search_max_idep = depthseed.smooth_idepth() + 2.0 * sqrtf(depthseed.smooth_variance());
		search_min_idep = search_min_idep < MIN_INV_DEPTH ? MIN_INV_DEPTH : search_min_idep;
		search_max_idep = search_max_idep > MAX_INV_DEPTH ? MAX_INV_DEPTH : search_max_idep;
	}

	search_error = search_point(
		x,y,width,height,epiline,
		gradient_max,gradient_alone_epipolar,search_min_idep,search_max_idep,camera_para,key_to_income,
		result_idep,result_var,result_eplength);

	float deff_idep = result_idep - depthseed.smooth_idepth();

	if(first_observe)
	{
		if(search_error == -3 || search_error == -2)
		{
			if(update_debug) debug_devptr->atXY(x,y) = 5;
			depthseed.blacklist() --;
		}
		if(search_error > 0 && result_var < VARIANCE_MAX)
		{
			if(update_debug) debug_devptr->atXY(x,y) = 4;
			depthseed.initialize(result_idep, result_var, VALIDITY_COUNTER_INITIAL_OBSERVE);
		}
	}
	else
	{
		// if just not good for stereo (e.g. some inf / nan occured; has inconsistent minimum; ..)
		if(search_error == -2)
		{
			if(update_debug) debug_devptr->atXY(x,y) = 6;
			depthseed.vaild_counter() -= VALIDITY_COUNTER_DEC;
			depthseed.vaild_counter() = depthseed.vaild_counter() < 0 ? 0 : depthseed.vaild_counter();
			depthseed.variance() *= FAIL_VAR_INC_FAC;
			if(depthseed.variance() > VARIANCE_MAX)
			{
				depthseed.set_invaild();
				depthseed.blacklist()--;
			}
		}
		else if(search_error == -1 || search_error == -3 || search_error == -4)
		{
			if(update_debug) debug_devptr->atXY(x,y) = 7;
			return;
		}
		else if(deff_idep*deff_idep > result_var + depthseed.smooth_variance())
		{
			depthseed.variance() *= FAIL_VAR_INC_FAC;
			if(depthseed.variance() > VARIANCE_MAX)
			{
				depthseed.set_invaild();
			}
		}
		else
		{
			float increase_var = depthseed.variance() * SUCC_VAR_INC_FAC;
			float new_variance =  (increase_var * result_var) / (increase_var + result_var);
			float new_idepth = (depthseed.idepth() * result_var + result_idep * increase_var) / (increase_var + result_var);
			if(new_variance < depthseed.variance())
				depthseed.variance() = new_variance;
			if(new_variance < 0.0001)
				new_variance = 0.0001;
			depthseed.idepth() = new_idepth;
			depthseed.vaild_counter() += VALIDITY_COUNTER_INC;
			float max_vaild = VALIDITY_COUNTER_MAX + gradient_max * VALIDITY_COUNTER_MAX_VARIABLE / 255.0f;
			depthseed.vaild_counter() = depthseed.vaild_counter() > max_vaild ? max_vaild : depthseed.vaild_counter();
			if(update_debug) debug_devptr->atXY(x,y) = 4;
		}
	}
	//write into the memory
	keyframe_devptr->atXY(x,y)= depthseed;
}

__device__ __forceinline__ 
float search_point(
  	const int &x,
  	const int &y,
  	const int &width,
  	const int &height,
  	const float2 &epipolar_line,
  	const float &gradient_max,
  	const float &my_gradient,
  	const float &min_idep,
  	const float &max_idep,
  	const float4 &camera_para,
  	const SE3<float> &key_to_income,
  	float &result_idep,
  	float &result_var,
  	float &result_eplength)
{
  	//read patch
  	float this_patch[5];
  	#pragma unroll
  	for(int i = -2; i <= 2; i++)
  		this_patch[i+2] = tex2D(keyframe_image_tex, x + i * epipolar_line.x + 0.5, y + i * epipolar_line.y + 0.5);

  	//calculate the far point and near point
  	float3 KiP = make_float3((x-cx(camera_para))/fx(camera_para), (y-cy(camera_para))/fy(camera_para), 1);
  	float3 RKiP = key_to_income.rotate(KiP);
  	float3 T = key_to_income.getTranslation();
  	float3 KRKiP = make_float3(fx(camera_para)*RKiP.x + cx(camera_para)*RKiP.z, fy(camera_para)*RKiP.y + cy(camera_para)*RKiP.z, RKiP.z);
  	float3 KT = make_float3(fx(camera_para)*T.x + cx(camera_para)*T.z, fy(camera_para)*T.y + cy(camera_para)*T.z, T.z);

  	float3 near_point = KRKiP + KT * max_idep;
  	float3 far_point = KRKiP + KT * min_idep;
  	float2 near_uv 	= make_float2(near_point.x / near_point.z, near_point.y / near_point.z);
  	float2 far_uv 	= make_float2(far_point.x / far_point.z, far_point.y / far_point.z);
  	float search_length =length(near_uv-far_uv);
  	float2 search_dir = (near_uv-far_uv) / search_length;

  	if(search_length < 3.0)
  	{
  		near_uv += search_dir * 2.0;
  		far_uv -= search_dir * 2.0;
  		search_length += 4.0;
  	}
  	if(far_uv.x <= 1 || far_uv.y <= 1 || far_uv.x >= width-2 || far_uv.y >= height-2)
  		return -1;

  	//warp infomation
  	float that_patch[5];
  	#pragma unroll
  	for(int i = -2; i <= 1; i++)
  		that_patch[i+2] = tex2D(income_image_tex, far_uv.x + i * search_dir.x + 0.5, far_uv.y + i * search_dir.y + 0.5);

  	//search!
  	float best_step = -1;
	float best_score = 1e9;
	float pre_best_score = 1e9;
	float post_best_score = 1e9;
	bool last_is_best = false;

	// best pre and post errors.
	float best_match_DiffErrPre=NAN, best_match_DiffErrPost=NAN;
	// alternating intermediate vars
	float e1A=NAN, e1B=NAN, e2A=NAN, e2B=NAN, e3A=NAN, e3B=NAN, e4A=NAN, e4B=NAN, e5A=NAN, e5B=NAN;

	float second_best_score = 1e9;
	float second_best_step = -1;

	float pre_score = 1e9; //for loop use
	float this_score = 1e9;
	int step = 0;
	for (; step <= search_length && step <= MAX_SEARCH; step += 1)
	{
		pre_score = this_score;
		this_score = 0.0f;
		float2 search_point = far_uv + step * search_dir;
		if (search_point.x <= 1 || search_point.x >= width - 1 || search_point.y <= 1 || search_point.y >= height - 1)
			break;
		that_patch[4] = tex2D(income_image_tex, search_point.x + 2.0 * search_dir.x + 0.5, search_point.y + 2.0 * search_dir.y + 0.5);


		if(step%2==0)
		{
			e1A = that_patch[4] - this_patch[4];	this_score += e1A*e1A;
			e2A = that_patch[3] - this_patch[3];	this_score += e2A*e2A;
			e3A = that_patch[2] - this_patch[2];	this_score += e3A*e3A;
			e4A = that_patch[1] - this_patch[1];	this_score += e4A*e4A;
			e5A = that_patch[0] - this_patch[0];	this_score += e5A*e5A;
		}
		else
		{
			e1B = that_patch[4] - this_patch[4];	this_score += e1B*e1B;
			e2B = that_patch[3] - this_patch[3];	this_score += e2B*e2B;
			e3B = that_patch[2] - this_patch[2];	this_score += e3B*e3B;
			e4B = that_patch[1] - this_patch[1];	this_score += e4B*e4B;
			e5B = that_patch[0] - this_patch[0];	this_score += e5B*e5B;
		}

		if (last_is_best)
		{
			post_best_score = this_score;
			best_match_DiffErrPost = e1A*e1B + e2A*e2B + e3A*e3B + e4A*e4B + e5A*e5B;
			last_is_best = false;
		}

		if (this_score < best_score)
		{
			//chaneg the best to second
			second_best_score = best_score;
			second_best_step = best_step;

			best_score = this_score;
			best_step = step;
			pre_best_score = pre_score;
			best_match_DiffErrPre = e1A*e1B + e2A*e2B + e3A*e3B + e4A*e4B + e5A*e5B;
			best_match_DiffErrPost = -1;
			post_best_score = 1e9;
			last_is_best = true;
		}
		else if (this_score < second_best_score)
		{
			second_best_score = this_score;
			second_best_step = step;
		}

		that_patch[0] = that_patch[1];
		that_patch[1] = that_patch[2];
		that_patch[2] = that_patch[3];
		that_patch[3] = that_patch[4];
	}

	if(best_score > 4 * MAX_ERROR_STEREO)
		return -3;

	// check if clear enough winner
	if(abs(second_best_step - best_step) > 1.0f && MIN_DISTANCE_ERROR_STEREO * best_score > second_best_score)
	{
		return -2;
	}

	float best_match_sub = best_step;
	bool didSubpixel = false;
	//subpixel
	{
		// ================== compute exact match =========================
		// compute gradients (they are actually only half the real gradient)
		float gradPre_pre = -(pre_best_score - best_match_DiffErrPre);
		float gradPre_this = +(best_score - best_match_DiffErrPre);
		float gradPost_this = -(best_score - best_match_DiffErrPost);
		float gradPost_post = +(post_best_score - best_match_DiffErrPost);

		// final decisions here.
		bool interpPost = false;
		bool interpPre = false;

		// if pre has zero-crossing
		if((gradPre_pre < 0) ^ (gradPre_this < 0))
		{
			// if post has zero-crossing
			if((gradPost_post < 0) ^ (gradPost_this < 0))
			{
			}
			else
				interpPre = true;
		}
		// if post has zero-crossing
		else if((gradPost_post < 0) ^ (gradPost_this < 0))
		{
			interpPost = true;
		}


		// DO interpolation!
		// minimum occurs at zero-crossing of gradient, which is a straight line => easy to compute.
		// the error at that point is also computed by just integrating.
		if(interpPre)
		{
			float d = gradPre_this / (gradPre_this - gradPre_pre);
			best_match_sub -= d;
			best_score = best_score - 2*d*gradPre_this - (gradPre_pre - gradPre_this)*d*d;
			didSubpixel = true;

		}
		else if(interpPost)
		{
			float d = gradPost_this / (gradPost_this - gradPost_post);
			best_match_sub -= d;
			best_score = best_score + 2*d*gradPost_this + (gradPost_post - gradPost_this)*d*d;
			didSubpixel = true;
		}
	}

	float gradAlongLine = 0;
	#pragma unroll
	for(int i = 1; i < 5; i++)
		gradAlongLine += (this_patch[i] - this_patch[i-1])*(this_patch[i] - this_patch[i-1]);
	if(best_score > MAX_ERROR_STEREO + sqrtf(gradAlongLine) * 20)
		return -3;

	//get the idepth
	float best_idepth;
	float alpha;
	if(search_dir.x*search_dir.x > search_dir.y*search_dir.y)
	{
		float best_u = far_uv.x + best_match_sub * search_dir.x;
		best_idepth = (best_u * KRKiP.z - KRKiP.x) / (KT.x - best_u*KT.z);
		alpha = (KRKiP.z*KT.x - KT.z*KRKiP.x) / ((KT.x - best_u*KT.z) * (KT.x - best_u*KT.z));
	}
	else
	{
		float best_v = far_uv.y + best_match_sub * search_dir.y;
		best_idepth = (best_v * KRKiP.z - KRKiP.y)/(KT.y - best_v*KT.z);
		alpha = (KRKiP.z*KT.y - KT.z*KRKiP.y) / ((KT.y - best_v*KT.z) * (KT.y - best_v*KT.z));
	}

	best_idepth = best_idepth < MIN_INV_DEPTH ? MIN_INV_DEPTH : best_idepth;
	best_idepth = best_idepth > MAX_INV_DEPTH ? MAX_INV_DEPTH : best_idepth;

	// ================= calc var (in NEW image) ====================
	float photoDispError = 4.0f * 4.0f / my_gradient / my_gradient;
	float trackingErrorFac = 0.25f;
	float geoDispError = trackingErrorFac * gradient_max * gradient_max / (my_gradient*my_gradient);
	result_var = alpha*alpha*( (didSubpixel?0.05:0.5) +  geoDispError + photoDispError);
	result_idep = best_idepth;
	result_eplength = step;
	return best_score;
}

__device__ __forceinline__ float subpixle_interpolate(const float &pre_cost, const float &cost, const float &post_cost)
{
	float a = pre_cost - 2.0f * cost + post_cost;
	float b = -pre_cost + post_cost;
	return -b / (2 * a);
}

}