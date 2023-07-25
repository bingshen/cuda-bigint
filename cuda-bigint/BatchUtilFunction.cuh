#pragma once
#include"BigIntDeviceFunction.h"

__global__ void kernel_batch_add(KernelBigIntBatch* dev_a,KernelBigIntBatch* dev_b,KernelBigIntBatch* dev_c,int n)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	while(i<n)
	{
		if(dev_a->sign[i]&&dev_b->sign[i])
		{
			if(dev_a->real_length[i]>dev_b->real_length[i])
				bigint_add(i,dev_a,dev_b,dev_c);
			else
				bigint_add(i,dev_b,dev_a,dev_c);
			dev_c->sign[i]=true;
		}
		else if(!dev_a->sign[i]&&!dev_b->sign[i])
		{
			if(dev_a->real_length[i]>dev_b->real_length[i])
				bigint_add(i,dev_a,dev_b,dev_c);
			else
				bigint_add(i,dev_b,dev_a,dev_c);
			dev_c->sign[i]=false;
		}
		else if(dev_a->sign[i]&&!dev_b->sign[i])
		{
			bool bigger=compare_abs(i,dev_a,dev_b)>0;
			if(bigger)
				bigint_subtract(i,dev_a,dev_b,dev_c);
			else
				bigint_subtract(i,dev_b,dev_a,dev_c);
			dev_c->sign[i]=bigger;
		}
		else
		{
			bool bigger=compare_abs(i,dev_b,dev_a)>0;
			if(bigger)
				bigint_subtract(i,dev_b,dev_a,dev_c);
			else
				bigint_subtract(i,dev_a,dev_b,dev_c);
			dev_c->sign[i]=bigger;
		}
		i=i+blockDim.x*gridDim.x;
	}
}

__global__ void kernel_batch_subtract(KernelBigIntBatch* dev_a,KernelBigIntBatch* dev_b,KernelBigIntBatch* dev_c,int n)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	while(i<n)
	{
		if(dev_a->sign[i]&&dev_b->sign[i])
		{
			bool bigger=compare_abs(i,dev_a,dev_b)>0;
			if(bigger)
				bigint_subtract(i,dev_a,dev_b,dev_c);
			else
				bigint_subtract(i,dev_b,dev_a,dev_c);
			dev_c->sign[i]=bigger;
		}
		else if(!dev_a->sign[i]&&!dev_b->sign[i])
		{
			bool bigger=compare_abs(i,dev_b,dev_a)>0;
			if(bigger)
				bigint_subtract(i,dev_b,dev_a,dev_c);
			else
				bigint_subtract(i,dev_a,dev_b,dev_c);
			dev_c->sign[i]=bigger;
		}
		else if(!dev_a->sign[i]&&dev_b->sign[i])
		{
			if(dev_a->real_length[i]>dev_b->real_length[i])
				bigint_add(i,dev_a,dev_b,dev_c);
			else
				bigint_add(i,dev_b,dev_a,dev_c);
			dev_c->sign[i]=false;
		}
		else
		{
			if(dev_a->real_length[i]>dev_b->real_length[i])
				bigint_add(i,dev_a,dev_b,dev_c);
			else
				bigint_add(i,dev_b,dev_a,dev_c);
			dev_c->sign[i]=true;
		}
		i=i+blockDim.x*gridDim.x;
	}
}

__global__ void kernel_batch_mult(KernelBigIntBatch* dev_a,KernelBigIntBatch* dev_b,KernelBigIntBatch* dev_c,int n)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	while(i<n)
	{
		bigint_mult(i,dev_a,dev_b,dev_c);
		dev_c->sign[i]=(dev_a->sign[i]==dev_b->sign[i]);
		i=i+blockDim.x*gridDim.x;
	}
}

__global__ void kernel_batch_div(KernelBigIntBatch* dev_a,KernelBigIntBatch* dev_b,KernelBigIntBatch* dev_c,int n)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	while(i<n)
	{
		bigint_div_mod(i,dev_a,dev_b,dev_c,false);
		dev_c->sign[i]=(dev_a->sign[i]==dev_b->sign[i]);
		i=i+blockDim.x*gridDim.x;
	}
}

__global__ void kernel_batch_mod(KernelBigIntBatch* dev_a,KernelBigIntBatch* dev_b,KernelBigIntBatch* dev_c,int n)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	while(i<n)
	{
		bigint_div_mod(i,dev_a,dev_b,dev_c,true);
		dev_c->sign[i]=dev_a->sign[i];
		i=i+blockDim.x*gridDim.x;
	}
}

__global__ void kernel_batch_left_shift(KernelBigIntBatch* dev_a,KernelBigIntBatch* dev_b,int* dev_bit,int n)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	while(i<n)
	{
		bigint_left_shift(i,dev_a,dev_b,dev_bit);
		i=i+blockDim.x*gridDim.x;
	}
}

__global__ void kernel_batch_right_shift(KernelBigIntBatch* dev_a,KernelBigIntBatch* dev_b,int* dev_bit,int n)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	while(i<n)
	{
		bigint_right_shift(i,dev_a,dev_b,dev_bit);
		i=i+blockDim.x*gridDim.x;
	}
}

__global__ void kernel_batch_power_mod(KernelBigIntBatch* dev_a,KernelBigIntBatch* dev_b,KernelBigIntBatch* dev_m,KernelBigIntBatch* dev_c,int n)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	while(i<n)
	{
		bigint_power_mod(i,dev_a,dev_b,dev_m,dev_c);
		i=i+blockDim.x*gridDim.x;
	}
}

__global__ void kernel_batch_fixpower_mod(KernelBigIntBatch* dev_a,KernelBigInt* dev_b,KernelBigIntBatch* dev_m,KernelBigIntBatch* dev_c,int n)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	while(i<n)
	{
		bigint_power_mod(i,dev_a,dev_b,dev_m,dev_c);
		i=i+blockDim.x*gridDim.x;
	}
}

__global__ void kernel_batch_fixpower_fixmod(KernelBigIntBatch* dev_a,KernelBigInt* dev_b,KernelBigInt* dev_m,KernelBigIntBatch* dev_c,int n)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	while(i<n)
	{
		bigint_power_mod(i,dev_a,dev_b,dev_m,dev_c);
		i=i+blockDim.x*gridDim.x;
	}
}