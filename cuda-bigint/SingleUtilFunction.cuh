#pragma once
#include"BigIntDeviceFunction.h"

__global__ void kernel_single_add(KernelBigInt* dev_a,KernelBigInt* dev_b,KernelBigInt* dev_c,int n)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	while(i<n)
	{
		if(dev_a[i].sign&&dev_b[i].sign)
		{
			if(dev_a[i].real_length>dev_b[i].real_length)
				bigint_add(dev_a[i],dev_b[i],dev_c[i]);
			else
				bigint_add(dev_b[i],dev_a[i],dev_c[i]);
			dev_c[i].sign=true;
		}
		else if(!dev_a[i].sign&&!dev_b[i].sign)
		{
			if(dev_a[i].real_length>dev_b[i].real_length)
				bigint_add(dev_a[i],dev_b[i],dev_c[i]);
			else
				bigint_add(dev_b[i],dev_a[i],dev_c[i]);
			dev_c[i].sign=false;
		}
		else if(dev_a[i].sign&&!dev_b[i].sign)
		{
			bool bigger=compare_abs(dev_a[i].blocks,dev_a[i].real_length,dev_b[i].blocks,dev_b[i].real_length)>0;
			if(bigger)
				bigint_subtract(dev_a[i],dev_b[i],dev_c[i]);
			else
				bigint_subtract(dev_b[i],dev_a[i],dev_c[i]);
			dev_c[i].sign=bigger;
		}
		else
		{
			bool bigger=compare_abs(dev_b[i].blocks,dev_b[i].real_length,dev_a[i].blocks,dev_a[i].real_length)>0;
			if(bigger)
				bigint_subtract(dev_b[i],dev_a[i],dev_c[i]);
			else
				bigint_subtract(dev_a[i],dev_b[i],dev_c[i]);
			dev_c[i].sign=bigger;
		}
		i=i+blockDim.x*gridDim.x;
	}
}

__global__ void kernel_single_subtract(KernelBigInt* dev_a,KernelBigInt* dev_b,KernelBigInt* dev_c,int n)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	while(i<n)
	{
		if(dev_a[i].sign&&dev_b[i].sign)
		{
			bool bigger=compare_abs(dev_a[i].blocks,dev_a[i].real_length,dev_b[i].blocks,dev_b[i].real_length)>0;
			if(bigger)
				bigint_subtract(dev_a[i],dev_b[i],dev_c[i]);
			else
				bigint_subtract(dev_b[i],dev_a[i],dev_c[i]);
			dev_c[i].sign=bigger;
		}
		else if(!dev_a[i].sign&&!dev_b[i].sign)
		{
			bool bigger=compare_abs(dev_b[i].blocks,dev_b[i].real_length,dev_a[i].blocks,dev_a[i].real_length)>0;
			if(bigger)
				bigint_subtract(dev_b[i],dev_a[i],dev_c[i]);
			else
				bigint_subtract(dev_a[i],dev_b[i],dev_c[i]);
			dev_c[i].sign=bigger;
		}
		else if(!dev_a[i].sign&&dev_b[i].sign)
		{
			if(dev_a[i].real_length>dev_b[i].real_length)
				bigint_add(dev_a[i],dev_b[i],dev_c[i]);
			else
				bigint_add(dev_b[i],dev_a[i],dev_c[i]);
			dev_c[i].sign=false;
		}
		else
		{
			if(dev_a[i].real_length>dev_b[i].real_length)
				bigint_add(dev_a[i],dev_b[i],dev_c[i]);
			else
				bigint_add(dev_b[i],dev_a[i],dev_c[i]);
			dev_c[i].sign=true;
		}
		i=i+blockDim.x*gridDim.x;
	}
}

__global__ void kernel_single_mult(KernelBigInt* dev_a,KernelBigInt* dev_b,KernelBigInt* dev_c,int n)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	while(i<n)
	{
		bigint_mult(dev_a[i],dev_b[i],dev_c[i]);
		dev_c[i].sign=(dev_a[i].sign==dev_b[i].sign);
		i=i+blockDim.x*gridDim.x;
	}
}

__global__ void kernel_single_div(KernelBigInt* dev_a,KernelBigInt* dev_b,KernelBigInt* dev_c,int n)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	while(i<n)
	{
		bigint_div_mod(dev_a[i],dev_b[i],dev_c[i],false);
		dev_c[i].sign=(dev_a[i].sign==dev_b[i].sign);
		i=i+blockDim.x*gridDim.x;
	}
}

__global__ void kernel_single_mod(KernelBigInt* dev_a,KernelBigInt* dev_b,KernelBigInt* dev_c,int n)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	while(i<n)
	{
		bigint_div_mod(dev_a[i],dev_b[i],dev_c[i],true);
		dev_c[i].sign=dev_a[i].sign;
		i=i+blockDim.x*gridDim.x;
	}
}

__global__ void kernel_single_left_shift(KernelBigInt* dev_a,KernelBigInt* dev_b,int* dev_bit,int n)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	while(i<n)
	{
		bigint_left_shift(dev_a[i],dev_b[i],dev_bit[i]);
		i=i+blockDim.x*gridDim.x;
	}
}

__global__ void kernel_single_right_shift(KernelBigInt* dev_a,KernelBigInt* dev_b,int* dev_bit,int n)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	while(i<n)
	{
		bigint_right_shift(dev_a[i],dev_b[i],dev_bit[i]);
		i=i+blockDim.x*gridDim.x;
	}
}

__global__ void kernel_single_power_mod(KernelBigInt* dev_a,KernelBigInt* dev_b,KernelBigInt* dev_m,KernelBigInt* dev_c,int n)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	while(i<n)
	{
		bigint_power_mod(dev_a[i],dev_b[i],dev_m[i],dev_c[i]);
		i=i+blockDim.x*gridDim.x;
	}
}

__global__ void kernel_single_fixpower_mod(KernelBigInt* dev_a,KernelBigInt* dev_b,KernelBigInt* dev_m,KernelBigInt* dev_c,int n)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	while(i<n)
	{
		bigint_power_mod(dev_a[i],dev_b,dev_m[i],dev_c[i]);
		i=i+blockDim.x*gridDim.x;
	}
}

__global__ void kernel_single_fixpower_fixmod(KernelBigInt* dev_a,KernelBigInt* dev_b,KernelBigInt* dev_m,KernelBigInt* dev_c,int n)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	while(i<n)
	{
		bigint_power_mod(dev_a[i],dev_b,dev_m,dev_c[i]);
		i=i+blockDim.x*gridDim.x;
	}
}