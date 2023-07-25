#pragma once
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<vcruntime_string.h>
#include"SingleUtilFunction.cuh"
#include"BatchUtilFunction.cuh"
#include"BatchCache.h"

auto gpu_add(KernelBigIntBatch* dev_a,KernelBigIntBatch* dev_b,KernelBigIntBatch* dev_c,int n)
{
	kernel_batch_add<<<24,1024>>>(dev_a,dev_b,dev_c,n);
	auto ret=cudaDeviceSynchronize();
	return ret;
}

auto gpu_add(KernelBigInt* dev_a,KernelBigInt* dev_b,KernelBigInt* dev_c,int n)
{
	kernel_single_add<<<24,1024>>>(dev_a,dev_b,dev_c,n);
	auto ret=cudaDeviceSynchronize();
	return ret;
}

auto gpu_subtract(KernelBigIntBatch* dev_a,KernelBigIntBatch* dev_b,KernelBigIntBatch* dev_c,int n)
{
	kernel_batch_subtract<<<24,1024>>>(dev_a,dev_b,dev_c,n);
	auto ret=cudaDeviceSynchronize();
	return ret;
}

auto gpu_subtract(KernelBigInt* dev_a,KernelBigInt* dev_b,KernelBigInt* dev_c,int n)
{
	kernel_single_subtract<<<24,1024>>>(dev_a,dev_b,dev_c,n);
	auto ret=cudaDeviceSynchronize();
	return ret;
}

auto gpu_mult(KernelBigIntBatch* dev_a,KernelBigIntBatch* dev_b,KernelBigIntBatch* dev_c,int n)
{
	kernel_batch_mult<<<24,256>>>(dev_a,dev_b,dev_c,n);
	auto ret=cudaDeviceSynchronize();
	return ret;
}

auto gpu_mult(KernelBigInt* dev_a,KernelBigInt* dev_b,KernelBigInt* dev_c,int n)
{
	kernel_single_mult<<<24,256>>>(dev_a,dev_b,dev_c,n);
	auto ret=cudaDeviceSynchronize();
	return ret;
}

auto gpu_div(KernelBigInt* dev_a,KernelBigInt* dev_b,KernelBigInt* dev_c,int n)
{
	kernel_single_div<<<24,256>>>(dev_a,dev_b,dev_c,n);
	auto ret=cudaDeviceSynchronize();
	return ret;
}

auto gpu_div(KernelBigIntBatch* dev_a,KernelBigIntBatch* dev_b,KernelBigIntBatch* dev_c,int n)
{
	kernel_batch_div<<<24,256>>>(dev_a,dev_b,dev_c,n);
	auto ret=cudaDeviceSynchronize();
	return ret;
}

auto gpu_mod(KernelBigIntBatch* dev_a,KernelBigIntBatch* dev_b,KernelBigIntBatch* dev_c,int n)
{
	kernel_batch_mod<<<24,256>>>(dev_a,dev_b,dev_c,n);
	auto ret=cudaDeviceSynchronize();
	return ret;
}

auto gpu_mod(KernelBigInt* dev_a,KernelBigInt* dev_b,KernelBigInt* dev_c,int n)
{
	kernel_single_mod<<<24,256>>>(dev_a,dev_b,dev_c,n);
	auto ret=cudaDeviceSynchronize();
	return ret;
}

auto gpu_left_shift(KernelBigIntBatch* dev_a,KernelBigIntBatch* dev_b,int* dev_bit,int n)
{
	kernel_batch_left_shift<<<24,256>>>(dev_a,dev_b,dev_bit,n);
	auto ret=cudaDeviceSynchronize();
	return ret;
}

auto gpu_left_shift(KernelBigInt* dev_a,KernelBigInt* dev_b,int* dev_bit,int n)
{
	kernel_single_left_shift<<<24,256>>>(dev_a,dev_b,dev_bit,n);
	auto ret=cudaDeviceSynchronize();
	return ret;
}

auto gpu_right_shift(KernelBigIntBatch* dev_a,KernelBigIntBatch* dev_b,int* dev_bit,int n)
{
	kernel_batch_right_shift<<<24,256>>>(dev_a,dev_b,dev_bit,n);
	auto ret=cudaDeviceSynchronize();
	return ret;
}

auto gpu_right_shift(KernelBigInt* dev_a,KernelBigInt* dev_b,int* dev_bit,int n)
{
	kernel_single_right_shift<<<24,256>>>(dev_a,dev_b,dev_bit,n);
	auto ret=cudaDeviceSynchronize();
	return ret;
}

auto gpu_power_mod(KernelBigInt* dev_a,KernelBigInt* dev_b,KernelBigInt* dev_m,KernelBigInt* dev_c,int n)
{
	kernel_single_power_mod<<<24,256>>>(dev_a,dev_b,dev_m,dev_c,n);
	auto ret=cudaDeviceSynchronize();
	return ret;
}

auto gpu_power_mod(KernelBigIntBatch* dev_a,KernelBigIntBatch* dev_b,KernelBigIntBatch* dev_m,KernelBigIntBatch* dev_c,int n)
{
	kernel_batch_power_mod<<<24,256>>>(dev_a,dev_b,dev_m,dev_c,n);
	auto ret=cudaDeviceSynchronize();
	return ret;
}

auto gpu_fixpower_mod(KernelBigInt* dev_a,KernelBigInt* dev_b,KernelBigInt* dev_m,KernelBigInt* dev_c,int n)
{
	kernel_single_fixpower_mod<<<24,256>>>(dev_a,dev_b,dev_m,dev_c,n);
	auto ret=cudaDeviceSynchronize();
	return ret;
}

auto gpu_fixpower_mod(KernelBigIntBatch* dev_a,KernelBigInt* dev_b,KernelBigIntBatch* dev_m,KernelBigIntBatch* dev_c,int n)
{
	kernel_batch_fixpower_mod<<<24,256>>>(dev_a,dev_b,dev_m,dev_c,n);
	auto ret=cudaDeviceSynchronize();
	return ret;
}

auto gpu_fixpower_fixmod(KernelBigInt* dev_a,KernelBigInt* dev_b,KernelBigInt* dev_m,KernelBigInt* dev_c,int n)
{
	kernel_single_fixpower_fixmod<<<24,256>>>(dev_a,dev_b,dev_m,dev_c,n);
	auto ret=cudaDeviceSynchronize();
	return ret;
}

auto gpu_fixpower_fixmod(KernelBigIntBatch* dev_a,KernelBigInt* dev_b,KernelBigInt* dev_m,KernelBigIntBatch* dev_c,int n)
{
	kernel_batch_fixpower_fixmod<<<24,256>>>(dev_a,dev_b,dev_m,dev_c,n);
	auto ret=cudaDeviceSynchronize();
	return ret;
}