#pragma once
#include"KernelBigInt.h"

#define BATCH_SIZE 196608

/*
* 采用SoA的内存结构来存储显卡需要的batch数据
* 好处是两个相邻线程读取的内存也是相邻的，GPU会自动优化读写效率
*/
struct KernelBigIntBatch:public ShareMemoryManage
{
	bool sign[BATCH_SIZE];
	int real_length[BATCH_SIZE];
	int blocks[block_length][BATCH_SIZE];

	__device__ __host__ void get_kernel_bigint(int i,KernelBigInt& ret)
	{
		ret.sign=sign[i];
		ret.real_length=real_length[i];
		for(int j=0;j<ret.real_length;++j)
			ret.blocks[j]=blocks[j][i];
	}

	__device__ __host__ void set_kernel_bigint(int i,KernelBigInt& ret)
	{
		sign[i]=ret.sign;
		real_length[i]=ret.real_length;
		for(int j=0;j<ret.real_length;++j)
			blocks[j][i]=ret.blocks[j];
	}

	__host__ void print(int i,int radix=10)
	{
		KernelBigInt ret;
		get_kernel_bigint(i,ret);
		ret.print(radix);
	}

	__host__ __device__ void print_blocks(int i)
	{
		KernelBigInt ret;
		get_kernel_bigint(i,ret);
		ret.print_blocks();
	}

	__device__ __host__ void clear()
	{
		for(int i=0;i<BATCH_SIZE;++i)
		{
			this->sign[i]=true;
			this->real_length[i]=0;
		}
	}
};

__device__ __host__ void convert_bigint_batch(KernelBigInt* a,KernelBigIntBatch* b,int n)
{
	for(int i=0;i<n;++i)
	{
		b->sign[i]=a[i].sign;
		b->real_length[i]=a[i].real_length;
		for(int j=0;j<a[i].real_length;++j)
			b->blocks[j][i]=a[i].blocks[j];
	}
}

__device__ __host__ void convert_bigint_array(KernelBigIntBatch* a,KernelBigInt* b,int n)
{
	for(int i=0;i<n;++i)
	{
		b[i].sign=a->sign[i];
		b[i].real_length=a->real_length[i];
		for(int j=0;j<a->real_length[i];++j)
			b[i].blocks[j]=a->blocks[j][i];
	}
}