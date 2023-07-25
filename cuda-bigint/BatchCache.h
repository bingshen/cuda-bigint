#pragma once
#include"KernelBigIntBatch.h"

#define CACHE_NUM 4
#define DIV_BID 0
#define DIV_KID 1
#define DIV_FID 2
#define DIV_RID 3

struct BatchCache:public ShareMemoryManage
{
	bool sign[CACHE_NUM][BATCH_SIZE];
	int real_length[CACHE_NUM][BATCH_SIZE];
	int blocks[CACHE_NUM][block_length][BATCH_SIZE];

	__device__ __host__ void get_kernel_bigint(int cache_id,int thread_id,KernelBigInt& ret)
	{
		int len=real_length[cache_id][thread_id];
		ret.sign=sign[cache_id][thread_id];
		ret.real_length=len;
		for(int j=0;j<len;++j)
			ret.blocks[j]=blocks[cache_id][j][thread_id];
	}

	__device__ __host__ void get_kernel_bigint(int cache_id,int thread_id,KernelBigIntBatch* ret)
	{
		int len=real_length[cache_id][thread_id];
		ret->sign[thread_id]=sign[cache_id][thread_id];
		ret->real_length[thread_id]=len;
		for(int j=0;j<len;++j)
			ret->blocks[j][thread_id]=blocks[cache_id][j][thread_id];
	}

	__device__ __host__ void clear(int thread_id)
	{
		for(int i=0;i<CACHE_NUM;++i)
		{
			sign[i][thread_id]=true;
			real_length[i][thread_id]=0;
			for(int j=0;j<block_length;++j)
			{
				blocks[i][j][thread_id]=0;
			}
		}
	}
};