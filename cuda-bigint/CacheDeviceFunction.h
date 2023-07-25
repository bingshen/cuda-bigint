#pragma once
#include"BatchCache.h"

__device__ __host__ int get_real_length(int id,KernelBigIntBatch* a,const int real_length)
{
	for(int i=real_length-1;i>=0;i--)
		if(a->blocks[i][id]!=0)
			return i+1;
	return 0;
}

__device__ __host__ inline int get_real_length(int cache_id,int thread_id,BatchCache* a,const int real_length)
{
	for(int i=real_length-1;i>=0;i--)
		if(a->blocks[cache_id][i][thread_id]!=0)
			return i+1;
	return 0;
}

__device__ __host__ void cache_subtract(int cache_id1,int cache_id2,int cache_id3,int thread_id,BatchCache* dev_cache)
{
	int a_len=dev_cache->real_length[cache_id1][thread_id];
	int b_len=dev_cache->real_length[cache_id2][thread_id];
	int carry=0;
	for(int i=0;i<b_len;++i)
	{
		long long x=(long long)(unsigned int)dev_cache->blocks[cache_id1][i][thread_id];
		long long y=(long long)(unsigned int)dev_cache->blocks[cache_id2][i][thread_id];
		long long block_sum=x-y-carry;
		carry=0;
		if(block_sum<0)
			carry=1;
		dev_cache->blocks[cache_id3][i][thread_id]=(int)((block_sum+BLOCK_MAX)&BLOCK_MASK);
	}
	for(int i=b_len;i<a_len;++i)
	{
		if(carry==0)
		{
			dev_cache->blocks[cache_id3][i][thread_id]=dev_cache->blocks[cache_id1][i][thread_id];
			continue;
		}
		long long x=(long long)(unsigned int)dev_cache->blocks[cache_id1][i][thread_id];
		long long block_sum=x-carry;
		carry=0;
		if(block_sum<0)
			carry=1;
		dev_cache->blocks[cache_id3][i][thread_id]=(int)((block_sum+BLOCK_MAX)&BLOCK_MASK);
	}
	dev_cache->real_length[cache_id3][thread_id]=get_real_length(cache_id3,thread_id,dev_cache,a_len);
}

__device__ __host__ inline int compare_abs(int cache_id1,int cache_id2,int thread_id,BatchCache* dev_cache)
{
	int a_len=dev_cache->real_length[cache_id1][thread_id];
	int b_len=dev_cache->real_length[cache_id2][thread_id];
	if(a_len>b_len)
		return 1;
	else if(a_len<b_len)
		return -1;
	else
	{
		for(int i=a_len-1;i>=0;i--)
		{
			long long x=(long long)(unsigned int)dev_cache->blocks[cache_id1][i][thread_id];
			long long y=(long long)(unsigned int)dev_cache->blocks[cache_id2][i][thread_id];
			if(x>y)
				return 1;
			else if(x==y)
				continue;
			else
				return -1;
		}
		return 0;
	}
}

__device__ __host__ inline bool is_bigger_equal(int cache_id1,int cache_id2,int thread_id,BatchCache* dev_cache)
{
	bool a_sign=dev_cache->sign[cache_id1][thread_id];
	bool b_sign=dev_cache->sign[cache_id2][thread_id];
	if((a_sign)&&(!b_sign))
		return true;
	else if((!a_sign)&&(b_sign))
		return false;
	else
	{
		int r=compare_abs(cache_id1,cache_id2,thread_id,dev_cache);
		if(a_sign&&b_sign)
			return r>=0;
		else
			return r<=0;
	}
}

__device__ __host__ inline bool is_bigger(int cache_id1,int cache_id2,int thread_id,BatchCache* dev_cache)
{
	bool a_sign=dev_cache->sign[cache_id1][thread_id];
	bool b_sign=dev_cache->sign[cache_id2][thread_id];
	if((a_sign)&&(!b_sign))
		return true;
	else if((!a_sign)&&(b_sign))
		return false;
	else
	{
		int r=compare_abs(cache_id1,cache_id2,thread_id,dev_cache);
		if(a_sign&&b_sign)
			return r>0;
		else
			return r<0;
	}
}

__device__ __host__ inline bool is_smaller_equal(int cache_id1,int cache_id2,int thread_id,BatchCache* dev_cache)
{
	bool a_sign=dev_cache->sign[cache_id1][thread_id];
	bool b_sign=dev_cache->sign[cache_id2][thread_id];
	if((a_sign)&&(!b_sign))
		return false;
	else if((!a_sign)&&(b_sign))
		return true;
	else
	{
		int r=compare_abs(cache_id1,cache_id2,thread_id,dev_cache);
		if(a_sign&&b_sign)
			return r<=0;
		else
			return r>=0;
	}
}

__device__ __host__ inline bool is_smaller(int cache_id1,int cache_id2,int thread_id,BatchCache* dev_cache)
{
	bool a_sign=dev_cache->sign[cache_id1][thread_id];
	bool b_sign=dev_cache->sign[cache_id2][thread_id];
	if((a_sign)&&(!b_sign))
		return false;
	else if((!a_sign)&&(b_sign))
		return true;
	else
	{
		int r=compare_abs(cache_id1,cache_id2,thread_id,dev_cache);
		if(a_sign&&b_sign)
			return r<0;
		else
			return r>0;
	}
}

__device__ __host__ inline bool is_equal(int cache_id1,int cache_id2,int thread_id,BatchCache* dev_cache)
{
	int a_len=dev_cache->real_length[cache_id1][thread_id];
	int b_len=dev_cache->real_length[cache_id2][thread_id];
	if(a_len!=b_len)
		return false;
	bool a_sign=dev_cache->sign[cache_id1][thread_id];
	bool b_sign=dev_cache->sign[cache_id2][thread_id];
	if(a_sign!=b_sign)
		return false;
	for(int i=0;i<a_len;++i)
	{
		long long x=(long long)(unsigned int)dev_cache->blocks[cache_id1][i][thread_id];
		long long y=(long long)(unsigned int)dev_cache->blocks[cache_id2][i][thread_id];
		if(x!=y)
			return false;
	}
	return true;
}

__device__ __host__ void cache_tiny_mult(int cache_id1,int cache_id2,int thread_id,unsigned int b,BatchCache* dev_cache)
{
	int a_len=dev_cache->real_length[cache_id1][thread_id];
	if(a_len==0||b==0)
	{
		dev_cache->real_length[cache_id2][thread_id]=0;
		return;
	}
	unsigned long long register_c[block_length];
	int total_len=a_len+1;
	memset(register_c,0,sizeof(unsigned long long)*(total_len));
	unsigned long long x,y;
	for(int i=0;i<a_len;++i)
	{
		x=(unsigned long long)(unsigned int)dev_cache->blocks[cache_id1][i][thread_id];
		y=(unsigned long long)(unsigned int)b;
		register_c[i]=register_c[i]+x*y;
		register_c[i+1]=register_c[i+1]+(register_c[i]>>BLOCK_BIT_SIZE);
		register_c[i]=register_c[i]&BLOCK_MASK;
	}
	for(int i=0;i<total_len;++i)
	{
		register_c[i+1]=register_c[i+1]+(register_c[i]>>BLOCK_BIT_SIZE);
		dev_cache->blocks[cache_id2][i][thread_id]=(int)(register_c[i]&BLOCK_MASK);
	}
	dev_cache->real_length[cache_id2][thread_id]=get_real_length(register_c,total_len);
	dev_cache->sign[cache_id2][thread_id]=dev_cache->sign[cache_id1][thread_id];
}

__device__ __host__ void cache_mult(int cache_id1,int cache_id2,int cache_id3,int thread_id,BatchCache* dev_cache)
{
	int a_len=dev_cache->real_length[cache_id1][thread_id];
	int b_len=dev_cache->real_length[cache_id2][thread_id];
	if(a_len==0||b_len==0)
	{
		dev_cache->real_length[cache_id3][thread_id]=0;
		return;
	}
	unsigned long long register_c[block_length];
	int total_len=a_len+b_len;
	memset(register_c,0,sizeof(unsigned long long)*(total_len));
	for(int i=0;i<a_len;++i)
	{
		for(int j=0;j<b_len;++j)
		{
			unsigned long long x=(unsigned long long)(unsigned int)dev_cache->blocks[cache_id1][i][thread_id];
			unsigned long long y=(unsigned long long)(unsigned int)dev_cache->blocks[cache_id2][j][thread_id];
			register_c[i+j]=register_c[i+j]+x*y;
			register_c[i+j+1]=register_c[i+j+1]+(register_c[i+j]>>BLOCK_BIT_SIZE);
			register_c[i+j]=register_c[i+j]&BLOCK_MASK;
		}
	}
	for(int i=0;i<total_len;++i)
	{
		register_c[i+1]=register_c[i+1]+(register_c[i]>>BLOCK_BIT_SIZE);
		dev_cache->blocks[cache_id3][i][thread_id]=(int)(register_c[i]&BLOCK_MASK);
	}
	dev_cache->real_length[cache_id3][thread_id]=get_real_length(register_c,total_len);
	dev_cache->sign[cache_id3][thread_id]=(dev_cache->sign[cache_id1][thread_id]==dev_cache->sign[cache_id2][thread_id]);
}