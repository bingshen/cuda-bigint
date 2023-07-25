#pragma once
#include"CacheDeviceFunction.h"

__device__ __host__ void bigint_add(int id,KernelBigIntBatch* a,KernelBigIntBatch* b,KernelBigIntBatch* c)
{
	int carry=0;
	for(int i=0;i<b->real_length[id];++i)
	{
		long long x=(long long)(unsigned int)a->blocks[i][id];
		long long y=(long long)(unsigned int)b->blocks[i][id];
		long long block_sum=x+y+carry;
		carry=(int)(block_sum>>BLOCK_BIT_SIZE);
		c->blocks[i][id]=(int)(block_sum&BLOCK_MASK);
	}
	for(int i=b->real_length[id];i<a->real_length[id];++i)
	{
		if(carry==0)
		{
			c->blocks[i][id]=a->blocks[i][id];
			continue;
		}
		long long x=(long long)(unsigned int)a->blocks[i][id];
		long long block_sum=x+carry;
		carry=(int)(block_sum>>BLOCK_BIT_SIZE);
		c->blocks[i][id]=(int)(block_sum&BLOCK_MASK);
	}
	c->blocks[a->real_length[id]][id]=carry;
	c->real_length[id]=get_real_length(id,c,a->real_length[id]+1);
}

__device__ __host__ void bigint_add(KernelBigInt& a,KernelBigInt& b,KernelBigInt& c)
{
	int carry=0;
	for(int i=0;i<b.real_length;++i)
	{
		long long x=(long long)(unsigned int)a.blocks[i];
		long long y=(long long)(unsigned int)b.blocks[i];
		long long block_sum=x+y+carry;
		carry=(int)(block_sum>>BLOCK_BIT_SIZE);
		c.blocks[i]=(int)(block_sum&BLOCK_MASK);
	}
	for(int i=b.real_length;i<a.real_length;++i)
	{
		if(carry==0)
		{
			c.blocks[i]=a.blocks[i];
			continue;
		}
		long long x=(long long)(unsigned int)a.blocks[i];
		long long block_sum=x+carry;
		carry=(int)(block_sum>>BLOCK_BIT_SIZE);
		c.blocks[i]=(int)(block_sum&BLOCK_MASK);
	}
	c.blocks[a.real_length]=carry;
	c.real_length=get_real_length(c.blocks,a.real_length+1);
}

__device__ __host__ void bigint_subtract(int id,KernelBigIntBatch* a,KernelBigIntBatch* b,KernelBigIntBatch* c)
{
	int carry=0;
	for(int i=0;i<b->real_length[id];++i)
	{
		long long x=(long long)(unsigned int)a->blocks[i][id];
		long long y=(long long)(unsigned int)b->blocks[i][id];
		long long block_sum=x-y-carry;
		carry=0;
		if(block_sum<0)
			carry=1;
		c->blocks[i][id]=(int)((block_sum+BLOCK_MAX)&BLOCK_MASK);
	}
	for(int i=b->real_length[id];i<a->real_length[id];++i)
	{
		if(carry==0)
		{
			c->blocks[i][id]=a->blocks[i][id];
			continue;
		}
		long long x=(long long)(unsigned int)a->blocks[i][id];
		long long block_sum=x-carry;
		carry=0;
		if(block_sum<0)
			carry=1;
		c->blocks[i][id]=(int)((block_sum+BLOCK_MAX)&BLOCK_MASK);
	}
	c->real_length[id]=get_real_length(id,c,a->real_length[id]);
}

__device__ __host__ void bigint_subtract(KernelBigInt& a,KernelBigInt& b,KernelBigInt& c)
{
	int carry=0;
	for(int i=0;i<b.real_length;++i)
	{
		long long x=(long long)(unsigned int)a.blocks[i];
		long long y=(long long)(unsigned int)b.blocks[i];
		long long block_sum=x-y-carry;
		carry=0;
		if(block_sum<0)
			carry=1;
		c.blocks[i]=(int)((block_sum+BLOCK_MAX)&BLOCK_MASK);
	}
	for(int i=b.real_length;i<a.real_length;++i)
	{
		if(carry==0)
		{
			c.blocks[i]=a.blocks[i];
			continue;
		}
		long long x=(long long)(unsigned int)a.blocks[i];
		long long block_sum=x-carry;
		carry=0;
		if(block_sum<0)
			carry=1;
		c.blocks[i]=(int)((block_sum+BLOCK_MAX)&BLOCK_MASK);
	}
	c.real_length=get_real_length(c.blocks,a.real_length);
}

__device__ __host__ void bigint_mult(int id,KernelBigIntBatch* a,KernelBigIntBatch* b,KernelBigIntBatch* c)
{
	int a_len=a->real_length[id];
	int b_len=b->real_length[id];
	if(a_len==0||b_len==0)
	{
		c->real_length[id]=0;
		return;
	}
	unsigned long long register_c[block_length];
	int total_len=a_len+b_len;
	memset(register_c,0,sizeof(unsigned long long)*(total_len));
	for(int i=0;i<a_len;++i)
	{
		for(int j=0;j<b_len;++j)
		{
			unsigned long long x=(unsigned long long)(unsigned int)a->blocks[i][id];
			unsigned long long y=(unsigned long long)(unsigned int)b->blocks[j][id];
			register_c[i+j]=register_c[i+j]+x*y;
			register_c[i+j+1]=register_c[i+j+1]+(register_c[i+j]>>BLOCK_BIT_SIZE);
			register_c[i+j]=register_c[i+j]&BLOCK_MASK;
		}
	}
	for(int i=0;i<total_len;++i)
	{
		register_c[i+1]=register_c[i+1]+(register_c[i]>>BLOCK_BIT_SIZE);
		c->blocks[i][id]=(int)(register_c[i]&BLOCK_MASK);
	}
	c->real_length[id]=get_real_length(register_c,total_len);
}

__device__ __host__ void bigint_mult(KernelBigInt& a,KernelBigInt& b,KernelBigInt& c)
{
	int a_len=a.real_length;
	int b_len=b.real_length;
	if(a_len==0||b_len==0)
	{
		c.real_length=0;
		return;
	}
	unsigned long long register_c[block_length];
	int total_len=a_len+b_len;
	memset(register_c,0,sizeof(unsigned long long)*(total_len));
	for(int i=0;i<a_len;++i)
	{
		for(int j=0;j<b_len;++j)
		{
			unsigned long long x=(unsigned long long)(unsigned int)a.blocks[i];
			unsigned long long y=(unsigned long long)(unsigned int)b.blocks[j];
			register_c[i+j]=register_c[i+j]+x*y;
			register_c[i+j+1]=register_c[i+j+1]+(register_c[i+j]>>BLOCK_BIT_SIZE);
			register_c[i+j]=register_c[i+j]&BLOCK_MASK;
		}
	}
	for(int i=0;i<total_len;++i)
	{
		register_c[i+1]=register_c[i+1]+(register_c[i]>>BLOCK_BIT_SIZE);
		c.blocks[i]=(int)(register_c[i]&BLOCK_MASK);
	}
	c.real_length=get_real_length(register_c,total_len);
}

__device__ __host__ void bigint_right_shift(int id,KernelBigIntBatch* target,KernelBigIntBatch* result,int bit)
{
	int shift_block=bit/BLOCK_BIT_SIZE;
	int rest_shift=bit%BLOCK_BIT_SIZE;
	int target_len=target->real_length[id];
	unsigned int register_c[block_length];
	for(int i=shift_block;i<target_len;++i)
		register_c[i-shift_block]=(unsigned int)target->blocks[i][id];
	unsigned int low_mask=(1<<rest_shift)-1;
	for(int i=0;i<target_len-shift_block-1;++i)
		register_c[i]=((register_c[i+1]&low_mask)<<(BLOCK_BIT_SIZE-rest_shift))|(register_c[i]>>rest_shift);
	register_c[target_len-shift_block-1]=register_c[target_len-shift_block-1]>>rest_shift;
	for(int i=0;i<target_len-shift_block;++i)
		result->blocks[i][id]=(int)register_c[i];
	result->real_length[id]=get_real_length(register_c,target_len-shift_block);
	result->sign[id]=target->sign[id];
}

__device__ __host__ void bigint_right_shift(int id,KernelBigIntBatch* target,KernelBigIntBatch* result,int* bit)
{
	int shift_block=bit[id]/BLOCK_BIT_SIZE;
	int rest_shift=bit[id]%BLOCK_BIT_SIZE;
	int target_len=target->real_length[id];
	unsigned int register_c[block_length];
	for(int i=shift_block;i<target_len;++i)
		register_c[i-shift_block]=(unsigned int)target->blocks[i][id];
	unsigned int low_mask=(1<<rest_shift)-1;
	for(int i=0;i<target_len-shift_block-1;++i)
		register_c[i]=((register_c[i+1]&low_mask)<<(BLOCK_BIT_SIZE-rest_shift))|(register_c[i]>>rest_shift);
	register_c[target_len-shift_block-1]=register_c[target_len-shift_block-1]>>rest_shift;
	for(int i=0;i<target_len-shift_block;++i)
		result->blocks[i][id]=(int)register_c[i];
	result->real_length[id]=get_real_length(register_c,target_len-shift_block);
	result->sign[id]=target->sign[id];
}

__device__ __host__ void bigint_right_shift(KernelBigInt& target,KernelBigInt& result,int bit)
{
	int shift_block=bit/BLOCK_BIT_SIZE;
	int rest_shift=bit%BLOCK_BIT_SIZE;
	int target_len=target.real_length;
	unsigned int register_c[block_length];
	for(int i=shift_block;i<target_len;++i)
		register_c[i-shift_block]=(unsigned int)target.blocks[i];
	unsigned int low_mask=(1<<rest_shift)-1;
	for(int i=0;i<target_len-shift_block-1;++i)
		register_c[i]=((register_c[i+1]&low_mask)<<(BLOCK_BIT_SIZE-rest_shift))|(register_c[i]>>rest_shift);
	register_c[target_len-shift_block-1]=register_c[target_len-shift_block-1]>>rest_shift;
	for(int i=0;i<target_len-shift_block;++i)
		result.blocks[i]=(int)register_c[i];
	result.real_length=get_real_length(register_c,target_len-shift_block);
	result.sign=target.sign;
}

__device__ __host__ void bigint_left_shift(int id,KernelBigIntBatch* target,KernelBigIntBatch* result,int bit)
{
	int shift_block=bit/BLOCK_BIT_SIZE;
	int rest_shift=bit%BLOCK_BIT_SIZE;
	int target_len=target->real_length[id];
	unsigned long long register_c[block_length];
	for(int i=0;i<shift_block;++i)
		register_c[i]=0;
	for(int i=shift_block;i<target_len+shift_block;++i)
		register_c[i]=(unsigned long long)(unsigned int)target->blocks[i-shift_block][id];
	unsigned long long k=0;
	for(int i=shift_block;i<target_len+shift_block;++i)
	{
		register_c[i]=(register_c[i]<<rest_shift)+k;
		k=register_c[i]>>BLOCK_BIT_SIZE;
		register_c[i]=register_c[i]&BLOCK_MASK;
	}
	for(int i=0;i<target_len+shift_block;++i)
		result->blocks[i][id]=(int)register_c[i];
	register_c[target_len+shift_block]=k;
	result->blocks[target_len+shift_block][id]=(int)k;
	result->real_length[id]=get_real_length(register_c,target_len+shift_block+1);
	result->sign[id]=target->sign[id];
}

__device__ __host__ void bigint_left_shift(int id,KernelBigIntBatch* target,KernelBigIntBatch* result,int* bit)
{
	int shift_block=bit[id]/BLOCK_BIT_SIZE;
	int rest_shift=bit[id]%BLOCK_BIT_SIZE;
	int target_len=target->real_length[id];
	unsigned long long register_c[block_length];
	for(int i=0;i<shift_block;++i)
		register_c[i]=0;
	for(int i=shift_block;i<target_len+shift_block;++i)
		register_c[i]=(unsigned long long)(unsigned int)target->blocks[i-shift_block][id];
	unsigned long long k=0;
	for(int i=shift_block;i<target_len+shift_block;++i)
	{
		register_c[i]=(register_c[i]<<rest_shift)+k;
		k=register_c[i]>>BLOCK_BIT_SIZE;
		register_c[i]=register_c[i]&BLOCK_MASK;
	}
	for(int i=0;i<target_len+shift_block;++i)
		result->blocks[i][id]=(int)register_c[i];
	register_c[target_len+shift_block]=k;
	result->blocks[target_len+shift_block][id]=(int)k;
	result->real_length[id]=get_real_length(register_c,target_len+shift_block+1);
	result->sign[id]=target->sign[id];
}

__device__ __host__ void bigint_left_shift(KernelBigInt& target,KernelBigInt& result,int bit)
{
	int shift_block=bit/BLOCK_BIT_SIZE;
	int rest_shift=bit%BLOCK_BIT_SIZE;
	int target_len=target.real_length;
	unsigned long long register_c[block_length];
	for(int i=0;i<shift_block;++i)
		register_c[i]=0;
	for(int i=shift_block;i<target_len+shift_block;++i)
		register_c[i]=(unsigned long long)(unsigned int)target.blocks[i-shift_block];
	unsigned long long k=0;
	for(int i=shift_block;i<target_len+shift_block;++i)
	{
		register_c[i]=(register_c[i]<<rest_shift)+k;
		k=register_c[i]>>BLOCK_BIT_SIZE;
		register_c[i]=register_c[i]&BLOCK_MASK;
	}
	for(int i=0;i<target_len+shift_block;++i)
		result.blocks[i]=(int)register_c[i];
	result.blocks[target_len+shift_block]=(int)k;
	result.sign=target.sign;
	result.real_length=get_real_length(result.blocks,target_len+shift_block+1);
}

__device__ __host__ inline int compare_abs(const int a[],const int a_len,const int b[],const int b_len)
{
	if(a_len>b_len)
		return 1;
	else if(a_len<b_len)
		return -1;
	else
	{
		for(int i=a_len-1;i>=0;i--)
		{
			long long x=(long long)(unsigned int)a[i];
			long long y=(long long)(unsigned int)b[i];
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

__device__ __host__ inline bool is_bigger_equal(KernelBigInt& a,KernelBigInt& b)
{
	if((a.sign)&&(!b.sign))
		return true;
	else if((!a.sign)&&(b.sign))
		return false;
	else
	{
		int r=compare_abs(a.blocks,a.real_length,b.blocks,b.real_length);
		if(a.sign&&b.sign)
			return r>=0;
		else
			return r<=0;
	}
}

__device__ __host__ inline bool is_bigger(KernelBigInt& a,KernelBigInt& b)
{
	if((a.sign)&&(!b.sign))
		return true;
	else if((!a.sign)&&(b.sign))
		return false;
	else
	{
		int r=compare_abs(a.blocks,a.real_length,b.blocks,b.real_length);
		if(a.sign&&b.sign)
			return r>0;
		else
			return r<0;
	}
}

__device__ __host__ inline bool is_smaller_equal(KernelBigInt& a,KernelBigInt& b)
{
	if((a.sign)&&(!b.sign))
		return false;
	else if((!a.sign)&&(b.sign))
		return true;
	else
	{
		int r=compare_abs(a.blocks,a.real_length,b.blocks,b.real_length);
		if(a.sign&&b.sign)
			return r<=0;
		else
			return r>=0;
	}
}

__device__ __host__ inline bool is_smaller(KernelBigInt& a,KernelBigInt& b)
{
	if((a.sign)&&(!b.sign))
		return false;
	else if((!a.sign)&&(b.sign))
		return true;
	else
	{
		int r=compare_abs(a.blocks,a.real_length,b.blocks,b.real_length);
		if(a.sign&&b.sign)
			return r<0;
		else
			return r>0;
	}
}

__device__ __host__ inline bool is_equal(KernelBigInt& a,KernelBigInt& b)
{
	if(a.sign!=b.sign)
		return false;
	if(a.real_length!=b.real_length)
		return false;
	for(int i=0;i<a.real_length;++i)
	{
		long long x=(long long)(unsigned int)a.blocks[i];
		long long y=(long long)(unsigned int)b.blocks[i];
		if(x!=y)
			return false;
	}
	return true;
}

__device__ __host__ inline int compare_abs(int id,KernelBigIntBatch* a,KernelBigIntBatch* b)
{
	int a_len=a->real_length[id];
	int b_len=b->real_length[id];
	if(a_len>b_len)
		return 1;
	else if(a_len<b_len)
		return -1;
	else
	{
		for(int i=a_len-1;i>=0;i--)
		{
			long long x=(long long)(unsigned int)a->blocks[i][id];
			long long y=(long long)(unsigned int)b->blocks[i][id];
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

__device__ __host__ inline bool is_bigger_equal(int id,KernelBigIntBatch* a,KernelBigIntBatch* b)
{
	if((a->sign[id])&&(!b->sign[id]))
		return true;
	else if((!a->sign[id])&&(b->sign[id]))
		return false;
	else
	{
		int r=compare_abs(id,a,b);
		if(a->sign[id]&&b->sign[id])
			return r>=0;
		else
			return r<=0;
	}
}

__device__ __host__ inline bool is_bigger(int id,KernelBigIntBatch* a,KernelBigIntBatch* b)
{
	if((a->sign[id])&&(!b->sign[id]))
		return true;
	else if((!a->sign[id])&&(b->sign[id]))
		return false;
	else
	{
		int r=compare_abs(id,a,b);
		if(a->sign[id]&&b->sign[id])
			return r>0;
		else
			return r<0;
	}
}

__device__ __host__ inline bool is_smaller_equal(int id,KernelBigIntBatch* a,KernelBigIntBatch* b)
{
	if((a->sign[id])&&(!b->sign[id]))
		return false;
	else if((!a->sign[id])&&(b->sign[id]))
		return true;
	else
	{
		int r=compare_abs(id,a,b);
		if(a->sign[id]&&b->sign[id])
			return r<=0;
		else
			return r>=0;
	}
}

__device__ __host__ inline bool is_smaller(int id,KernelBigIntBatch* a,KernelBigIntBatch* b)
{
	if((a->sign[id])&&(!b->sign[id]))
		return false;
	else if((!a->sign[id])&&(b->sign[id]))
		return true;
	else
	{
		int r=compare_abs(id,a,b);
		if(a->sign[id]&&b->sign[id])
			return r<0;
		else
			return r>0;
	}
}

__device__ __host__ inline bool is_equal(int id,KernelBigIntBatch* a,KernelBigIntBatch* b)
{
	int a_len=a->real_length[id];
	int b_len=b->real_length[id];
	if(a->sign[id]!=b->sign[id])
		return false;
	if(a_len!=b_len)
		return false;
	for(int i=0;i<a_len;++i)
	{
		long long x=(long long)(unsigned int)a->blocks[i][id];
		long long y=(long long)(unsigned int)b->blocks[i][id];
		if(x!=y)
			return false;
	}
	return true;
}

__device__ __host__ int count_leading_zeros(int x)
{
#ifdef __CUDA_ARCH__
	return __clz(x);
#else
	return __lzcnt(x);
#endif
}

__device__ __host__ void tiny_mult(KernelBigInt& a,unsigned int b,KernelBigInt& c)
{
	int a_len=a.real_length;
	if(a_len==0||b==0)
	{
		c.real_length=0;
		return;
	}
	unsigned long long register_c[block_length];
	int total_len=a_len+1;
	memset(register_c,0,sizeof(unsigned long long)*(total_len));
	unsigned long long x,y;
	for(int i=0;i<a_len;++i)
	{
		x=(unsigned long long)(unsigned int)a.blocks[i];
		y=(unsigned long long)(unsigned int)b;
		register_c[i]=register_c[i]+x*y;
		register_c[i+1]=register_c[i+1]+(register_c[i]>>BLOCK_BIT_SIZE);
		register_c[i]=register_c[i]&BLOCK_MASK;
	}
	for(int i=0;i<total_len;++i)
	{
		register_c[i+1]=register_c[i+1]+(register_c[i]>>BLOCK_BIT_SIZE);
		c.blocks[i]=(int)(register_c[i]&BLOCK_MASK);
	}
	c.real_length=get_real_length(register_c,total_len);
	c.sign=a.sign;
}

__device__ __host__ KernelBigInt knuth_div(KernelBigInt& a,KernelBigInt& b)
{
	int b_len=b.real_length;
	int move_block=1;
	int d=count_leading_zeros(b.blocks[b_len-1])+BLOCK_BIT_SIZE*move_block;
	bigint_left_shift(a,a,d);
	bigint_left_shift(b,b,d);
	int u_len=a.real_length;
	int v_len=b.real_length;
	unsigned long long v1=(unsigned long long)(unsigned int)b.blocks[v_len-1];
	unsigned long long v2=(unsigned long long)(unsigned int)b.blocks[v_len-2];
	a.blocks[u_len]=0,u_len=u_len+1;
	a.real_length=u_len;
	int ret_len=u_len-v_len;
	int f_len=v_len+1;
	int head_pos=ret_len-1;
	unsigned long long u0,u1,u2,chunk,q_hat,r_hat;
	KernelBigInt register_f,register_k,register_ret;
	for(int i=head_pos+1;i<head_pos+f_len;++i)
		register_f.blocks[i-head_pos]=a.blocks[i];
	register_f.real_length=f_len;
	register_f.sign=true;
	unsigned int tiny;
	for(int i=0;i<ret_len;++i)
	{
		int head_pos=ret_len-i-1;
		register_f.blocks[0]=a.blocks[head_pos];
		register_f.real_length=get_real_length(register_f.blocks,f_len);
		u0=(unsigned long long)(unsigned int)register_f.blocks[f_len-1];
		u1=(unsigned long long)(unsigned int)register_f.blocks[f_len-2];
		u2=(unsigned long long)(unsigned int)register_f.blocks[f_len-3];
		chunk=(u0<<BLOCK_BIT_SIZE)|u1;
		q_hat=chunk/v1;
		r_hat=chunk-q_hat*v1;
		if(q_hat*v2>((r_hat<<BLOCK_BIT_SIZE)|u2))
			q_hat=q_hat-1;
		tiny=(unsigned int)(q_hat&BLOCK_MASK);
		tiny_mult(b,tiny,register_k);
		if(is_bigger(register_k,register_f))
		{
			q_hat=q_hat-1;
			tiny=(unsigned int)(q_hat&BLOCK_MASK);
			tiny_mult(b,tiny,register_k);
		}
		bigint_subtract(register_f,register_k,register_f);
		for(int j=f_len-1;j>=register_f.real_length;j--)
			register_f.blocks[j]=0;
		for(int j=f_len-1;j>=1;--j)
			register_f.blocks[j]=register_f.blocks[j-1];
		register_ret.blocks[head_pos]=(int)tiny;
	}
	bigint_right_shift(a,a,d);
	bigint_right_shift(b,b,d);
	register_ret.real_length=get_real_length(register_ret.blocks,ret_len);
	register_ret.sign=true;
	return register_ret;
}

__device__ __host__ void tiny_mult(int id,KernelBigIntBatch* a,unsigned int b,KernelBigInt& c)
{
	int a_len=a->real_length[id];
	if(a_len==0||b==0)
	{
		c.real_length=0;
		return;
	}
	unsigned long long register_c[block_length];
	int total_len=a_len+1;
	memset(register_c,0,sizeof(unsigned long long)*(total_len));
	unsigned long long x,y;
	for(int i=0;i<a_len;++i)
	{
		x=(unsigned long long)(unsigned int)a->blocks[i][id];
		y=(unsigned long long)(unsigned int)b;
		register_c[i]=register_c[i]+x*y;
		register_c[i+1]=register_c[i+1]+(register_c[i]>>BLOCK_BIT_SIZE);
		register_c[i]=register_c[i]&BLOCK_MASK;
	}
	for(int i=0;i<total_len;++i)
	{
		register_c[i+1]=register_c[i+1]+(register_c[i]>>BLOCK_BIT_SIZE);
		c.blocks[i]=(int)(register_c[i]&BLOCK_MASK);
	}
	c.real_length=get_real_length(register_c,total_len);
	c.sign=a->sign[id];
}

__device__ __host__ KernelBigInt knuth_div(int id,KernelBigIntBatch* a,KernelBigIntBatch* b)
{
	int b_len=b->real_length[id];
	int move_block=1;
	int d=count_leading_zeros(b->blocks[b_len-1][id])+BLOCK_BIT_SIZE*move_block;
	bigint_left_shift(id,a,a,d);
	bigint_left_shift(id,b,b,d);
	int u_len=a->real_length[id];
	int v_len=b->real_length[id];
	unsigned long long v1=(unsigned long long)(unsigned int)b->blocks[v_len-1][id];
	unsigned long long v2=(unsigned long long)(unsigned int)b->blocks[v_len-2][id];
	a->blocks[u_len][id]=0,u_len=u_len+1;
	a->real_length[id]=u_len;
	int ret_len=u_len-v_len;
	int f_len=v_len+1;
	int head_pos=ret_len-1;
	unsigned long long u0,u1,u2,chunk,q_hat,r_hat;
	KernelBigInt register_f,register_k,register_ret;
	for(int i=head_pos+1;i<head_pos+f_len;++i)
		register_f.blocks[i-head_pos]=a->blocks[i][id];
	register_f.real_length=f_len;
	register_f.sign=true;
	unsigned int tiny;
	for(int i=0;i<ret_len;++i)
	{
		int head_pos=ret_len-i-1;
		register_f.blocks[0]=a->blocks[head_pos][id];
		register_f.real_length=get_real_length(register_f.blocks,f_len);
		u0=(unsigned long long)(unsigned int)register_f.blocks[f_len-1];
		u1=(unsigned long long)(unsigned int)register_f.blocks[f_len-2];
		u2=(unsigned long long)(unsigned int)register_f.blocks[f_len-3];
		chunk=(u0<<BLOCK_BIT_SIZE)|u1;
		q_hat=chunk/v1;
		r_hat=chunk-q_hat*v1;
		if(q_hat*v2>((r_hat<<BLOCK_BIT_SIZE)|u2))
			q_hat=q_hat-1;
		tiny=(unsigned int)(q_hat&BLOCK_MASK);
		tiny_mult(id,b,tiny,register_k);
		if(is_bigger(register_k,register_f))
		{
			q_hat=q_hat-1;
			tiny=(unsigned int)(q_hat&BLOCK_MASK);
			tiny_mult(id,b,tiny,register_k);
		}
		bigint_subtract(register_f,register_k,register_f);
		for(int j=f_len-1;j>=register_f.real_length;j--)
			register_f.blocks[j]=0;
		for(int j=f_len-1;j>=1;--j)
			register_f.blocks[j]=register_f.blocks[j-1];
		register_ret.blocks[head_pos]=(int)tiny;
	}
	bigint_right_shift(id,a,a,d);
	bigint_right_shift(id,b,b,d);
	register_ret.real_length=get_real_length(register_ret.blocks,ret_len);
	register_ret.sign=true;
	return register_ret;
}

__device__ __host__ void bigint_div_mod(KernelBigInt& a,KernelBigInt& b,KernelBigInt& c,bool is_mod)
{
	if(is_smaller(a,b))
	{
		if(is_mod)
		{
			int a_len=a.real_length;
			for(int i=0;i<a_len;++i)
				c.blocks[i]=a.blocks[i];
			c.real_length=a_len;
			c.sign=true;
		}
		else
		{
			c.real_length=0;
		}
		return;
	}
	if(!is_mod)
		c=knuth_div(a,b);
	else
	{
		KernelBigInt register_r=knuth_div(a,b);
		bigint_mult(b,register_r,register_r);
		bigint_subtract(a,register_r,c);
	}
}

__device__ __host__ void bigint_div_mod(int id,KernelBigIntBatch* a,KernelBigIntBatch* b,KernelBigIntBatch* c,bool is_mod)
{
	if(is_smaller(id,a,b))
	{
		if(is_mod)
		{
			int a_len=a->real_length[id];
			for(int i=0;i<a_len;++i)
				c->blocks[i][id]=a->blocks[i][id];
			c->real_length[id]=a_len;
			c->sign[id]=true;
		}
		else
		{
			c->real_length[id]=0;
		}
		return;
	}
	if(!is_mod)
	{
		KernelBigInt register_r=knuth_div(id,a,b);
		c->set_kernel_bigint(id,register_r);
	}
	else
	{
		KernelBigInt register_b,register_a;
		KernelBigInt register_r=knuth_div(id,a,b);
		b->get_kernel_bigint(id,register_b);
		a->get_kernel_bigint(id,register_a);
		bigint_mult(register_b,register_r,register_r);
		bigint_subtract(register_a,register_r,register_r);
		c->set_kernel_bigint(id,register_r);
	}
}

__device__ __host__ void bigint_power_mod(KernelBigInt& a,KernelBigInt& b,KernelBigInt& m,KernelBigInt& c)
{
	KernelBigInt times(b);
	KernelBigInt temp(a);
	c.blocks[0]=1;c.sign=true;c.real_length=1;
	while(times.real_length>0)
	{
		if(times.blocks[0]&1)
		{
			bigint_mult(c,temp,c);
			bigint_div_mod(c,m,c,true);
		}
		bigint_mult(temp,temp,temp);
		bigint_div_mod(temp,m,temp,true);
		bigint_right_shift(times,times,1);
	}
}

__device__ __host__ void bigint_power_mod(KernelBigInt& a,KernelBigInt* b,KernelBigInt& m,KernelBigInt& c)
{
	KernelBigInt times(*b);
	KernelBigInt temp(a);
	c.blocks[0]=1;c.sign=true;c.real_length=1;
	while(times.real_length>0)
	{
		if(times.blocks[0]&1)
		{
			bigint_mult(c,temp,c);
			bigint_div_mod(c,m,c,true);
		}
		bigint_mult(temp,temp,temp);
		bigint_div_mod(temp,m,temp,true);
		bigint_right_shift(times,times,1);
	}
}

__device__ __host__ void bigint_power_mod(KernelBigInt& a,KernelBigInt* b,KernelBigInt* m,KernelBigInt& c)
{
	KernelBigInt times(*b),mod(*m);
	KernelBigInt temp(a);
	c.blocks[0]=1;c.sign=true;c.real_length=1;
	while(times.real_length>0)
	{
		if(times.blocks[0]&1)
		{
			bigint_mult(c,temp,c);
			bigint_div_mod(c,mod,c,true);
		}
		bigint_mult(temp,temp,temp);
		bigint_div_mod(temp,mod,temp,true);
		bigint_right_shift(times,times,1);
	}
}

__device__ __host__ void bigint_power_mod(int id,KernelBigIntBatch* a,KernelBigIntBatch* b,KernelBigIntBatch* m,KernelBigIntBatch* c)
{
	KernelBigInt times,temp,ret,mod;
	b->get_kernel_bigint(id,times);
	a->get_kernel_bigint(id,temp);
	m->get_kernel_bigint(id,mod);
	ret.blocks[0]=1;ret.sign=true;ret.real_length=1;
	while(times.real_length>0)
	{
		if(times.blocks[0]&1)
		{
			bigint_mult(ret,temp,ret);
			bigint_div_mod(ret,mod,ret,true);
		}
		bigint_mult(temp,temp,temp);
		bigint_div_mod(temp,mod,temp,true);
		bigint_right_shift(times,times,1);
	}
	c->set_kernel_bigint(id,ret);
}

__device__ __host__ void bigint_power_mod(int id,KernelBigIntBatch* a,KernelBigInt* b,KernelBigInt* m,KernelBigIntBatch* c)
{
	KernelBigInt times(*b),temp,ret,mod(*m);
	a->get_kernel_bigint(id,temp);
	ret.blocks[0]=1;ret.sign=true;ret.real_length=1;
	while(times.real_length>0)
	{
		if(times.blocks[0]&1)
		{
			bigint_mult(ret,temp,ret);
			bigint_div_mod(ret,mod,ret,true);
		}
		bigint_mult(temp,temp,temp);
		bigint_div_mod(temp,mod,temp,true);
		bigint_right_shift(times,times,1);
	}
	c->set_kernel_bigint(id,ret);
}

__device__ __host__ void bigint_power_mod(int id,KernelBigIntBatch* a,KernelBigInt* b,KernelBigIntBatch* m,KernelBigIntBatch* c)
{
	KernelBigInt times(*b),temp,ret,mod;
	a->get_kernel_bigint(id,temp);
	m->get_kernel_bigint(id,mod);
	ret.blocks[0]=1;ret.sign=true;ret.real_length=1;
	while(times.real_length>0)
	{
		if(times.blocks[0]&1)
		{
			bigint_mult(ret,temp,ret);
			bigint_div_mod(ret,mod,ret,true);
		}
		bigint_mult(temp,temp,temp);
		bigint_div_mod(temp,mod,temp,true);
		bigint_right_shift(times,times,1);
	}
	c->set_kernel_bigint(id,ret);
}