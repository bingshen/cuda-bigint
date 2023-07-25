#pragma once
#include<string.h>
#include<stdio.h>
#include<stdlib.h>
#include"ShareMemoryManage.h"
#include<curand.h>
#include<curand_kernel.h>
#include<random>
#include"BigIntDeviceFunction.h"
#include<gmp.h>
#include<cstdlib>
#include<ctime>

const int BLOCK_BIT_SIZE=32;
const long long BLOCK_MASK=0xFFFFFFFFLL;
const long long BLOCK_MAX=0x100000000LL;
const int bit_length=4096;
const int block_length=160;

enum KERNEL_BIGINT_RANDOM_TYPE
{
	KERNEL_BIGINT_RANDOM_BIT,
	KERNEL_BIGINT_RANDOM_PRIME_BIT
};

__device__ __host__ int get_real_length(const unsigned int a[],const int real_length)
{
	for(int i=real_length-1;i>=0;i--)
		if(a[i]!=0)
			return i+1;
	return 0;
}

__device__ __host__ int get_real_length(const unsigned long long a[],const int real_length)
{
	for(int i=real_length-1;i>=0;i--)
		if(a[i]!=0)
			return i+1;
	return 0;
}

__device__ __host__ int get_real_length(const int a[],const int real_length)
{
	for(int i=real_length-1;i>=0;i--)
		if(a[i]!=0)
			return i+1;
	return 0;
}

struct KernelBigInt:public ShareMemoryManage
{
	bool sign;
	int real_length;
	int blocks[block_length];

	__device__ __host__ constexpr char get_dights(int i)
	{
		char DIGHTS[38]="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
		return DIGHTS[i];
	}

	__host__ void to_string(char* str,const int blocks[],const bool sign,const int radix,const int real_length)
	{
		int len=0;
		if(!sign)
			str[len++]='-';
		unsigned int* rest=new unsigned int[real_length];
		unsigned int* target=new unsigned int[real_length];
		for(int i=0;i<real_length;++i)
			target[i]=(unsigned int)blocks[i];
		while(true)
		{
			unsigned long long temp=0;
			unsigned long long rest_dight_sum=0;
			for(int i=real_length-1;i>=0;i--)
			{
				temp=(temp<<BLOCK_BIT_SIZE)|target[i];
				rest[i]=(unsigned int)(temp/radix);
				temp=temp%radix;
				rest_dight_sum=rest_dight_sum+rest[i];
			}
			str[len++]=get_dights((int)temp);
			if(rest_dight_sum==0)
				break;
			memcpy(target,rest,sizeof(unsigned int)*real_length);
		}
		int start_pos=0;
		if(!sign)
			start_pos=1;
		for(int i=start_pos;i<(len+1)/2;++i)
		{
			char x=str[len-i-1+start_pos];
			str[len-i-1+start_pos]=str[i];
			str[i]=x;
		}
		str[len++]='\0';
		delete[] target;
		delete[] rest;
	}

	__host__ void print(int radix=10)
	{
		char str[bit_length];
		to_string(str,blocks,sign,radix,real_length);
		printf("%d------>%s\n",real_length,str);
	}

	__host__ __device__ void print_blocks() const
	{
		if(!sign)
			printf("- ");
		for(int i=0;i<real_length;++i)
			printf("%u,",this->blocks[i]);
		printf("\n");
	}

	__device__ __host__ unsigned char get_dight_num(char x)
	{
		if(x>='0'&&x<='9')
			return x-'0';
		else if(x>='A'&&x<='Z')
			return x-'A'+10;
		else
			return x-'a'+10;
	}

	__device__ __host__ int cuda_strlen(const char* str)
	{
		int len=0;
		if(str==NULL)
			return len;
		while(str[len])
			len++;
		return len;
	}

	__host__ __device__ int string_init(const char* n,int blocks[],bool& t_sign,const int radix)
	{
		int len=cuda_strlen(n);
		unsigned char target[bit_length];
		unsigned char rest[bit_length];
		int start_pos=0;
		if(n[0]=='-')
		{
			start_pos=1;
			t_sign=false;
		}
		else
			t_sign=true;
		for(int i=start_pos;i<len;++i)
			target[i-start_pos]=get_dight_num(n[i]);
		int block_id=0;
		while(true)
		{
			unsigned long long temp=0;
			unsigned long long rest_dight_sum=0;
			for(int i=0;i<len;++i)
			{
				temp=temp*radix+target[i];
				rest[i]=(int)(temp>>BLOCK_BIT_SIZE);
				temp=temp&BLOCK_MASK;
				rest_dight_sum=rest_dight_sum+rest[i];
			}
			blocks[block_id++]=(int)temp;
			if(rest_dight_sum==0)
				break;
			memcpy(target,rest,sizeof(char)*len);
		}
		return block_id;
	}

	__host__ int random_blocks(const int bit,std::uniform_int_distribution<unsigned int> &distr,std::mt19937 &eng)
	{
		int size=bit/32;
		int remainingBits=bit%32;
		memset(blocks,0,sizeof(int)*(size+1));
		for(int i=0;i<size;++i)
			blocks[i]=(int)distr(eng);
		if(remainingBits!=0)
		{
			unsigned int temp=distr(eng);
			blocks[size]=(int)(temp&((1ll<<remainingBits)-1));
		}
		return size+1;
	}

	__host__ KernelBigInt(KERNEL_BIGINT_RANDOM_TYPE type,std::uniform_int_distribution<unsigned int> &distr,std::mt19937 &eng,const int bit)
	{
		if(type==KERNEL_BIGINT_RANDOM_BIT)
		{
			int len=random_blocks(bit,distr,eng);
			this->real_length=get_real_length(blocks,len);
			this->sign=true;
		}
		else
		{
			gmp_randstate_t state;
			gmp_randinit_default(state);
			gmp_randseed_ui(state,distr(eng));
			mpz_t p;
			mpz_init(p);
			mpz_urandomb(p,state,bit);
			mpz_nextprime(p,p);
			char str[2048];
			mpz_get_str(str,10,p);
			gmp_randclear(state);
			this->real_length=this->string_init(str,this->blocks,this->sign,10);
		}
	}

	__host__ static void generate_rsa_key(KernelBigInt* n,KernelBigInt* d,int bit,std::uniform_int_distribution<unsigned int>& distr,std::mt19937& eng)
	{
		gmp_randstate_t state;
		gmp_randinit_default(state);
		gmp_randseed_ui(state,distr(eng));
		mpz_t p,q,N,phi,D,e,one;
		mpz_init(p);
		mpz_init(q);
		mpz_init(phi);
		mpz_init(N);
		mpz_init(D);
		mpz_init_set_ui(one,1);
		mpz_init_set_ui(e,65537);
		mpz_urandomb(p,state,bit);
		mpz_nextprime(p,p);
		mpz_urandomb(q,state,bit);
		mpz_nextprime(q,q);
		mpz_mul(N,p,q);
		mpz_sub(p,p,one);
		mpz_sub(q,q,one);
		mpz_mul(phi,p,q);
		mpz_invert(D,e,phi);
		char NS[2048],DS[2048];
		mpz_get_str(NS,10,N);
		mpz_get_str(DS,10,D);
		n->real_length=n->string_init(NS,n->blocks,n->sign,10);
		d->real_length=d->string_init(DS,d->blocks,d->sign,10);
	}

	__device__ __host__ int int_init(const long long n,int blocks[],bool& t_sign)
	{
		if(n<0)
			t_sign=false;
		else
			t_sign=true;
		long long extend_n=abs(n);
		int block_id=0;
		while(extend_n)
		{
			blocks[block_id]=(int)(extend_n&BLOCK_MASK);
			extend_n=extend_n>>BLOCK_BIT_SIZE;
			block_id++;
		}
		return block_id;
	}

	__device__ __host__ KernelBigInt(const int n)
	{
		this->real_length=int_init(n,blocks,sign);
	}

	__device__ __host__ KernelBigInt(const long long n)
	{
		this->real_length=int_init(n,blocks,sign);
	}

	__device__ __host__ KernelBigInt()
	{
		this->sign=true;
		this->real_length=0;
	}

	__device__ __host__ void clear()
	{
		this->sign=true;
		this->real_length=0;
	}

	__device__ __host__ KernelBigInt(const int blocks[],const int real_length,const bool sign)
	{
		this->sign=sign;
		this->real_length=real_length;
		memcpy(this->blocks,blocks,sizeof(int)*real_length);
	}

	__device__ __host__ KernelBigInt(const KernelBigInt& n)
	{
		memcpy(this->blocks,n.blocks,sizeof(int)*n.real_length);
		this->sign=n.sign;
		this->real_length=n.real_length;
	}

	__host__ __device__ KernelBigInt(const char* n,int radix=10)
	{
		this->real_length=string_init(n,blocks,sign,radix);
	}

	__device__ __host__ KernelBigInt& operator = (const KernelBigInt& n)
	{
		memcpy(this->blocks,n.blocks,n.real_length*sizeof(int));
		this->sign=n.sign;
		this->real_length=n.real_length;
		return *this;
	}
};