#include"CaseTest.h"

int main()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	for(int device=0; device<deviceCount; ++device)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp,device);
		cout<<deviceProp.name<<endl;
	}
	int test_num1[]={100000,1000000,10000000,100000000,1000000000};
	int tl1=sizeof(test_num1)/sizeof(int);
	int test_num2[]={100,1000,10000,100000,1000000};
	int tl2=sizeof(test_num1)/sizeof(int);
	CaseTest test;
	test.random_init(2048,2048,512);
	//GPU计算第一次会慢一些，先空转一次，把这个误差消掉
	test.empty_run(1000);
	for(int i=0;i<tl1;i++)
	{
		int N=test_num1[i];
		printf("test num:%d\n",N);
		test.gmp_add_test(N);
		test.single_add(N);
		test.batch_add(N);
		test.gmp_subtract_test(N);
		test.single_subtract(N);
		test.batch_subtract(N);
		test.gmp_mult_test(N);
		test.single_mult(N);
		test.batch_mult(N);
	}
	test.random_init(2048,512,512);
	for(int i=0;i<tl1;i++)
	{
		int N=test_num1[i];
		printf("test num:%d\n",N);
		test.gmp_div_test(N);
		test.single_div(N);
		test.batch_div(N);
		test.gmp_mod_test(N);
		test.single_mod(N);
		test.batch_mod(N);
	}
	test.random_init(1024,1024,1024);
	for(int i=0;i<tl2;i++)
	{
		int n=test_num2[i];
		printf("test num:%d\n",n);
		test.gmp_power_mod_test(n);
		test.single_fixpower_mod(n);
		test.batch_fixpower_mod(n);
	}
	test.random_init(16,16,16);
	for(int i=0;i<tl2;i++)
	{
		int n=test_num2[i];
		printf("test num:%d\n",n);
		test.gmp_rsa_test(n);
		test.single_rsa_encrypt(n);
		test.batch_rsa_encrypt(n);
	}
	return 0;
}