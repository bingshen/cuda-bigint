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
	const int N=100000000;
	const int n=1000000;
	CaseTest test;
	test.random_init(2048,2048,512);
	test.single_add(N);
	test.batch_add(N);
	test.single_subtract(N);
	test.batch_subtract(N);
	test.single_mult(N);
	test.batch_mult(N);
	test.random_init(2048,512,512);
	test.single_div(N);
	test.batch_div(N);
	test.single_mod(N);
	test.batch_mod(N);
	test.random_init(1024,1024,1024);
	test.single_fixpower_mod(n);
	test.batch_fixpower_mod(n);
	test.random_init(16,16,16);
	test.single_rsa_encrypt(n);
	test.batch_rsa_encrypt(n);
	return 0;
}