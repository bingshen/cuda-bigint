#pragma once
#include<time.h>
#include<iostream>
#include<chrono>
#include"BigIntUtil.cuh"
#include"BatchCache.h"
#include<random>
#include<gmp.h>

using namespace std;
using namespace chrono;

#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "curand.lib")

typedef std::chrono::high_resolution_clock Clock;

class CaseTest
{
private:
    mt19937 eng{13ULL};
    uniform_int_distribution<unsigned int> distr;
    KernelBigInt* a;
    KernelBigInt* b;
    KernelBigInt* m;
    KernelBigInt* c;
    KernelBigInt* fix_b;
    KernelBigIntBatch* dev_a;
    KernelBigIntBatch* dev_b;
    KernelBigIntBatch* dev_m;
    KernelBigIntBatch* dev_c;

public:
    CaseTest()
    {
        a=new KernelBigInt[BATCH_SIZE];
        b=new KernelBigInt[BATCH_SIZE];
        m=new KernelBigInt[BATCH_SIZE];
        c=new KernelBigInt[BATCH_SIZE];
        dev_a=new KernelBigIntBatch;
        dev_b=new KernelBigIntBatch;
        dev_m=new KernelBigIntBatch;
        dev_c=new KernelBigIntBatch;
    }

    ~CaseTest()
    {
        delete[] a;
        delete[] b;
        delete[] c;
        delete[] dev_a;
        delete[] dev_b;
        delete[] dev_c;
    }

    void random_init(const int a_bit,const int b_bit,const int m_bit)
    {
        Clock::time_point start=Clock::now();
        for(int i=0;i<BATCH_SIZE;++i)
        {
            a[i]=KernelBigInt(KERNEL_BIGINT_RANDOM_BIT,distr,eng,a_bit);
            b[i]=KernelBigInt(KERNEL_BIGINT_RANDOM_BIT,distr,eng,b_bit);
            m[i]=KernelBigInt(KERNEL_BIGINT_RANDOM_BIT,distr,eng,m_bit);
        }
        this->fix_b=new KernelBigInt(KERNEL_BIGINT_RANDOM_BIT,distr,eng,b_bit);
        convert_bigint_batch(a,dev_a,BATCH_SIZE);
        convert_bigint_batch(b,dev_b,BATCH_SIZE);
        convert_bigint_batch(m,dev_m,BATCH_SIZE);
        convert_bigint_batch(c,dev_c,BATCH_SIZE);
        auto end=Clock::now();
        double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()/1000000000.0;
        std::cout<<"random cost:"<<cost_time<<'\n';
    }

    void single_add(const int n);
    void batch_add(const int n);
    void single_subtract(const int n);
    void batch_subtract(const int n);
    void single_mult(const int n);
    void batch_mult(const int n);
    void single_div(const int n);
    void batch_div(const int n);
    void single_mod(const int n);
    void batch_mod(const int n);
    void single_power_mod(const int n);
    void batch_power_mod(const int n);
    void single_fixpower_mod(const int n);
    void batch_fixpower_mod(const int n);
    void generate_prime(const int n);
    void single_rsa_encrypt(const int n);
    void batch_rsa_encrypt(const int n);
    void gmp_add_test(const int n);
    void gmp_subtract_test(const int n);
    void gmp_mult_test(const int n);
    void gmp_div_test(const int n);
    void gmp_mod_test(const int n);
    void gmp_power_mod_test(const int n);
    void gmp_rsa_test(const int n);
};

void CaseTest::single_add(const int n)
{
    Clock::time_point total_start=Clock::now();
    int batch_times=n/BATCH_SIZE+1;
    for(int batch_id=0;batch_id<batch_times;++batch_id)
    {
        int mini_batch_size=min(n-batch_id*BATCH_SIZE,BATCH_SIZE);
        Clock::time_point batch_start=Clock::now();
        auto ret=gpu_add(a,b,c,mini_batch_size);
        auto batch_end=Clock::now();
        double batch_cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(batch_end-batch_start).count()/1000000000.0;
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"row store add total_cost:"<<cost_time<<"s\n";
}

void CaseTest::batch_add(const int n)
{
    Clock::time_point total_start=Clock::now();
    int batch_times=n/BATCH_SIZE+1;
    for(int batch_id=0;batch_id<batch_times;++batch_id)
    {
        int mini_batch_size=min(n-batch_id*BATCH_SIZE,BATCH_SIZE);
        Clock::time_point batch_start=Clock::now();
        auto ret=gpu_add(dev_a,dev_b,dev_c,mini_batch_size);
        auto batch_end=Clock::now();
        double batch_cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(batch_end-batch_start).count()/1000000000.0;
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"colum store add total_cost:"<<cost_time<<"s\n";
}

void CaseTest::single_subtract(const int n)
{
    Clock::time_point total_start=Clock::now();
    int batch_times=n/BATCH_SIZE+1;
    for(int batch_id=0;batch_id<batch_times;++batch_id)
    {
        int mini_batch_size=min(n-batch_id*BATCH_SIZE,BATCH_SIZE);
        Clock::time_point batch_start=Clock::now();
        auto ret=gpu_subtract(a,b,c,mini_batch_size);
        auto batch_end=Clock::now();
        double batch_cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(batch_end-batch_start).count()/1000000000.0;
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"row store subtract total_cost:"<<cost_time<<"s\n";
}

void CaseTest::batch_subtract(const int n)
{
    Clock::time_point total_start=Clock::now();
    int batch_times=n/BATCH_SIZE+1;
    for(int batch_id=0;batch_id<batch_times;++batch_id)
    {
        int mini_batch_size=min(n-batch_id*BATCH_SIZE,BATCH_SIZE);
        Clock::time_point batch_start=Clock::now();
        auto ret=gpu_subtract(dev_a,dev_b,dev_c,mini_batch_size);
        auto batch_end=Clock::now();
        double batch_cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(batch_end-batch_start).count()/1000000000.0;
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"colum store subtract total_cost:"<<cost_time<<"s\n";
}

void CaseTest::single_mult(const int n)
{
    Clock::time_point total_start=Clock::now();
    int batch_times=n/BATCH_SIZE+1;
    for(int batch_id=0;batch_id<batch_times;++batch_id)
    {
        int mini_batch_size=min(n-batch_id*BATCH_SIZE,BATCH_SIZE);
        Clock::time_point batch_start=Clock::now();
        auto ret=gpu_mult(a,b,c,mini_batch_size);
        auto batch_end=Clock::now();
        double batch_cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(batch_end-batch_start).count()/1000000000.0;
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"row store multiply total_cost:"<<cost_time<<"s\n";
}

void CaseTest::batch_mult(const int n)
{
    Clock::time_point total_start=Clock::now();
    int batch_times=n/BATCH_SIZE+1;
    for(int batch_id=0;batch_id<batch_times;++batch_id)
    {
        int mini_batch_size=min(n-batch_id*BATCH_SIZE,BATCH_SIZE);
        Clock::time_point batch_start=Clock::now();
        auto ret=gpu_mult(dev_a,dev_b,dev_c,mini_batch_size);
        auto batch_end=Clock::now();
        double batch_cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(batch_end-batch_start).count()/1000000000.0;
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"colum store multiply total_cost:"<<cost_time<<"s\n";
}

void CaseTest::single_div(const int n)
{
    Clock::time_point total_start=Clock::now();
    int batch_times=n/BATCH_SIZE+1;
    for(int batch_id=0;batch_id<batch_times;++batch_id)
    {
        int mini_batch_size=min(n-batch_id*BATCH_SIZE,BATCH_SIZE);
        Clock::time_point batch_start=Clock::now();
        auto ret=gpu_div(a,b,c,mini_batch_size);
        auto batch_end=Clock::now();
        double batch_cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(batch_end-batch_start).count()/1000000000.0;
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"row store divide total_cost:"<<cost_time<<"s\n";
}

void CaseTest::batch_div(const int n)
{
    Clock::time_point total_start=Clock::now();
    int batch_times=n/BATCH_SIZE+1;
    for(int batch_id=0;batch_id<batch_times;++batch_id)
    {
        int mini_batch_size=min(n-batch_id*BATCH_SIZE,BATCH_SIZE);
        Clock::time_point batch_start=Clock::now();
        auto ret=gpu_div(dev_a,dev_b,dev_c,mini_batch_size);
        auto batch_end=Clock::now();
        double batch_cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(batch_end-batch_start).count()/1000000000.0;
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"colum store divide total_cost:"<<cost_time<<"s\n";
}

void CaseTest::single_mod(const int n)
{
    Clock::time_point total_start=Clock::now();
    int batch_times=n/BATCH_SIZE+1;
    for(int batch_id=0;batch_id<batch_times;++batch_id)
    {
        int mini_batch_size=min(n-batch_id*BATCH_SIZE,BATCH_SIZE);
        Clock::time_point batch_start=Clock::now();
        auto ret=gpu_mod(a,b,c,mini_batch_size);
        auto batch_end=Clock::now();
        double batch_cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(batch_end-batch_start).count()/1000000000.0;
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"row store mod total_cost:"<<cost_time<<"s\n";
}

void CaseTest::batch_mod(const int n)
{
    Clock::time_point total_start=Clock::now();
    int batch_times=n/BATCH_SIZE+1;
    for(int batch_id=0;batch_id<batch_times;++batch_id)
    {
        int mini_batch_size=min(n-batch_id*BATCH_SIZE,BATCH_SIZE);
        Clock::time_point batch_start=Clock::now();
        auto ret=gpu_mod(dev_a,dev_b,dev_c,mini_batch_size);
        auto batch_end=Clock::now();
        double batch_cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(batch_end-batch_start).count()/1000000000.0;
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"colum store mod total_cost:"<<cost_time<<"s\n";
}

void CaseTest::single_power_mod(const int n)
{
    Clock::time_point total_start=Clock::now();
    int batch_times=n/BATCH_SIZE+1;
    for(int batch_id=0;batch_id<batch_times;++batch_id)
    {
        int mini_batch_size=min(n-batch_id*BATCH_SIZE,BATCH_SIZE);
        Clock::time_point batch_start=Clock::now();
        auto ret=gpu_power_mod(a,b,m,c,mini_batch_size);
        auto batch_end=Clock::now();
        double batch_cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(batch_end-batch_start).count()/1000000000.0;
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"row store power-mod total_cost:"<<cost_time<<"s\n";
}

void CaseTest::batch_power_mod(const int n)
{
    Clock::time_point total_start=Clock::now();
    int batch_times=n/BATCH_SIZE+1;
    for(int batch_id=0;batch_id<batch_times;++batch_id)
    {
        int mini_batch_size=min(n-batch_id*BATCH_SIZE,BATCH_SIZE);
        Clock::time_point batch_start=Clock::now();
        auto ret=gpu_power_mod(dev_a,dev_b,dev_m,dev_c,mini_batch_size);
        auto batch_end=Clock::now();
        double batch_cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(batch_end-batch_start).count()/1000000000.0;
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"colum store power-mod total_cost:"<<cost_time<<"s\n";
}

void CaseTest::single_fixpower_mod(const int n)
{
    Clock::time_point total_start=Clock::now();
    int batch_times=n/BATCH_SIZE+1;
    for(int batch_id=0;batch_id<batch_times;++batch_id)
    {
        int mini_batch_size=min(n-batch_id*BATCH_SIZE,BATCH_SIZE);
        Clock::time_point batch_start=Clock::now();
        auto ret=gpu_fixpower_mod(a,fix_b,m,c,mini_batch_size);
        auto batch_end=Clock::now();
        double batch_cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(batch_end-batch_start).count()/1000000000.0;
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"row store power-mod total_cost:"<<cost_time<<"s\n";
}

void CaseTest::batch_fixpower_mod(const int n)
{
    Clock::time_point total_start=Clock::now();
    int batch_times=n/BATCH_SIZE+1;
    for(int batch_id=0;batch_id<batch_times;++batch_id)
    {
        int mini_batch_size=min(n-batch_id*BATCH_SIZE,BATCH_SIZE);
        Clock::time_point batch_start=Clock::now();
        auto ret=gpu_fixpower_mod(dev_a,fix_b,dev_m,dev_c,mini_batch_size);
        auto batch_end=Clock::now();
        double batch_cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(batch_end-batch_start).count()/1000000000.0;
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"colum store power-mod total_cost:"<<cost_time<<"s\n";
}

void CaseTest::generate_prime(const int n)
{
    Clock::time_point total_start=Clock::now();
    for(int i=0;i<n;++i)
    {
        KernelBigInt x(KERNEL_BIGINT_RANDOM_PRIME_BIT,distr,eng,1024); 
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"total_cost:"<<cost_time<<"s\n";
}

void CaseTest::single_rsa_encrypt(const int N)
{
    Clock::time_point total_start=Clock::now();
    KernelBigInt* n=new KernelBigInt;
    KernelBigInt* d=new KernelBigInt;
    KernelBigInt* e=new KernelBigInt(65537);
    KernelBigInt::generate_rsa_key(n,d,512,distr,eng);
    int batch_times=N/BATCH_SIZE+1;
    double encrypt_total=0.0;
    double decrypt_total=0.0;

    for(int batch_id=0;batch_id<batch_times;++batch_id)
    {
        int mini_batch_size=min(N-batch_id*BATCH_SIZE,BATCH_SIZE);
        Clock::time_point encrypt_start=Clock::now();
        auto ret1=gpu_fixpower_fixmod(a,e,n,c,mini_batch_size);
        auto encrypt_end=Clock::now();
        double encrypt_cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(encrypt_end-encrypt_start).count()/1000000000.0;
        encrypt_total=encrypt_total+encrypt_cost_time;
        Clock::time_point decrypt_start=Clock::now();
        auto ret2=gpu_fixpower_fixmod(c,d,n,b,mini_batch_size);
        auto decrypt_end=Clock::now();
        double decrypt_cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(decrypt_end-decrypt_start).count()/1000000000.0;
        decrypt_total=decrypt_total+decrypt_cost_time;
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"row store encrypt_total:"<<encrypt_total<<"s\n";
    std::cout<<"row store decrypt_total:"<<decrypt_total<<"s\n";
    std::cout<<"row store total_cost:"<<cost_time<<"s\n";
}

void CaseTest::batch_rsa_encrypt(const int N)
{
    Clock::time_point total_start=Clock::now();
    KernelBigInt* n=new KernelBigInt;
    KernelBigInt* d=new KernelBigInt;
    KernelBigInt* e=new KernelBigInt(65537);
    KernelBigInt::generate_rsa_key(n,d,512,distr,eng);
    int batch_times=N/BATCH_SIZE+1;
    double encrypt_total=0.0;
    double decrypt_total=0.0;

    for(int batch_id=0;batch_id<batch_times;++batch_id)
    {
        int mini_batch_size=min(N-batch_id*BATCH_SIZE,BATCH_SIZE);
        Clock::time_point encrypt_start=Clock::now();
        auto ret1=gpu_fixpower_fixmod(dev_a,e,n,dev_c,mini_batch_size);
        auto encrypt_end=Clock::now();
        double encrypt_cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(encrypt_end-encrypt_start).count()/1000000000.0;
        encrypt_total=encrypt_total+encrypt_cost_time;
        Clock::time_point decrypt_start=Clock::now();
        auto ret2=gpu_fixpower_fixmod(dev_c,d,n,dev_b,mini_batch_size);
        auto decrypt_end=Clock::now();
        double decrypt_cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(decrypt_end-decrypt_start).count()/1000000000.0;
        decrypt_total=decrypt_total+decrypt_cost_time;
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"colum store encrypt_total:"<<encrypt_total<<"s\n";
    std::cout<<"colum store decrypt_total:"<<decrypt_total<<"s\n";
    std::cout<<"colum store total_cost:"<<cost_time<<"s\n";
}

void CaseTest::gmp_add_test(const int n)
{
    auto total_start=Clock::now();
    gmp_randstate_t state;
    gmp_randinit_default(state);
    gmp_randseed_ui(state,distr(eng));
    mpz_t p,q,r;
    mpz_init(p);
    mpz_init(q);
    mpz_init(r);
    mpz_urandomb(p,state,2048);
    mpz_urandomb(q,state,2048);
    for(int i=0;i<n;i++)
    {
        mpz_add(r,p,q);
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"gmp add cost:"<<cost_time<<"s\n";
}

void CaseTest::gmp_subtract_test(const int n)
{
    auto total_start=Clock::now();
    gmp_randstate_t state;
    gmp_randinit_default(state);
    gmp_randseed_ui(state,distr(eng));
    mpz_t p,q,r;
    mpz_init(p);
    mpz_init(q);
    mpz_init(r);
    mpz_urandomb(p,state,2048);
    mpz_urandomb(q,state,2048);
    for(int i=0;i<n;i++)
    {
        mpz_add(r,p,q);
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"gmp subtract cost:"<<cost_time<<"s\n";
}

void CaseTest::gmp_mult_test(const int n)
{
    auto total_start=Clock::now();
    gmp_randstate_t state;
    gmp_randinit_default(state);
    gmp_randseed_ui(state,distr(eng));
    mpz_t p,q,r;
    mpz_init(p);
    mpz_init(q);
    mpz_init(r);
    mpz_urandomb(p,state,2048);
    mpz_urandomb(q,state,2048);
    for(int i=0;i<n;i++)
    {
        mpz_mul(r,p,q);
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"gmp mult cost:"<<cost_time<<"s\n";
}

void CaseTest::gmp_div_test(const int n)
{
    auto total_start=Clock::now();
    gmp_randstate_t state;
    gmp_randinit_default(state);
    gmp_randseed_ui(state,distr(eng));
    mpz_t p,q,r;
    mpz_init(p);
    mpz_init(q);
    mpz_init(r);
    mpz_urandomb(p,state,2048);
    mpz_urandomb(q,state,512);
    for(int i=0;i<n;i++)
    {
        mpz_div(r,p,q);
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"gmp div cost:"<<cost_time<<"s\n";
}

void CaseTest::gmp_mod_test(const int n)
{
    auto total_start=Clock::now();
    gmp_randstate_t state;
    gmp_randinit_default(state);
    gmp_randseed_ui(state,distr(eng));
    mpz_t p,q,r;
    mpz_init(p);
    mpz_init(q);
    mpz_init(r);
    mpz_urandomb(p,state,2048);
    mpz_urandomb(q,state,512);
    for(int i=0;i<n;i++)
    {
        mpz_mod(r,p,q);
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"gmp mod cost:"<<cost_time<<"s\n";
}

void CaseTest::gmp_power_mod_test(const int n)
{
    auto total_start=Clock::now();
    gmp_randstate_t state;
    gmp_randinit_default(state);
    gmp_randseed_ui(state,distr(eng));
    mpz_t a,p,m,r;
    mpz_init(a);
    mpz_init(p);
    mpz_init(m);
    mpz_init(r);
    mpz_urandomb(a,state,1024);
    mpz_urandomb(p,state,1024);
    mpz_urandomb(m,state,1024);
    for(int i=0;i<n;i++)
    {
        mpz_powm(r,a,p,m);
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"gmp power_mod cost:"<<cost_time<<"s\n";
}

void CaseTest::gmp_rsa_test(const int n)
{
    Clock::time_point total_start=Clock::now();
    gmp_randstate_t state;
    gmp_randinit_default(state);
    gmp_randseed_ui(state,distr(eng));
    mpz_t p,q,N,phi,D,e,one,a,c,r;
    mpz_init(p);
    mpz_init(q);
    mpz_init(phi);
    mpz_init(N);
    mpz_init(D);
    mpz_init(a);
    mpz_init(c);
    mpz_init(r);
    mpz_init_set_ui(one,1);
    mpz_init_set_ui(e,65537);
    mpz_urandomb(a,state,16);
    mpz_urandomb(p,state,512);
    mpz_nextprime(p,p);
    mpz_urandomb(q,state,512);
    mpz_nextprime(q,q);
    mpz_mul(N,p,q);
    mpz_sub(p,p,one);
    mpz_sub(q,q,one);
    mpz_mul(phi,p,q);
    mpz_invert(D,e,phi);
    double encrypt_total=0.0;
    double decrypt_total=0.0;

    for(int i=0;i<n;i++)
    {
        auto encrypt_start=Clock::now();
        mpz_powm(c,a,e,N);
        auto encrypt_end=Clock::now();
        double encrypt_cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(encrypt_end-encrypt_start).count()/1000000000.0;
        encrypt_total=encrypt_total+encrypt_cost_time;
        auto decrypt_start=Clock::now();
        mpz_powm(r,c,D,N);
        auto decrypt_end=Clock::now();
        double decrypt_cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(decrypt_end-decrypt_start).count()/1000000000.0;
        decrypt_total=decrypt_total+decrypt_cost_time;
    }
    auto total_end=Clock::now();
    double cost_time=std::chrono::duration_cast<std::chrono::nanoseconds>(total_end-total_start).count()/1000000000.0;
    std::cout<<"gmp encrypt_total:"<<encrypt_total<<"s\n";
    std::cout<<"gmp decrypt_total:"<<decrypt_total<<"s\n";
    std::cout<<"gmp total_cost:"<<cost_time<<"s\n";
}