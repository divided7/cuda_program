#include <stdio.h>
#include <iostream>
using namespace std;
// CUDA kernel function to add the elements of two arrays on the GPU
__global__ void VecAdd(float* A, float* B, float* C)
{
    // Get the index of the thread
    int i = threadIdx.x;
    // Each thread computes one element of C
    // by adding the corresponding elements of A and B
    C[i] = A[i] + B[i];
}

int main()
{
    const int N = 1025; // 1M elements
    std::cout << "N:" << N << std::endl;
    // 这行代码定义了一个常量整数N，它等于2的20次方，即1048576（也就是1百万）。
    // 这里使用位左移操作符<<来快速计算2的幂。N表示向量A、B和C中的元素数量，每个向量都包含1百万个浮点数。
    float *h_A, *h_B, *h_C; // host_A
    // 这行代码定义了三个指向浮点数的指针h_A、h_B和h_C。这些指针用于在主机（CPU）内存中存储向量A、B和C。
    // 通常，这些向量在CPU上初始化，然后复制到GPU上用于并行计算。
    float *d_A, *d_B, *d_C; // device_A
    // 这行代码定义了另外三个指向浮点数的指针d_A、d_B和d_C。这些指针用于在设备（GPU）内存中存储向量A、B和C的副本。
    // 在CUDA编程中，数据通常在主机内存和设备内存之间传输，以便利用GPU进行并行计算。


    // Allocate space for device copies of A, B, C 为向量A、B和C在GPU上的副本分配内存空间。
    // cuda Memory allocation CUDA内存分配
    std::cout << "&d_A:" << &d_A << "(void **)&d_A:" << (void **)&d_A << std::endl; //两个是相同的地址，都是指针 d_A 在 CPU 内存中的地址
    std::cout << "sizeof(float):" << sizeof(float) << std::endl; //返回float的大小，结果是4 （字节）
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaMalloc((void **)&d_C, N * sizeof(float));
    // cudaMalloc是一个 CUDA 函数，用于在 GPU 上分配内存。它允许在 GPU 上为数据分配内存空间。
    // ((void **)&d_A): 这部分是一个类型转换，将指向指针 d_A 的地址的指针传递给 cudaMalloc 函数。这是因为 cudaMalloc 函数期望接收一个指向指针的指针作为其第一个参数，以便将分配的内存块的地址存储在该指针指向的位置。void * 是通用指针类型，因此需要进行类型转换。
    // d_A是一个指向GPU上内存的指针，N * sizeof(float)计算了需要分配的总字节数（每个元素是float类型，总共有N个元素）。
    // 因此，整个行代码的作用是在 GPU 上分配足够大的内存块，以存储 N 个浮点数的数组，并将其地址存储在 d_A 指针所指向的位置。
    // cudaMalloc的返回值（一个cudaError_t类型的值）通常应该被检查以确保内存分配成功。
    // 在这里，代码没有显示检查返回值，但在实际的CUDA程序中，这是很重要的。

    // Allocate space for host copies of A, B, C 这段代码是在主机端（CPU）上为存储浮点数数组的空间动态分配内存。
    h_A = (float *)malloc(N * sizeof(float));
    h_B = (float *)malloc(N * sizeof(float));
    h_C = (float *)malloc(N * sizeof(float));
    // h_A, h_B, h_C：这些变量通常是指向浮点数数组的指针，用于存储在主机内存中的数据。
    // (float *)malloc(N * sizeof(float))：这是用于分配内存的典型方式。malloc 函数用于在堆上分配内存空间。
    // malloc 接受一个参数，即要分配的内存空间的大小（以字节为单位）。
    // N * sizeof(float) 给出了要分配的内存块的大小，其中 N 是浮点数数组的元素数量，sizeof(float) 给出了一个 float 类型的大小。
    // 因此，N * sizeof(float) 表示了要分配的浮点数数组的总字节数。
    // (float *) 是将 malloc 返回的 void * 类型指针转换为指向 float 类型的指针。



    // Initialize h_A and h_B arrays on the host // 这里host指cpu主机
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Copy h_A and h_B to d_A and d_B on the device
    // 步骤类似tensor.to("cuda:0"),cudaMemcpyHostToDevice代表是从cpu->cuda
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch a kernel on the GPU with one thread for each element
    VecAdd<<<1, N>>>(d_A, d_B, d_C); //调用VecAdd函数
    // 这段代码中使用的内核启动配置 VecAdd<<<1, N>>>(d_A, d_B, d_C); 中的第二个参数是线程块的大小，第三个参数是线程块内的线程数量。
    // 在 CUDA 中，每个线程块的最大大小是由硬件限制的，通常为 1024（这是针对当前（2022年1月）大多数 NVIDIA GPU 的情况）。
    // 这就是为什么一旦 N 的值大于 1024 时，会出现错误的原因。
    // Copy result from device back to host
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            fprintf(stderr,"Error: Result verification failed at element %d!\n", i);
            std::cout << h_C[i] << "!=" << h_A[i] << "+" << h_B[i] << std::endl;
            // exit(EXIT_FAILURE);
        }
        else{
            std::cout << "Success!" << h_C[i] << "=" << h_A[i] << "+" << h_B[i] << std::endl;
        }
    }

    printf("Test PASSED! 成功!\n");

    // Free device memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    // Free host memory
    free(h_A); free(h_B); free(h_C);

    return 0;
}