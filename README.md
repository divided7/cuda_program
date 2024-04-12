# cuda_program
cuda编程
## 准备工作
* 确保nvcc可用，命令行使用`nvcc --version`，如果正常提示版本则没有问题
* 如果不存在，则安装cuda-tooklit;如果torch可以正常使用cuda大概率是path路径问题，配置一下path（Linux在~/.bashrc中添加`export PATH="/usr/local/cuda/bin:$PATH"`
  需要自行检查一下cuda路径是否正确）
## 1 向量加法
伪代码如下:
```cpp
∕∕ Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
  int i = threadIdx.x; // threadIdx.x 是一个从 0 开始的整数，表示当前线程在其线程块中的索引。当你在一个线程块中启动一个 CUDA 核函数时，threadIdx.x 会为每个线程分配一个唯一的索引。这些索引从 0 开始，逐个增加，直到线程块中的最后一个线程。
  C[i] = A[i] + B[i];
}
int main()
{
  ...
  ∕∕ Kernel invocation with N threads
  VecAdd<<<1, N>>>(A, B, C); // 这里的1是指线程块的数量，N是指每个线程块中的线程数量。在3080Ti上最大线程数量N为1024
  ...
}
```
实际实现参考:
```bash
cd CUDA算子
nvcc -o vecAdd vecAdd.cu && ./vecAdd && rm -rf vecAdd
```
实际测试发现N取值最大为1024，根据GPT回答VecAdd<<<1, N>>>(d_A, d_B, d_C)中的第二个参数N是线程块的大小，第三个参数是线程块内的线程数量。在 CUDA 中，每个线程块的最大大小是由硬件限制的，通常为 1024（这是针对当前（2022年1月）大多数 NVIDIA GPU 的情况）。

### 1.1矩阵加法（线程数为二维）
```cpp
∕∕ Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
  int i = threadIdx.x;
  int j = threadIdx.y;
  C[i][j] = A[i][j] + B[i][j];
}
int main()
{
  ...
  ∕∕ Kernel invocation with one block of N * N * 1 threads
  int numBlocks = 1;
  dim3 threadsPerBlock(N, N);
  MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
  ...
}
```
threadIdx.x是一个 dim3 结构体，包含三个成员 x、y 和 z，分别用于表示线程在 x、y 和 z 维度上的索引。也就是说，即使我计算两个10 \* 10尺寸的矩阵加法，但是调用的cuda是一个1 \* 100形状的计算单元。如果希望以10*10的计算单元形状，需要指定线程块数量10，每个线程块的线程数量为10。
根据查阅nvidia cuda c++ program例子，`void MatAdd(float A[N][N], float B[N][N],float C[N][N])` 需要修改，变量需要是一维的，因为在分配cuda地址的时候用不了二重指针。且找到的矩阵乘法例子也是一维的输入。

### 1.2矩阵加法（线程数为二维，线程块为多线程块）
```cpp
∕∕ Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < N && j < N)
  C[i][j] = A[i][j] + B[i][j];
}
int main()
{
  ...
  ∕∕ Kernel invocation
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(N ∕ threadsPerBlock.x, N ∕ threadsPerBlock.y);
  MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
  ...
}
```
