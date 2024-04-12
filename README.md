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
int i = threadIdx.x;
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
