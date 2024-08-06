## Lab11 - CUDA图像卷积

### 实验要求

**任务一**

通过CUDA实现直接卷积（滑窗法），输入从256增加至4096或者输入从32增加至512.

**输入**：Input和Kernel(3x3)

**问题描述**：用直接卷积的方式对Input进行卷积，这里只需要实现2D, height * width，通道channel(depth)设置为3，Kernel (Filter)大小设置为3 * 3 * 3，个数为3，步幅(stride)分别设置为1，2，3，可能需要通过填充(padding)配合步幅(stride)完成CNN操作。注：实验的卷积操作不需要考虑bias(b)，bias设置为0.

**输出**：输出卷积结果以及计算时间

**任务二**

使用im2col方法结合上次实验实现的GEMM实现卷积操作。输入从256增加至4096或者输入从32增加至512

**输入**：Input和Kernel (Filter)

**问题描述**：用im2col的方式对Input进行卷积，这里只需要实现2D, height*width，通道channel(depth)设置为3，Kernel (Filter)大小设置为3 * 3 * 3，个数为3。 注：实验的卷积操作不需要考虑bias(b)，bias设置为0，步幅(stride)分别设置为1，2，3。

**输出**：卷积结果和时间。

**任务三**

NVIDIA cuDNN是用于深度神经网络的GPU加速库。它强调性能、易用性和低内存开销。

使用cuDNN提供的卷积方法进行卷积操作，记录其相应Input的卷积时间，与自己实现的卷积操作进行比较。如果性能不如cuDNN，用文字描述可能的改进方法。

### 实验过程

**1.任务一**

实现直接卷积，先根据输入图像规模，分配好input和output以及Kernel的内存，为了验证卷积操作的正确性，使用python随机生成输入并使用``torch.nn.functional.conv2d``得到卷积输出，保存为二进制文件，在实现CUDA卷积时则直接读取二进制文件得到图像输入、Kernel输入以及期望卷积输出，python生成样本代码如下：

```python
import torch

def save_tensor(tensor, filename):
    tensor.numpy().astype('float32').tofile(filename)

input_tensor = torch.randn(1, 3, 256, 256)   # (batch_size, channels, height, width)
kernel_tensor = torch.randn(1, 3, 3, 3)    # (out_channels, in_channels, kernel_height, kernel_width)

# Save input tensors to binary files
save_tensor(input_tensor, 'input_tensor.bin')
save_tensor(kernel_tensor, 'kernel_tensor.bin')

# Perform convolution
output_tensor = torch.nn.functional.conv2d(input_tensor, kernel_tensor, stride=1, padding=0)

# Save tensors to binary files
save_tensor(output_tensor, 'output_tensor.bin')
```

注意到output_width = ((input_width + 2 * padding - 3) / stride + 1)，实验中设置input_width = 256、1024、4096，要求使用不同的stride，当使用stride = 1时，padding = 0；stride = 2或3时，padding = 1，能够保证矩阵元素都被计算到，根据stride和padding的不同，得到output_width 不同，在随机生成kernel和调用conv2d计算output需要分为3次。因此初始化数据过程比较冗长，此处不做展示，以下是实现直接卷积部分的代码：

```c
__global__ void directConv2D(const float* input, const float* kernel, float* output,
                             int height, int width,int out_height,int out_width,int stride, int padding) {
    // 每个线程负责输出矩阵的一个元素
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < out_width && out_y < out_height) {
        float result = 0.0f;
        for (int c = 0; c < 3; ++c) {	// 3 channel
            for (int i = 0; i < 3; ++i) {	// kernel size : 3x3 
                for (int j = 0; j < 3; ++j) {
                    int in_x = out_x * stride + i - padding;
                    int in_y = out_y * stride + j - padding;
                    if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                        result += input[(c * height + in_y) * width + in_x] * kernel[(c * 3 + j) * 3 + i];
                    }
                }
            }
        }
        output[out_y * out_width + out_x] = result;
    }
}
```

编译运行指令：

```shell
nvcc matConv.cu -o con
./con
```

运行结果：

![image-20240605170213432](C:\Users\asus\Desktop\大三下\并行\11-CUDA\report.assets\image-20240605170213432.png)

**2.任务二**

使用im2col方法结合上次实验实现的GEMM实现卷积操作，主要过程是将input转为矩阵形式，再使用kernel对矩阵进行乘积，分为两个使用核函数的过程。

使用CUDA将input转为矩阵形式，每个线程负责矩阵的一个元素，计算得到该元素对应input的位置索引，赋值

```c
__global__ void imtocol(const float* input, float* mat,int col_size,
                              int height, int width, int stride, int padding,
                              int output_height, int output_width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int n = index;
    if(index >= col_size)
        return;
    
    // index to input_index
    int w_out = index % output_width;
    index /= output_width;
    int h_out = index % output_height;
    index /= output_height;
    int k_w = index % 3;
    index /= 3;
    int k_h = index % 3;
    index /= 3;
    int c = index;

    int h_in = h_out * stride + k_h - padding;
    int w_in = w_out * stride + k_w - padding;

    mat[n] = (w_in >= 0 && w_in < width && h_in >= 0 && h_in < height) ?
                input[(c * height + h_in) * width + w_in] : 0;
}
```

针对一维矩阵kernel与矩阵乘积的核函数如下

```c
template<int BLOCK_DIM>
__global__ void mat_multi(float *A, float *B, float *C, int m, int n, int k)
{
    int thread_x = threadIdx.x;
    int index = blockIdx.x * BLOCK_DIM + thread_x;  
    // 将数组A放入共享内存中,因此注意 BLOCK_DIM 需大于等于 3*3*3
    __shared__ float s_A[27];
    if(thread_x < 27){
        s_A[thread_x] = A[thread_x];
    }
    __syncthreads();
    if (index >= k)
        return;  
    float sum = 0;
    for(int i = 0 ;i < 27;i++){
        sum += s_A[i] * B[i*k + index];
    }
    C[index] = sum;
}
```

编译运行指令：

```shell
nvcc matIm2col.cu -o col
./col
```

运行结果：

<img src="C:\Users\asus\Desktop\大三下\并行\11-CUDA\report.assets\image-20240606212318550.png" alt="image-20240606212318550" style="zoom: 80%;" />

经实验微调，使用直接卷积时，在``BLOCK_DIM``为64时能够取得相对好性能；使用im2col时，在``BLOCK_DIM_1``为64，``BLOCK_DIM_2``为108时能够取得相对好性能。因此在该条件下对比两种方法的耗时性能，时间单位 us。

| 方式\规模 | 256  | 1024 | 4096  |
| --------- | ---- | ---- | ----- |
| 直接卷积  | 578  | 3584 | 53148 |
| im2col    | 648  | 4843 | 72192 |

**3.任务三**

使用cuDNN的卷积操作进行同样操作（对同一input使用三个kernel设置3个stride得到3个output）

编译运行指令：

```shell
export LD_LIBRARY_PATH=/opt/conda/lib:SLD_LIBRARY_PATH
nvcc matcuDNN.cu -I/opt/conda/include -L/opt/conda/lib -lcudnn -o cud
./cud
```

运行结果：

![image-20240607124514013](C:\Users\asus\Desktop\大三下\并行\11-CUDA\report.assets\image-20240607124514013.png)

经多次运行可以看出，使用cuDNN库实现的卷积操作要明显耗时少于自己实现的两种方法，要提升卷积速度还应该注意内存管理，减少内存分配和释放的开销，减少冗余计算，思考如何更高效地使用共享内存以及让读取数据在内存中对齐，利用缓存优化性能
