#include <stdio.h>
#include <sys/time.h>
#include <fstream>
#include <vector>

double cpuSecond()//获取当前时间,转化为微秒单位
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return((double)tp.tv_sec*1e6+(double)tp.tv_usec);
}

void show(float* A, int n, int show_n){// 展示左上角 n*n 部分矩阵
    for (int i = 0; i < show_n; i++) {
        for (int j = 0; j < show_n; j++) {
            printf("%12.6f", A[i*n + j]);
        }
        printf("\n");
    }
}

void loadMat(const char* filename, float* tensor, int size) { // 导入输入二进制文件input和kernel
    FILE* file = fopen(filename, "rb");
    fread(tensor, sizeof(float), size, file);
    fclose(file);
}

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

template<int BLOCK_DIM>
__global__ void mat_multi(float *A, float *B, float *C, int m, int n, int k)
{
    int thread_x = threadIdx.x;
    int index = blockIdx.x * BLOCK_DIM + thread_x;  
    // 直接将数组A放入共享内存中,因此注意 BLOCK_DIM 需大于等于 3*3*3
    __shared__ float s_A[27];
    if(thread_x < 27){
        s_A[thread_x] = A[thread_x];
    }
    
    __syncthreads();
    
    if (index >= k)
        return;
        
    // B的每个列只使用一遍,因此没必要放入共享内存
    float sum = 0;
    for(int i = 0 ;i < 27;i++){
        sum += s_A[i] * B[i*k + index];
    }
    C[index] = sum;
}


int main(int argc, char** argv) {
    constexpr int BLOCK_DIM_1 = 64;
    constexpr int BLOCK_DIM_2 = 108;
    srand(time(NULL));
    double iStart,iElaps;    // 时间记录
    
    int height = 4096;
    int width = 4096;
    int stride[3] = {1,2,3};
    int padding[3] = {0,1,1};
    int out_width[3],out_height[3],output_size[3];
    int input_size = height * width * 3;
    int kernel_size = 3 * 3 * 3;
    /// add
    int col_width[3],col_size[3];
    int col_height = kernel_size;
    int num_blocks[3],num_block_1[3];
    
    for(int i=0;i<3;i++){
        out_width[i] = (width + 2 * padding[i] - 3) / stride[i] + 1;
        out_height[i] = (height + 2 * padding[i] - 3) / stride[i] + 1;
        output_size[i] = out_width[i] * out_height[i];
        /// add
        col_width[i] = out_height[i] * out_width[i];
        col_size[i] = col_width[i] * col_height;
        num_blocks[i] = (col_size[i] + BLOCK_DIM_1 - 1) / BLOCK_DIM_1;
        num_block_1[i] = (col_width[i] + BLOCK_DIM_2 - 1) / BLOCK_DIM_2;
    }

    float* h_input = (float*)malloc(input_size * sizeof(float));
    float* h_kernel_1 = (float*)malloc(kernel_size * sizeof(float));
    float* h_kernel_2 = (float*)malloc(kernel_size * sizeof(float));
    float* h_kernel_3 = (float*)malloc(kernel_size * sizeof(float));
    float* h_output_1 = (float*)malloc(output_size[0] * sizeof(float));
    float* h_expected_output_1 = (float*)malloc(output_size[0] * sizeof(float));
    float* h_output_2 = (float*)malloc(output_size[1] * sizeof(float));
    float* h_expected_output_2 = (float*)malloc(output_size[1] * sizeof(float));
    float* h_output_3 = (float*)malloc(output_size[2] * sizeof(float));
    float* h_expected_output_3 = (float*)malloc(output_size[2] * sizeof(float));
    //initialize
    loadMat("input_tensor.bin", h_input, input_size);
    loadMat("kernel_tensor_1.bin", h_kernel_1, kernel_size);
    loadMat("kernel_tensor_2.bin", h_kernel_2, kernel_size);
    loadMat("kernel_tensor_3.bin", h_kernel_3, kernel_size);
    loadMat("output_tensor_1.bin", h_expected_output_1, output_size[0]);
    loadMat("output_tensor_2.bin", h_expected_output_2, output_size[1]);
    loadMat("output_tensor_3.bin", h_expected_output_3, output_size[2]);
    float *d_input, *d_kernel, *d_output_1, *d_output_2, *d_output_3;
    float *d_mat_1, *d_mat_2, *d_mat_3;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output_1, output_size[0] * sizeof(float));
    cudaMalloc(&d_output_2, output_size[1] * sizeof(float));
    cudaMalloc(&d_output_3, output_size[2] * sizeof(float));
    /// add
    cudaMalloc(&d_mat_1, col_size[0] * sizeof(float));
    cudaMalloc(&d_mat_2, col_size[1] * sizeof(float));
    cudaMalloc(&d_mat_3, col_size[2] * sizeof(float));
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // 开始时间
    iStart=cpuSecond();
    // 1: d_input 转为 目标矩阵
    imtocol<<<num_blocks[0], BLOCK_DIM_1>>>(d_input,d_mat_1,col_size[0],height,width,stride[0],padding[0],out_height[0],out_width[0]);
    cudaDeviceSynchronize();
    // 2: 同 kernel 矩阵相乘
    cudaMemcpy(d_kernel, h_kernel_1, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    mat_multi<BLOCK_DIM_2><<<num_block_1[0], BLOCK_DIM_2>>>(d_kernel,d_mat_1,d_output_1,1,col_height,col_width[0]);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_1, d_output_1, output_size[0] * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 1: d_input 转为 目标矩阵
    imtocol<<<num_blocks[1], BLOCK_DIM_1>>>(d_input,d_mat_2,col_size[1],height,width,stride[1],padding[1],out_height[1],out_width[1]);
    cudaDeviceSynchronize();
    // 2: 同 kernel 矩阵相乘
    cudaMemcpy(d_kernel, h_kernel_2, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    mat_multi<BLOCK_DIM_2><<<num_block_1[1], BLOCK_DIM_2>>>(d_kernel,d_mat_2,d_output_2,1,col_height,col_width[1]);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_2, d_output_2, output_size[1] * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 1: d_input 转为 目标矩阵
    imtocol<<<num_blocks[2], BLOCK_DIM_1>>>(d_input,d_mat_3,col_size[2],height,width,stride[2],padding[2],out_height[2],out_width[2]);
    cudaDeviceSynchronize();
    // 2: 同 kernel 矩阵相乘
    cudaMemcpy(d_kernel, h_kernel_3, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    mat_multi<BLOCK_DIM_2><<<num_block_1[2], BLOCK_DIM_2>>>(d_kernel,d_mat_3,d_output_3,1,col_height,col_width[2]);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_3, d_output_3, output_size[2] * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 结束时间
    iElaps=cpuSecond()-iStart;
    printf("Time taken: %f us\n",iElaps);
    
    printf(" Top left corner of Conv output_1: \n");
    show(h_output_1,out_width[0],6);
    printf(" Top left corner of Conv output_2: \n");
    show(h_output_2,out_width[1],6);
    printf(" Top left corner of Conv output_3: \n");
    show(h_output_3,out_width[2],6);
    // 检查卷积正确性
    bool correct = true;
    for (size_t i = 0; i < output_size[0]; ++i) {
        if (fabs(h_output_1[i] - h_expected_output_1[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    for (size_t i = 0; i < output_size[1]; ++i) {
        if (fabs(h_output_2[i] - h_expected_output_2[i]) > 1e-5) {
            correct = false;
            fprintf(stderr, "Mismatch at index %zu: %f (CUDA) vs %f (PyTorch)\n", i, h_output_2[i], h_expected_output_2[i]);
            break;
        }
    }
    for (size_t i = 0; i < output_size[2]; ++i) {
        if (fabs(h_output_3[i] - h_expected_output_3[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    if (correct) {
        printf("Convolution result is correct\n");
    } else {
        printf("Convolution result is incorrect\n");
    }
    
    free(h_input);
    free(h_kernel_1);
    free(h_kernel_2);
    free(h_kernel_3);
    free(h_output_1);
    free(h_output_2);
    free(h_output_3);
    free(h_expected_output_1);
    free(h_expected_output_2);
    free(h_expected_output_3);
    
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output_1);
    cudaFree(d_output_2);
    cudaFree(d_output_3);
    cudaFree(d_mat_1);
    cudaFree(d_mat_2);
    cudaFree(d_mat_3);
    
    cudaDeviceReset();

    return 0;
}