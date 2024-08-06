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

__global__ void directConv2D(const float* input, const float* kernel, float* output,
                             int height, int width,int out_height,int out_width,int stride, int padding) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < out_width && out_y < out_height) {
        float result = 0.0f;
        for (int c = 0; c < 3; ++c) {  // For each channel
            for (int i = 0; i < 3; ++i) {
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

int main(int argc, char** argv) {
    constexpr int BLOCK_DIM = 64;
    srand(time(NULL));
    double iStart,iElaps;    // 时间记录
    
    int height = 4096;
    int width = 4096;
    int stride[3] = {1,2,3};
    int padding[3] = {0,1,1};
    int out_width[3],out_height[3],output_size[3];
    int input_size = height * width * 3;
    int kernel_size = 3 * 3 * 3;
    for(int i=0;i<3;i++){
        out_width[i] = (width + 2 * padding[i] - 3) / stride[i] + 1;
        out_height[i] = (height + 2 * padding[i] - 3) / stride[i] + 1;
        output_size[i] = out_width[i] * out_height[i];
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
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output_1, output_size[0] * sizeof(float));
    cudaMalloc(&d_output_2, output_size[1] * sizeof(float));
    cudaMalloc(&d_output_3, output_size[2] * sizeof(float));
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid_1((out_width[0]+ BLOCK_DIM - 1) / BLOCK_DIM,(out_height[0]+ BLOCK_DIM - 1) / BLOCK_DIM);
    dim3 grid_2((out_width[1]+ BLOCK_DIM - 1) / BLOCK_DIM,(out_height[1]+ BLOCK_DIM - 1) / BLOCK_DIM);
    dim3 grid_3((out_width[2]+ BLOCK_DIM - 1) / BLOCK_DIM,(out_height[2]+ BLOCK_DIM - 1) / BLOCK_DIM);
    // 开始时间
    iStart=cpuSecond();
    // 第一个 kernel
    cudaMemcpy(d_kernel, h_kernel_1, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    directConv2D<<<grid_1, block>>>(d_input, d_kernel, d_output_1, height, width,out_height[0],out_width[0], stride[0], padding[0]);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_1, d_output_1, output_size[0] * sizeof(float), cudaMemcpyDeviceToHost);
    // 第二个 kernel
    cudaMemcpy(d_kernel, h_kernel_2, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    directConv2D<<<grid_2, block>>>(d_input, d_kernel, d_output_2, height, width,out_height[1],out_width[1],stride[1], padding[1]);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_2, d_output_2, output_size[1] * sizeof(float), cudaMemcpyDeviceToHost);
    // 第三个 kernel
    cudaMemcpy(d_kernel, h_kernel_3, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    directConv2D<<<grid_3, block>>>(d_input, d_kernel, d_output_3, height, width,out_height[2],out_width[2],stride[2], padding[2]);
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
    
    cudaDeviceReset();

    return 0;
}