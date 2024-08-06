#include <stdio.h>
#include <sys/time.h>
#include <fstream>
#include <vector>
#include <cudnn.h>

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



int main() {

    // Initialize cuDNN
    cudnnHandle_t cudnn;
    double iStart,iElaps;    // 时间记录
    cudnnCreate(&cudnn);

    const int batch_size = 1;
    const int in_channels = 3;
    const int in_height = 256;
    const int in_width = 256;
    const int kernel_height = 3;
    const int kernel_width = 3;
    const int out_channels = 1;
    //const int padding = 1;
    //const int stride = 1;
    int stride[3] = {1,2,3};
    int padding[3] = {0,1,1};
    int out_height[3],out_width[3],output_size[3];
    int input_size = in_height * in_width * in_channels;
    int kernel_size = in_channels * kernel_height * kernel_width;
    for(int i=0;i<3;i++){
        out_width[i] = (in_width + 2 * padding[i] - 3) / stride[i] + 1;
        out_height[i] = (in_height + 2 * padding[i] - 3) / stride[i] + 1;
        output_size[i] = out_width[i] * out_height[i];
    }

    // input 
    cudnnTensorDescriptor_t input_descriptor;
    cudnnCreateTensorDescriptor(&input_descriptor);   
    cudnnSetTensor4dDescriptor(input_descriptor,CUDNN_TENSOR_NHWC,CUDNN_DATA_FLOAT,
                                    batch_size,in_channels,in_height,in_width);
    // kernel
    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnCreateFilterDescriptor(&kernel_descriptor);
    cudnnSetFilter4dDescriptor(kernel_descriptor,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,
                                   out_channels,in_channels,kernel_height,kernel_width);
                                   
    // output_1
    cudnnTensorDescriptor_t output_descriptor_1;
    cudnnCreateTensorDescriptor(&output_descriptor_1);
    cudnnSetTensor4dDescriptor(output_descriptor_1,CUDNN_TENSOR_NHWC,CUDNN_DATA_FLOAT,
                                    batch_size,out_channels,out_height[0],out_width[0]);
    // output_2
    cudnnTensorDescriptor_t output_descriptor_2;
    cudnnCreateTensorDescriptor(&output_descriptor_2);
    cudnnSetTensor4dDescriptor(output_descriptor_2,CUDNN_TENSOR_NHWC,CUDNN_DATA_FLOAT,
                                    batch_size,out_channels,out_height[1],out_width[1]);
    // output_3
    cudnnTensorDescriptor_t output_descriptor_3;
    cudnnCreateTensorDescriptor(&output_descriptor_3);
    cudnnSetTensor4dDescriptor(output_descriptor_3,CUDNN_TENSOR_NHWC,CUDNN_DATA_FLOAT,
                                    batch_size,out_channels,out_height[2],out_width[2]);         
    // convolution_1
    cudnnConvolutionDescriptor_t convolution_descriptor_1;
    cudnnCreateConvolutionDescriptor(&convolution_descriptor_1);
    cudnnSetConvolution2dDescriptor(convolution_descriptor_1,
                                    padding[0], padding[0],
                                    stride[0], stride[0],
                                    1, 1,
                                    CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT);
    // convolution_2
    cudnnConvolutionDescriptor_t convolution_descriptor_2;
    cudnnCreateConvolutionDescriptor(&convolution_descriptor_2);
    cudnnSetConvolution2dDescriptor(convolution_descriptor_2,
                                    padding[1], padding[1],
                                    stride[1], stride[1],
                                    1, 1,
                                    CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT);
    // convolution_3
    cudnnConvolutionDescriptor_t convolution_descriptor_3;
    cudnnCreateConvolutionDescriptor(&convolution_descriptor_3);
    cudnnSetConvolution2dDescriptor(convolution_descriptor_3,
                                    padding[2], padding[2],
                                    stride[2], stride[2],
                                    1, 1,
                                    CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT);
    // convolution algorithm_1
    cudnnConvolutionFwdAlgo_t convolution_algorithm_1;
    cudnnGetConvolutionForwardAlgorithm(cudnn,input_descriptor,kernel_descriptor,convolution_descriptor_1,
                        output_descriptor_1,CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,0,&convolution_algorithm_1);
    // convolution algorithm_2
    cudnnConvolutionFwdAlgo_t convolution_algorithm_2;
    cudnnGetConvolutionForwardAlgorithm(cudnn,input_descriptor,kernel_descriptor,convolution_descriptor_2,
                        output_descriptor_2,CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,0,&convolution_algorithm_2);
    // convolution algorithm_3
    cudnnConvolutionFwdAlgo_t convolution_algorithm_3;
    cudnnGetConvolutionForwardAlgorithm(cudnn,input_descriptor,kernel_descriptor,convolution_descriptor_3,
                        output_descriptor_3,CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,0,&convolution_algorithm_3);
    
    void *d_workspace_1, *d_workspace_2, *d_workspace_3;
    // Get workspace size_1
    size_t workspace_bytes_1 = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn,input_descriptor,kernel_descriptor,convolution_descriptor_1,
                                                output_descriptor_1,convolution_algorithm_1,&workspace_bytes_1);
    cudaMalloc(&d_workspace_1, workspace_bytes_1);
    // Get workspace size_2
    size_t workspace_bytes_2 = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn,input_descriptor,kernel_descriptor,convolution_descriptor_2,
                                                output_descriptor_2,convolution_algorithm_2,&workspace_bytes_2);
    cudaMalloc(&d_workspace_2, workspace_bytes_2);
    // Get workspace size_3
    size_t workspace_bytes_3 = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn,input_descriptor,kernel_descriptor,convolution_descriptor_3,
                                                output_descriptor_3,convolution_algorithm_3,&workspace_bytes_3);
    cudaMalloc(&d_workspace_3, workspace_bytes_3);

    // Allocate memory for input, kernel, and output
    float *d_input, *d_kernel,*d_output_1,*d_output_2,*d_output_3;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output_1, output_size[0] * sizeof(float));
    cudaMalloc(&d_output_2, output_size[1] * sizeof(float));
    cudaMalloc(&d_output_3, output_size[2] * sizeof(float));
    /////////////////到这
    // Initialize input and kernel
    float* h_input = (float*)malloc(input_size * sizeof(float));
    float* h_kernel_1 = (float*)malloc(kernel_size * sizeof(float));
    float* h_kernel_2 = (float*)malloc(kernel_size * sizeof(float));
    float* h_kernel_3 = (float*)malloc(kernel_size * sizeof(float));
    float* h_output_1 = (float*)malloc(output_size[0] * sizeof(float));
    float* h_output_2 = (float*)malloc(output_size[1] * sizeof(float));
    float* h_output_3 = (float*)malloc(output_size[2] * sizeof(float));
    loadMat("input_tensor.bin", h_input, input_size);
    loadMat("kernel_tensor_1.bin", h_kernel_1, kernel_size);
    loadMat("kernel_tensor_2.bin", h_kernel_2, kernel_size);
    loadMat("kernel_tensor_3.bin", h_kernel_3, kernel_size);
    
    // cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel_1, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // Perform convolution
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // 开始时间
    iStart=cpuSecond();

    cudnnConvolutionForward(cudnn,&alpha,
                                input_descriptor,d_input,
                                kernel_descriptor,d_kernel,
                                convolution_descriptor_1,convolution_algorithm_1,
                                d_workspace_1,workspace_bytes_1,
                                &beta,output_descriptor_1,d_output_1);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_1, d_output_1, output_size[0] * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_kernel, h_kernel_2, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    cudnnConvolutionForward(cudnn,&alpha,
                                input_descriptor,d_input,
                                kernel_descriptor,d_kernel,
                                convolution_descriptor_2,convolution_algorithm_2,
                                d_workspace_2,workspace_bytes_2,
                                &beta,output_descriptor_2,d_output_2);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_2, d_output_2, output_size[1] * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_kernel, h_kernel_3, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    cudnnConvolutionForward(cudnn,&alpha,
                                input_descriptor,d_input,
                                kernel_descriptor,d_kernel,
                                convolution_descriptor_3,convolution_algorithm_3,
                                d_workspace_3,workspace_bytes_3,
                                &beta,output_descriptor_3,d_output_3);
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

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output_1);
    cudaFree(d_output_2);
    cudaFree(d_output_3);
    cudaFree(d_workspace_1);
    cudaFree(d_workspace_2);
    cudaFree(d_workspace_3);
    
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor_1);
    cudnnDestroyTensorDescriptor(output_descriptor_2);
    cudnnDestroyTensorDescriptor(output_descriptor_3);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor_1);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor_2);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor_3);
    cudnnDestroy(cudnn);
    
    free(h_input);
    free(h_kernel_1);
    free(h_kernel_2);
    free(h_kernel_3);
    free(h_output_1);
    free(h_output_2);
    free(h_output_3);
    return 0;
}
