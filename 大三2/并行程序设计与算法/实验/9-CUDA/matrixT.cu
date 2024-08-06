#include <stdio.h>
#include <sys/time.h>
double cpuSecond()//获取当前时间,转化为微秒单位
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return((double)tp.tv_sec*1e6+(double)tp.tv_usec);
}


void initialize(double *A,int n){
    // 随机初始化矩阵 A
    for (int i = 0; i < (n * n); i++) {
        A[i] = (double)(rand() % 100);
    }
}

__global__ void mat_transpose(double *A, double *A_T, int n){
    // 在整个矩阵中的坐标
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (y < n && x < n) {
        A_T[x * n + y] = A[y * n + x];
    }
}
template <int BLOCK_DIM,int NUMS_THREAD>
__global__ void mat_transpose_share(double *A, double *A_T, int n){
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ double sdata[BLOCK_DIM][BLOCK_DIM+1];
    
    int x = bx * BLOCK_DIM + tx;
    int y = by * BLOCK_DIM + ty;
    
    int stride = BLOCK_DIM/NUMS_THREAD;
    
    if(x < n){
        #pragma unroll
        for (int y_off = 0; y_off < BLOCK_DIM; y_off += stride) {
            if (y + y_off < n) {
                sdata[ty + y_off][tx] = A[(y + y_off) * n + x]; 
            }
        }
    }
    
    __syncthreads();

    x = by * BLOCK_DIM + tx;
    y = bx * BLOCK_DIM + ty;
    if (x < n) {
        for (int y_off = 0; y_off < BLOCK_DIM; y_off += stride) {
            if (y + y_off < n) {
                A_T[(y + y_off) * n + x] = sdata[tx][ty + y_off];
            }
        }
    }

}
template <int BLOCK_DIM,int NUMS_THREAD>
__global__ void mat_transpose_share_v1(double *A, double *A_T, int n){
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ double sdata[BLOCK_DIM][BLOCK_DIM+1];
    
    //int stride = BLOCK_DIM/NUMS_THREAD;
    
    int x = bx * BLOCK_DIM + tx;
    int y = by * BLOCK_DIM + ty * NUMS_THREAD;
    
    if(x < n){
        #pragma unroll
        for (int y_off = 0; y_off < NUMS_THREAD; y_off += 1) {
            if (y + y_off < n) {
                sdata[ty + y_off][tx] = A[(y + y_off) * n + x]; 
            }
        }
    }
    
    __syncthreads();

    x = by * BLOCK_DIM + tx * NUMS_THREAD;
    y = bx * BLOCK_DIM + ty;
    if (x < n) {
        for (int y_off = 0; y_off < NUMS_THREAD; y_off += 1) {
            if (y + y_off < n) {
                A_T[(y + y_off) * n + x] = sdata[tx][ty + y_off];
            }
        }
    }

}

int main(int argc, char** argv) {
    int n;
    constexpr int BLOCK_DIM = 32;
    constexpr int NUMS_THREAD = 32;
    double* A,* A_dev,* A_T_dev;
    double iStart,iElaps;    // 开始结束时间
    printf("Enter values for n[512, 2048]:\n");
    scanf("%d",&n);

    A = (double*)malloc(n * n * sizeof(double));
    initialize(A,n);    // 随机初始化 A
    
    printf(" Top left corner of matrix A: \n");
    for (int i = n - 1; i >= n - 6; i--) {
        for (int j = n - 1; j >= n - 6; j--) {
            printf("%12.0f", A[i*n + j]);
        }
        printf("\n");
    }
    
    cudaMalloc((void**)&A_dev,n * n * sizeof(double));
    cudaMalloc((void**)&A_T_dev,n * n * sizeof(double));
    cudaMemcpy(A_dev,A,n * n * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(A_T_dev,A,n * n * sizeof(double),cudaMemcpyHostToDevice);
    
    //dim3 block(BLOCK_DIM,BLOCK_DIM);
    dim3 block(BLOCK_DIM,BLOCK_DIM/NUMS_THREAD);  // 线程划分多数据
    dim3 grid((n + BLOCK_DIM - 1) / BLOCK_DIM,(n + BLOCK_DIM - 1) / BLOCK_DIM);
    // 开始
    int test_times = 10000;
    iStart=cpuSecond();
    for(int i = 0; i < test_times; i++){
        //mat_transpose<<<grid, block>>>(A_dev,A_T_dev,n);
        //mat_transpose_share<BLOCK_DIM,NUMS_THREAD><<<grid, block>>>(A_dev,A_T_dev,n);
        mat_transpose_share_v1<BLOCK_DIM,NUMS_THREAD><<<grid, block>>>(A_dev,A_T_dev,n);
    }
    // 结束
    cudaDeviceSynchronize();    //同步CPU和GPU
    iElaps=cpuSecond()-iStart;
    printf("Time taken: %f us\n",iElaps/test_times);
    
    // 打印转置矩阵
    cudaMemcpy(A,A_T_dev,n * n * sizeof(double),cudaMemcpyDeviceToHost);
    printf(" Top left corner of matrix A_T: \n");
    for (int i = n - 1; i >= n - 6; i--) {
        for (int j = n - 1; j >= n - 6; j--) {
            printf("%12.0f", A[i*n + j]);
        }
        printf("\n");
    }
    
    // 释放内存
    cudaFree(A_dev);
    cudaFree(A_T_dev);
    free(A);
    
    cudaDeviceReset();
    
    return 0;
}