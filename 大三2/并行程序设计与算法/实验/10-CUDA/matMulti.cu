#include <stdio.h>
#include <sys/time.h>
double cpuSecond()//获取当前时间,转化为微秒单位
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return((double)tp.tv_sec*1e6+(double)tp.tv_usec);
}

void initialize(double *A,int m,int n){// 随机初始化矩阵
    for (int i = 0; i < (m * n); i++) {
        A[i] = (double)(rand() % 100);
    }
}
void show(double* A,int n){// 展示左上角 n*n 部分矩阵
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            printf("%12.0f", A[i*n + j]);
        }
        printf("\n");
    }
}

__global__ void mat_multi(double *A, double *B,double *C,int m,int n,int k)
{
    int threadId = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < m * k)
    {
        int row = threadId / k;
        int column = threadId % k;
        C[threadId] = 0;
        for (int i = 0; i < n; i++)
        {
            C[threadId] += A[row * n + i] * B[i * k + column];
        }
    }
}

template<int BLOCK_DIM>
__global__ void mat_multi_shared(double *A, double *B, double *C, int m, int n, int k)
{
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
 
    if ((thread_y + block_y * BLOCK_DIM) * n + block_x * BLOCK_DIM + thread_x >= m * n)
        return;
 
    int begin_a = block_y * BLOCK_DIM * n;
    int end_a = begin_a + n;
    int step_a = BLOCK_DIM;
 
    int begin_b = block_x * BLOCK_DIM;
    int step_b = BLOCK_DIM * k;
 
    int sum = 0;
     
    // 一个线程块划分到A的BLOCKDIM个一整行，B的BLOCKDIM个一整列
    for (int index_a = begin_a,index_b = begin_b; 
        index_a < end_a; index_a += step_a, index_b += step_b)
    {
        __shared__ double Block_A[BLOCK_DIM][BLOCK_DIM];
        __shared__ double Block_B[BLOCK_DIM][BLOCK_DIM];
        Block_A[thread_y][thread_x] = A[index_a + thread_y * n + thread_x];
        Block_B[thread_y][thread_x] = B[index_b + thread_y * k + thread_x];
 
        __syncthreads();
 
        for (int j = 0; j < BLOCK_DIM; j++)
        {
            sum += Block_A[thread_y][j] * Block_B[j][thread_x];
        }
 
        __syncthreads();
    }
    C[begin_b + block_y * BLOCK_DIM * k + thread_y * k + thread_x] = sum;
}

template<int BLOCK_DIM,int NUMS_THREAD>
__global__ void mat_multi_shared_v1(double *A, double *B, double *C, int m, int n, int k)
{
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
 
    if (((thread_y + block_y * BLOCK_DIM) * n + block_x * BLOCK_DIM + thread_x)* NUMS_THREAD >= m * n)
        return;
 
    int begin_a = block_y * BLOCK_DIM * n;
    int end_a = begin_a + n;
    int step_a = BLOCK_DIM * NUMS_THREAD;
 
    int begin_b = block_x * BLOCK_DIM;
    int step_b = BLOCK_DIM * k * NUMS_THREAD;
    
    int sum = 0;
     
    // 一个线程块划分到A的BLOCKDIM个一整行，B的BLOCKDIM个一整列
    for (int index_a = begin_a, index_b = begin_b; 
        index_a < end_a; index_a += step_a, index_b += step_b)
    {
        __shared__ double Block_A[BLOCK_DIM][BLOCK_DIM * NUMS_THREAD];
        __shared__ double Block_B[BLOCK_DIM * NUMS_THREAD][BLOCK_DIM];
        for(int i = 0; i < NUMS_THREAD; i++){
            Block_A[thread_y][thread_x* NUMS_THREAD+i] = A[index_a + thread_y * n + thread_x* NUMS_THREAD+i];
            Block_B[thread_y* NUMS_THREAD+i][thread_x] = B[index_b + (thread_y* NUMS_THREAD+i) * k + thread_x];
        }

        __syncthreads(); 
        
        for (int j = 0; j < BLOCK_DIM * NUMS_THREAD; j++){
            sum += Block_A[thread_y][j] * Block_B[j][thread_x];
        }
        __syncthreads();   
    }
    C[begin_b + block_y * BLOCK_DIM * k + thread_y * k + thread_x] = sum;
}


int main(int argc, char** argv) {
    srand(time(NULL));
    constexpr int BLOCK_DIM = 8;
    constexpr int NUMS_THREAD = 8;

    int m,n,k;
    double* A,* B,* C,* A_dev,* B_dev,* C_dev;
    double iStart,iElaps;    // 开始结束时间
    
    printf("Enter values for m,n,k[128, 2048]:");
    scanf("%d%d%d",&m,&n,&k);
    // 分配CPU内存,初始化A,B
    A = (double*)malloc(m * n * sizeof(double));
    B = (double*)malloc(n * k * sizeof(double));
    C = (double*)malloc(m * k * sizeof(double));
    initialize(A,m,n);    // 随机初始化 A
    initialize(B,n,k);    // 随机初始化 B
    printf(" Top left corner of matrix A: \n");
    show(A,6);
    printf(" Top left corner of matrix B: \n");
    show(B,6);
    // 分配GPU内存
    cudaMalloc((void**)&A_dev,m * n * sizeof(double));
    cudaMalloc((void**)&B_dev,k * n * sizeof(double));
    cudaMalloc((void**)&C_dev,m * k * sizeof(double));
    cudaMemcpy(A_dev,A,m * n * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev,B,k * n * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(C_dev,C,m * k * sizeof(double),cudaMemcpyHostToDevice);
    // 初始化结束
    dim3 block(BLOCK_DIM,BLOCK_DIM);
    dim3 grid(m/BLOCK_DIM,k/ BLOCK_DIM);
    // 开始
    int test_times = 1;
    iStart=cpuSecond();
    for(int i = 0; i < test_times; i++){
        //mat_multi<<<grid, block>>>(A_dev,B_dev,C_dev,m,n,k);
        
        //mat_multi_shared<BLOCK_DIM><<<grid, block>>>(A_dev,B_dev,C_dev,m,n,k);
        
        mat_multi_shared_v1<BLOCK_DIM,NUMS_THREAD><<<grid, block>>>(A_dev,B_dev,C_dev,m,n,k);
    }
    // 结束
    cudaDeviceSynchronize();    //同步CPU和GPU
    iElaps=cpuSecond()-iStart;
    printf("Time taken: %f us\n",iElaps/test_times);
    
    cudaMemcpy(C,C_dev,n * m * sizeof(double),cudaMemcpyDeviceToHost);
    printf(" Top left corner of matrix C: \n");
    show(C,6);
    
    // 释放内存
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
    free(A);
    free(B);
    free(C);
    
    cudaDeviceReset();
    
    return 0;
}