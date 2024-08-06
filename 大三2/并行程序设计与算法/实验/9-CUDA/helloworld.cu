#include <stdio.h>

__global__ void helloFromGPU(){
    printf("Hello World from Thread (%d, %d) in Block %d!\n",threadIdx.x,threadIdx.y,blockIdx.x);
}
int main(int argc, char** argv) {
    int m,n,k;
    printf("Enter values for n, m, k(1,32):\n");
    scanf("%d%d%d",&n,&m,&k);
    dim3 block(m,k);
    dim3 grid(n,1);
    printf("Hello World from the host!\n");
    helloFromGPU<<<grid, block>>>();
    cudaDeviceReset();
    return 0;
}