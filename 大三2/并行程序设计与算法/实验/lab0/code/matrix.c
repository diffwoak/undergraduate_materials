#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void matrix_multiple(double* A,double* B,double* C,int m,int n,int k){
    for(int B_i = 0;B_i< n;B_i++){
        double* tmp = (double *)malloc( k*sizeof( double ));
        for(int B_j = 0;B_j < k;B_j++){
            tmp[B_j] = B[B_j*n+B_i];
        }
        for(int A_i = 0; A_i < m;A_i++){
            for(int A_j = 0; A_j < k;A_j++){
                C[A_i*n+B_i]+=A[A_i*k+A_j]*tmp[A_j];
            }
        }
        free(tmp);
    }
}
void matrix_multiple_colum(double* A,double* B,double* C,int m,int n,int k){
    for(int B_i = 0;B_i< n;B_i++){
        for(int A_i = 0; A_i < m;A_i++){
            for(int A_j = 0; A_j < k;A_j++){
                C[A_i*n+B_i]+=A[A_i*k+A_j]*B[A_j*n+B_i];
            }
        }
    }
}

int main()
{
    double *A, *B, *C;
    int m=0, n=0, k=0, i, j;
    srand(time(NULL)); // 初始化随机数种子
    
    while (m < 512 || m > 2048 || k < 512 || k > 2048 || n < 512 || n > 2048){
        printf("Enter values for m, k, n (512-2048): ");
        if(scanf("%d %d %d", &m, &k, &n)){}
    }

    printf (" Initializing data for matrix multiplication C=A*B for matrix \n"
            " A(%ix%i) and matrix B(%ix%i)\n", m, k, k, n);
    A = (double *)malloc( m*k*sizeof( double ));
    B = (double *)malloc( k*n*sizeof( double ) );
    C = (double *)malloc( m*n*sizeof( double ) );
    if (A == NULL || B == NULL || C == NULL) {
      printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      free(A);
      free(B);
      free(C);
      return 1;
    }
    for (i = 0; i < (m*k); i++) {
        A[i] = (double)(rand() % 100);
    }

    for (i = 0; i < (k*n); i++) {
        B[i] = (double)(rand() % 100);
    }

    for (i = 0; i < (m*n); i++) {
        C[i] = 0.0;
    }
    clock_t start = clock(); // 开始计时
    matrix_multiple(A,B,C,m,n,k);
    // matrix_multiple_colum(A,B,C,m,n,k);  // 未优化循环调整循环顺序
    clock_t end = clock(); // 结束计时
    double time = (double)(end - start) / CLOCKS_PER_SEC; // 计算耗时

    printf ("\n Computations completed.\n");

    printf (" Top left corner of matrix A: \n");
    for (i=0; i<min(m,6); i++) {
      for (j=0; j<min(k,6); j++) {
        printf ("%12.0f", A[j+i*k]);
      }
      printf ("\n");
    }

    printf ("\n Top left corner of matrix B: \n");
    for (i=0; i<min(k,6); i++) {
      for (j=0; j<min(n,6); j++) {
        printf ("%12.0f", B[j+i*n]);
      }
      printf ("\n");
    }

    printf ("\n Top left corner of matrix C: \n");
    for (i=0; i<min(m,6); i++) {
      for (j=0; j<min(n,6); j++) {
        printf ("%12.5G", C[j+i*n]);
      }
      printf ("\n");
    }
    printf("\nTime taken: %f seconds\n", time);

    free(A);
    free(B);
    free(C);
    return 0;
}