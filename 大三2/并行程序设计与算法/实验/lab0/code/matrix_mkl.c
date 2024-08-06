#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <time.h>

int main()
{
    double *A, *B, *C;
    int m=0, n=0, k=0, i, j;
    double alpha, beta;
    srand(time(NULL)); // 初始化随机数种子

    while (m < 512 || m > 2048 || k < 512 || k > 2048 || n < 512 || n > 2048);{
        printf("Enter values for m, k, n (512-2048): ");
        scanf("%d %d %d", &m, &k, &n);
    }
    printf (" Initializing data for matrix multiplication C=A*B for matrix \n"
            " A(%ix%i) and matrix B(%ix%i)\n\n", m, k, k, n);
    alpha = 1.0; beta = 0.0;
    A = (double *)mkl_malloc( m*k*sizeof( double ), 64 );
    B = (double *)mkl_malloc( k*n*sizeof( double ), 64 );
    C = (double *)mkl_malloc( m*n*sizeof( double ), 64 );
    if (A == NULL || B == NULL || C == NULL) {
      printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      mkl_free(A);
      mkl_free(B);
      mkl_free(C);
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
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n, k, alpha, A, k, B, n, beta, C, n);
    clock_t end = clock(); // 结束计时
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC; // 计算耗时

    printf ("\n Computations completed.\n\n");

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

    printf("\nTime taken: %f seconds\n", time_spent);
   
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    return 0;
}