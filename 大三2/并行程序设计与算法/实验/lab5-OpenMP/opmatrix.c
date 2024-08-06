#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define min(x,y) (((x) < (y)) ? (x) : (y))


void matrix_multiple(double* A, double* B, double* C, int m, int n, int k, int num_threads) {
#   pragma omp for schedule(dynamic) 
    for (int B_i = 0; B_i < n; B_i++) {
        for (int A_i = 0; A_i < m; A_i++) {
            C[A_i * n + B_i] = 0;
            for (int A_j = 0; A_j < k; A_j++) {
                C[A_i * n + B_i] += A[A_i * k + A_j] * B[A_j * n + B_i];
            }
        }
    }

}

int main(int argc, char* argv[]) {
    double* A, * B, * C;
    int m = 0, n = 0, k = 0;
    int num_threads = strtol(argv[1], NULL, 10);


    srand(time(NULL)); // 初始化随机数种子

    while (m < 128 || m > 2048 || k < 128 || k > 2048 || n < 128 || n > 2048) {
        printf("Enter values for m, k, n (128-2048): ");
        //scanf("%d %d %d", &m, &k, &n);
        scanf("%d", &m);
        k = m;
        n = m;
    }

    printf(" Initializing data for matrix multiplication C=A*B for matrix \n"
        " A(%ix%i) and matrix B(%ix%i)\n", m, k, k, n);
    A = (double*)malloc(m * k * sizeof(double));
    B = (double*)malloc(k * n * sizeof(double));
    C = (double*)malloc(m * n * sizeof(double));

    for (int i = 0; i < (m * k); i++) {
        A[i] = (double)(rand() % 100);
    }

    for (int i = 0; i < (k * n); i++) {
        B[i] = (double)(rand() % 100);
    }

    for (int i = 0; i < (m * n); i++) {
        C[i] = 0.0;
    }
    double times = 0; int test_time = 10;
    for (int i = 0; i < test_time; i++) {
    clock_t start = clock(); // 开始计时
#       pragma omp parallel num_threads(num_threads)
    matrix_multiple(A, B, C, m, n, k, num_threads);
    clock_t end = clock(); // 结束计时
    double time = (double)(end - start) / CLOCKS_PER_SEC; // 计算耗时
    printf("Time taken: %f seconds\n", time);
    times += time;
    }
    printf("average time taken: %f", times / test_time);

    /*printf("\n Computations completed.\n");
    printf(" Top left corner of matrix A: \n");
    for (int i = 0; i < min(m, 6); i++) {
        for (int j = 0; j < min(k, 6); j++) {
            printf("%12.0f", A[j + i * k]);
        }
        printf("\n");
    }

    printf("\n Top left corner of matrix B: \n");
    for (int i = 0; i < min(k, 6); i++) {
        for (int j = 0; j < min(n, 6); j++) {
            printf("%12.0f", B[j + i * n]);
        }
        printf("\n");
    }

    printf("\n Top left corner of matrix C: \n");
    for (int i = 0; i < min(m, 6); i++) {
        for (int j = 0; j < min(n, 6); j++) {
            printf("%12.5G", C[j + i * n]);
        }
        printf("\n");
    }
    */
    
    
    
    

    free(A);
    free(B);
    free(C);
    return 0;
}
