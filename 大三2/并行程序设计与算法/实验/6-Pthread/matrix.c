#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "parallel.h"

#define min(x,y) (((x) < (y)) ? (x) : (y))


struct functor_args {
    double* A, * B, * C;
    int m, k, n;
};

void* matrix_multiple_thread(void* args) {
    struct ThreadData* data = (struct ThreadData*)args;
    struct functor_args* arg = (struct functor_args*)data->arg;
    for (int B_i = 0; B_i < arg->n; B_i++) {
        for (int A_i = data->start; A_i < data->end; A_i++) {
            arg->C[A_i * arg->n + B_i] = 0;
            for (int A_j = 0; A_j < arg->k; A_j++) {
                arg->C[A_i * arg->n + B_i] += arg->A[A_i * arg->k + A_j] * arg->B[A_j * arg->n + B_i];
            }
        }
    }
    pthread_exit(NULL);

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

    struct functor_args args;
    args.A = A;
    args.B = B;
    args.C = C;
    args.m = m;
    args.k = k;
    args.n = n;

    clock_t start = clock(); // 开始计时
    parallel_for(0, m, 1, matrix_multiple_thread, (void*)&args, num_threads);
    clock_t end = clock(); // 结束计时
    double time = (double)(end - start) / CLOCKS_PER_SEC; // 计算耗时
    printf("Time taken: %f seconds\n", time);

    printf("\n Computations completed.\n");
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
    
    free(A);
    free(B);
    free(C);
    return 0;
}
