#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#define min(x,y) (((x) < (y)) ? (x) : (y))

struct ThreadData {
    double* A;
    double* B;
    double* C;
    int m;
    int n;
    int k;
    int start;
    int end;
};

void* matrix_multiple_thread(void* arg) {
    struct ThreadData* data = (struct ThreadData*)arg;
    for (int B_i = 0; B_i < data->n; B_i++) {
        for (int A_i = data->start; A_i < data->end; A_i++) {
            data->C[A_i * data->n + B_i] = 0;
            for (int A_j = 0; A_j < data->k; A_j++) {
                data->C[A_i * data->n + B_i] += data->A[A_i * data->k + A_j] * data->B[A_j * data->n + B_i];
            }
        }
    }
    pthread_exit(NULL);
}

void matrix_multiple(double* A, double* B, double* C, int m, int n, int k, int num_threads) {
    pthread_t threads[num_threads];
    struct ThreadData thread_data[num_threads];

    int rows_per_process = m / num_threads;
    int remaining_rows = m % num_threads;

    int offset = 0;
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].A = A;
        thread_data[i].B = B;
        thread_data[i].C = C;
        thread_data[i].m = m;
        thread_data[i].n = n;
        thread_data[i].k = k;
        thread_data[i].start = offset;
        offset += (i < remaining_rows) ? rows_per_process + 1 : rows_per_process;
        thread_data[i].end = offset;
        pthread_create(&threads[i], NULL, matrix_multiple_thread, (void*)&thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

int main(int argc, char* argv[]) {
    double* A, * B, * C;
    int m = 0, n = 0, k = 0, i, j;
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
    if (A == NULL || B == NULL || C == NULL) {
        printf("\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
        free(A);
        free(B);
        free(C);
        return 1;
    }
    for (i = 0; i < (m * k); i++) {
        A[i] = (double)(rand() % 100);
    }

    for (i = 0; i < (k * n); i++) {
        B[i] = (double)(rand() % 100);
    }

    for (i = 0; i < (m * n); i++) {
        C[i] = 0.0;
    }
    double times = 0; int test_time = 10;
    for (int i = 0; i < test_time; i++) {
        clock_t start = clock(); // 开始计时

        matrix_multiple(A, B, C, m, n, k, num_threads);
        clock_t end = clock(); // 结束计时
        double time = (double)(end - start) / CLOCKS_PER_SEC; // 计算耗时
        printf("Time taken: %f seconds\n", time);
        times += time;
    }
    printf("average time taken: %f", times / test_time);
    /*printf("\n Computations completed.\n");

    printf(" Top left corner of matrix A: \n");
    for (i = 0; i < min(m, 6); i++) {
        for (j = 0; j < min(k, 6); j++) {
            printf("%12.0f", A[j + i * k]);
        }
        printf("\n");
    }

    printf("\n Top left corner of matrix B: \n");
    for (i = 0; i < min(k, 6); i++) {
        for (j = 0; j < min(n, 6); j++) {
            printf("%12.0f", B[j + i * n]);
        }
        printf("\n");
    }

    printf("\n Top left corner of matrix C: \n");
    for (i = 0; i < min(m, 6); i++) {
        for (j = 0; j < min(n, 6); j++) {
            printf("%12.5G", C[j + i * n]);
        }
        printf("\n");
    }*/
    
  

    free(A);
    free(B);
    free(C);
    return 0;
}
