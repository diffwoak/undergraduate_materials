#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

struct ShareData {
    int m, n, k;
};

void mpi_multiple(double* A, double* B, double* C, int m, int n, int k) {
    for (int B_i = 0; B_i < n; B_i++) {
        for (int A_i = 0; A_i < m; A_i++) {
            C[A_i * n + B_i] = 0;
            for (int A_j = 0; A_j < k; A_j++) {
                C[A_i * n + B_i] += A[A_i * k + A_j] * B[B_i * k + A_j];
            }
        }
    }
}

void matrix_transpose(double* B, double* B_T, int k, int n) {
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            B_T[j * k + i] = B[i * n + j];
        }
    }
}

int main() {
    int comm_sz;
    int my_rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    struct ShareData temp;
    MPI_Datatype types[3] = {MPI_INT,MPI_INT,MPI_INT };
    int block_lengths[3] = { 1, 1, 1 };
    MPI_Aint address[3], startaddr;
    MPI_Aint displacements[3];

    MPI_Get_address(&temp, &startaddr);
    MPI_Get_address(&temp.m, &address[0]);
    MPI_Get_address(&temp.n, &address[1]);
    MPI_Get_address(&temp.k, &address[2]);
    displacements[0] = address[0] - startaddr;
    displacements[1] = address[1] - startaddr;
    displacements[2] = address[2] - startaddr;

    MPI_Datatype data_type;
    MPI_Type_create_struct(3, block_lengths, displacements, types, &data_type);
    MPI_Type_commit(&data_type);

    if (my_rank == 0) {
        double* A, * B, * C, * B_T;
        int m = 0, n = 0, k = 0;
        int* send_counts, * recv_counts, * send_offset, * recv_offset;
        double start_time, end_time;
        struct ShareData send_data;
        send_data.m = 0; send_data.n = 0; send_data.k = 0;
        // 读取并初始化矩阵参数
        while (m < 128 || m > 2048 || k < 128 || k > 2048 || n < 128 || n > 2048) {
            printf("Enter values for m, k, n (128-2048): ");
            if (scanf("%d %d %d", &m, &k, &n)) {}
        }
        A = (double*)malloc(m * k * sizeof(double));
        B = (double*)malloc(k * n * sizeof(double));
        B_T = (double*)malloc(k * n * sizeof(double));
        C = (double*)malloc(m * n * sizeof(double));
        send_counts = (int*)malloc(comm_sz * sizeof(int));
        recv_counts = (int*)malloc(comm_sz * sizeof(int));
        send_offset = (int*)malloc(comm_sz * sizeof(int));
        recv_offset = (int*)malloc(comm_sz * sizeof(int));
        // 初始化矩阵数据
        for (int i = 0; i < (m * k); i++) {
            A[i] = (double)(rand() % 100);
        }
        for (int i = 0; i < (k * n); i++) {
            B[i] = (double)(rand() % 100);
        }
        for (int i = 0; i < (m * n); i++) {
            C[i] = 0.0;
        }
        matrix_transpose(B, B_T, k, n);
        // 分发任务
        int rows_per_process = m / comm_sz;
        int remaining_rows = m % comm_sz;
        int soffset = 0, roffset = 0;
        for (int i = 0; i < comm_sz; i++) {
            int tmp = (i < remaining_rows) ? rows_per_process + 1 : rows_per_process;
            send_counts[i] = k * tmp;
            recv_counts[i] = n * tmp;
            send_offset[i] = soffset;
            recv_offset[i] = roffset;
            soffset += send_counts[i];
            roffset += recv_counts[i];
            //printf("\nSend[%d]: offset = %d , count = %d",i, send_offset[i], send_counts[i]);
            //printf("\nRecv[%d]: offset = %d , count = %d",i, recv_offset[i], recv_counts[i]);
        }
        // 广播矩阵规模
        send_data.m = m; send_data.n = n; send_data.k = k;
        MPI_Bcast(&send_data, 1, data_type, 0, MPI_COMM_WORLD);
        start_time = MPI_Wtime();
        // 分发矩阵A、B
        MPI_Scatterv(A, send_counts, send_offset, MPI_DOUBLE, MPI_IN_PLACE, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(B_T, k * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        double* C_partial = (double*)malloc(send_counts[0] * n * sizeof(double));
        // 矩阵乘法
        mpi_multiple(A, B_T, C_partial, send_counts[0]/k, n, k);
        // 收集到矩阵C中
        MPI_Gatherv(C_partial, recv_counts[0], MPI_DOUBLE, C, recv_counts, recv_offset, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        end_time = MPI_Wtime();

        // 打印结果和计算时间
        printf("Computations completed.\n");
        printf(" Top left corner of matrix A: \n");
        for (int i = 0; i < min(m, 6); i++) {
            for (int j = 0; j < min(k, 6); j++) {
                printf("%12.0f", A[j + i * k]);
            }
            printf("\n");
        }

        printf(" Top left corner of matrix B: \n");
        for (int i = 0; i < min(k, 6); i++) {
            for (int j = 0; j < min(n, 6); j++) {
                printf("%12.0f", B[j + i * n]);
            }
            printf("\n");
        }
        printf("Top left corner of matrix C:\n");
        for (int i = 0; i < min(m, 6); i++) {
            for (int j = 0; j < min(n, 6); j++) {
                printf("%12.5G", C[j + i * n]);
            }
            printf("\n");
        }
        printf("Time taken: %f seconds\n", end_time - start_time);

        free(A);
        free(B);
        free(B_T);
        free(C);
        free(C_partial);
        free(send_counts);
        free(send_offset);
        free(recv_counts);
        free(recv_offset);
    }
    else {
        // 接收分配给该进程的 A 的部分和整个 B_T
        struct ShareData recv_data;
        int rows, m, k, n;
        // 接收矩阵规模参数
        MPI_Bcast(&recv_data, 1, data_type, 0, MPI_COMM_WORLD);
        m = recv_data.m; n = recv_data.n; k = recv_data.k;
        rows = m / comm_sz;
        if (my_rank < m % comm_sz) rows++;

        double* A_partial = (double*)malloc(rows * k * sizeof(double));
        double* C_partial = (double*)malloc(rows * n * sizeof(double));
        double* B_T = (double*)malloc(k * n * sizeof(double));
        // 接收矩阵A、B
        MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, A_partial, rows * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(B_T, k * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // 矩阵乘法运算
        mpi_multiple(A_partial, B_T, C_partial, rows, n, k);
        // 矩阵C发送到主进程
        MPI_Gatherv(C_partial, rows * n, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        free(A_partial);
        free(C_partial);
        free(B_T);
    }

    MPI_Finalize();
    return 0;
}
