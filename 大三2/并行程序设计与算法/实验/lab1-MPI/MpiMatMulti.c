#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>


void mpi_multiple(double* A, double* B, double* C,int m,int n,int k){
    for(int B_i = 0;B_i< n;B_i++){
        for(int A_i = 0; A_i < m;A_i++){
            C[A_i * n + B_i] = 0;
            for(int A_j = 0; A_j < k;A_j++){
                C[A_i * n + B_i] += A[A_i * k + A_j] * B[B_i * k + A_j];
            }
        }
    }
}
void matrix_transpose(double* B, double* B_T,int k,int n) {
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            B_T[j * n + i] = B[i * n + j];
        }
    }
}


int main()
{   
    int  comm_sz;
    int  my_rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) {
        double* A, * B, * C, * B_T;
        int m = 0, n = 0, k = 0;
        int per_row, first_row, last_row;
        int* send_var;
        // 初始化参数
        //将0号进程作为主进程，负责读取参数，初始化矩阵
        while (m < 128 || m > 2048 || k < 128 || k > 2048 || n < 128 || n > 2048) {
            printf("Enter values for m, k, n (128-2048): ");
            if (scanf("%d %d %d", &m, &k, &n)) {}
        }
        double tmp = (double)m / (comm_sz - 1);
        per_row = (tmp == (int)tmp) ? (int)tmp : ((int)tmp + 1);

        printf(" Initializing data for matrix multiplication C=A*B for matrix \n"
            " A(%ix%i) and matrix B(%ix%i)\n", m, k, k, n);
        A = (double*)malloc(m * k * sizeof(double));
        B = (double*)malloc(k * n * sizeof(double));
        B_T = (double*)malloc(k * n * sizeof(double));
        C = (double*)malloc(m * n * sizeof(double));
        send_var = (int*)malloc(3 * sizeof(int));
        send_var[1] = k; send_var[2] = n;
        for (int i = 0; i < (m * k); i++) {
            A[i] = (double)(rand() % 100);
        }
        for (int i = 0; i < (k * n); i++) {
            B[i] = (double)(rand() % 100);
        }
        for (int i = 0; i < (m * n); i++) {
            C[i] = 0.0;
        }
        printf(" Top left corner of matrix A:\n");
        for (int i = 0; i < min(m, 6); i++) {
            for (int j = 0; j < min(k, 6); j++) {
                printf("%12.0f", A[j + i * k]);
            }
            printf("\n");
        }
        printf(" Top left corner of matrix B:\n");
        for (int i = 0; i < min(k, 6); i++) {
            for (int j = 0; j < min(n, 6); j++) {
                printf("%12.0f", B[j + i * n]);
            }
            printf("\n");
        }
        
        matrix_transpose(B, B_T, k, n);
        free(B);
        // 开始执行
        // 将矩阵乘法的工作划分到其它进程, 将矩阵A按行分割，以及整个B，发送到其他进程
        double start_time = MPI_Wtime();
        if (comm_sz == 1) { //当进程数为1时，直接使用0号进程计算矩阵乘法
            mpi_multiple(A, B_T, C, m, n, k);
        }
        else {
            printf("\n Start sending matrix rows of A and B to subprocess.");
            for (int i = 0; i < comm_sz - 1; i++) {
                first_row = i * per_row; last_row = min((i + 1) * per_row , m );
                printf("\n Send rows of A from %d to %d to process %d", first_row, last_row,i+1);
                send_var[0] = last_row - first_row;               
                MPI_Send(&send_var[0], 3, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
                MPI_Send(&A[first_row * k], (last_row - first_row) * k, MPI_DOUBLE, i + 1, 1, MPI_COMM_WORLD);
                MPI_Send(&B_T[0], k * n,MPI_DOUBLE , i + 1, 2, MPI_COMM_WORLD);
            }
            free(A);
            free(B_T);
            free(send_var);

            // 主进程收集每个子进程计算结果得到完整的矩阵C
            printf("\n Start receving matrix rows of C from subprocess.\n");
            for (int i = 0; i < comm_sz - 1; i++) {
                first_row = i * per_row; last_row = min((i + 1) * per_row, m );
                MPI_Recv(&C[first_row * n], (last_row - first_row) * n, MPI_DOUBLE, i + 1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //printf("\n Receiving matrix C from process %d.",i+1);
            } 
        }
        double end_time = MPI_Wtime();
        printf("\nComputations completed.\n");
        printf("\nTop left corner of matrix C:\n");
        for (int i = 0; i < min(m, 6); i++) {
            for (int j = 0; j < min(n, 6); j++) {
                printf("%12.5G", C[j + i * n]);
            }
            printf("\n");
        }
        printf("Time taken: %f seconds\n", end_time - start_time);
        free(C);
        
    }
    else {
        //子进程负责接收矩阵A的部分行和整个矩阵B，对二者进行矩阵乘法
        
        int n, k,rows;
        int* recv_var = (int*)malloc(4 * sizeof(int));
        double* A_row, *B_T, *C;

        MPI_Recv(&recv_var[0], 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        rows = recv_var[0]; k = recv_var[1]; n = recv_var[2];
        free(recv_var);
        A_row = (double*)malloc(rows * k * sizeof(double));
        B_T = (double*)malloc(n * k * sizeof(double));
        C = (double*)malloc(rows * n * sizeof(double));

        MPI_Recv(&A_row[0], rows * k, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&B_T[0], n * k, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        mpi_multiple(A_row, B_T, C, rows, k, n);
        //得到结果为矩阵C的对应部分行，再传送C回主进程。
        
        MPI_Ssend(&C[0], rows * n, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
       
        free(A_row);
        free(B_T);
        free(C);
    }
    MPI_Finalize();
    return 0;
}