#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define min(x,y) (((x) < (y)) ? (x) : (y))


void floyd(double **matrix,int  n,int num_threads) {
#   pragma omp parallel num_threads(num_threads)
    for (int t = 0; t < n; t++) {
#   pragma omp for schedule(dynamic) 
        for (int v = 0; v < n; v++) {
            if (v != t) {
                for (int w = 0; w < n; w++) {
                    if (w != t) {
                        matrix[v][w] = min(matrix[v][w], matrix[v][t] + matrix[t][w]);
                    }
                }
            }
        }
    }

}

int main(int argc, char* argv[]) {
    int num_threads = strtol(argv[1], NULL, 10);

    //FILE* file = fopen("updated_mouse.csv", "r");
    FILE* file = fopen("updated_flower.csv", "r");
    if (file == NULL) {
        perror("Unable to open file");
        return EXIT_FAILURE;
    }
    // 动态调整最大节点数
    int maxNode = 0;
    int source, target;
    double distance;
    char line[256];
    // 跳过第一行
    fgets(line, sizeof(line), file);
    // 确定最大编号
    while (fgets(line, sizeof(line), file)) {
        sscanf(line, "%d,%d,%lf", &source, &target, &distance);
        if (source > maxNode) { maxNode = source;}
        if (target > maxNode) { maxNode = target;  }
    }
    printf("maxNode:  %d\n", maxNode);
    // 计算矩阵大小
    int size = maxNode + 1;
    double init_value = 2 * size;
    // 分配矩阵内存
    double** matrix = malloc(size * sizeof(double*));
    for (int i = 0; i < size; i++) {
        matrix[i] = malloc(size * sizeof(double));
    }
    // 初始化矩阵
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i == j) {
                matrix[i][j] = 0.0;  // 对角线元素为0
            }
            else {
                matrix[i][j] = init_value; // 其他元素初始化为指定值
            }
        }
    }
    // 回到文件开头重新读取数据
    rewind(file);
    // 跳过第一行标题
    fgets(line, sizeof(line), file);
    // 读取填充矩阵
    while (fgets(line, sizeof(line), file)) {
        sscanf(line, "%d,%d,%lf", &source, &target, &distance);
        //printf("%d,%d,%lf\n", source, target, distance);
        matrix[source][target] = distance;
        matrix[target][source] = distance;
    }
    fclose(file);

    // 打印无向图矩阵
    printf("Original matrix(top right 10):\n");
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            if (matrix[i][j] == init_value) {
                printf("%7s", "inf");
            }
            else {
                printf("%7.3f", matrix[i][j]);
            }
        }
        printf("\n");
    }
    double times = 0; int test_time = 50;
    for (int i = 0; i < test_time; i++) {
        double start = omp_get_wtime();
        floyd(matrix, size, num_threads);
        double end = omp_get_wtime();
        double time = end - start;
        
        /*printf("After matrix(top right 10):\n");
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                if (matrix[i][j] == init_value) {
                    printf("%7s", "inf");
                }
                else {
                    printf("%7.3f", matrix[i][j]);
                }
            }
            printf("\n");
        }*/
        
        printf("Time taken: %f seconds\n", time);
        times += time;
    }
    printf("average time taken: %f\n", times / test_time);

    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
    return 0;
}