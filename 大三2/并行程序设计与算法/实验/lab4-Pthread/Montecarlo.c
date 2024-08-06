#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define MIN_N 1024
#define MAX_N 65536

pthread_mutex_t mutex;
int sum = 0;

struct ThreadData {
    float* x;
    float* y;
    int start;
    int end;
};

void* partial_count(void* arg) {
    struct ThreadData* data = (struct ThreadData*)arg;
    int tmp = 0;
    for (int i = data->start; i < data->end; i++) {
        if (data->x[i] * data->x[i] + data->y[i] * data->y[i] <= 1) {
            tmp += 1;
        }
    }
    pthread_mutex_lock(&mutex);
    sum += tmp;
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
    int n = 0;
    int num_threads = strtol(argv[1], NULL, 10);
    pthread_mutex_init(&mutex, NULL);
    srand(time(NULL));
    
    
    while (n < MIN_N || n > MAX_N) {
        printf("Enter the value of n (1024 - 65536): ");
        scanf("%d", &n);
    }
    pthread_t threads[num_threads];
    struct ThreadData thread_data[num_threads];
    int rows_per_process = n / num_threads;
    int remaining_rows = n % num_threads;
    float* x = (float*)malloc(n * sizeof(float));
    float* y = (float*)malloc(n * sizeof(float));
    double times = 0; int test_time = 1;
    for (int p = 0; p < test_time; p++) {
        sum = 0;
        for (int i = 0; i < n; i++) {
            x[i] = (float)rand() / RAND_MAX;
            y[i] = (float)rand() / RAND_MAX;
        }

        int offset = 0;
        clock_t start = clock();
        for (int i = 0; i < num_threads; i++) {
            thread_data[i].x = x;
            thread_data[i].y = y;
            thread_data[i].start = offset;
            offset += (i < remaining_rows) ? rows_per_process + 1 : rows_per_process;
            thread_data[i].end = offset;
            pthread_create(&threads[i], NULL, partial_count, (void*)&thread_data[i]);
        }

        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }

        clock_t end = clock();
        double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

        printf("estimate pi: %f\n", (float)4 * sum / n);
        printf("Time taken: %f seconds\n", time_taken);
        times += time_taken;
    }
    //printf("average time: %f seconds\n", times / test_time);
    

    free(x);
    free(y);
    pthread_mutex_destroy(&mutex);

    return 0;
}
