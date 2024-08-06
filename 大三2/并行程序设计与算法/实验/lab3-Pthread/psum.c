#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define MIN_N 1000000
#define MAX_N 128000000

pthread_mutex_t mutex;
long long sum = 0;

struct ThreadData {
    int* A;
    int start;
    int end;
    long long sum;
};

void* partial_sum(void* arg) {
    struct ThreadData* data = (struct ThreadData*)arg;
    long long tmp = 0;
    for (int i = data->start; i < data->end; i++) {
        tmp += data->A[i];
    }
    pthread_mutex_lock(&mutex);
    sum += tmp;
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
    int n = 0;
    int num_threads = strtol(argv[1], NULL, 10);
    srand(time(NULL));

    
    while (n < MIN_N || n > MAX_N) {
        printf("Enter the value of n (1M - 128M): ");
        scanf("%d", &n);
    }

    int* A = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        A[i] =  rand() % 100;
    }

    pthread_t threads[num_threads];
    struct ThreadData thread_data[num_threads];
    int rows_per_process = n / num_threads;
    int remaining_rows = n % num_threads;
    int offset = 0;

    pthread_mutex_init(&mutex, NULL);

    clock_t start = clock();

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].A = A;
        thread_data[i].start = offset;
        offset += (i < remaining_rows) ? rows_per_process + 1 : rows_per_process;
        thread_data[i].end = offset;
        pthread_create(&threads[i], NULL, partial_sum, (void*)&thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        //sum += thread_data[i].sum;
    }

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("A: [");
    for (int i = 0; i < 6; i++) {
        printf("%d ", A[i]);
    }
    printf("...]\n");

    printf("Sum of A: %lld\n", sum);
    printf("Time taken: %f seconds\n", time_taken);

    free(A);
    pthread_mutex_destroy(&mutex);

    return 0;
}
