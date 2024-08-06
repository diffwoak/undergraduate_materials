#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>


struct ThreadData {
    int start;
    int end;
    void* arg;
};

void parallel_for(int start, int end, int inc, void* (*functor)(void*), void* arg, int num_threads) {
    pthread_t threads[num_threads];
    struct ThreadData thread_data[num_threads];
    int m = end - start;
    int rows_per_process = m / num_threads;
    int remaining_rows = m % num_threads;
    int offset = 0;
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].start = offset;
        offset += (i < remaining_rows) ? rows_per_process + 1 : rows_per_process;
        thread_data[i].end = offset;
        thread_data[i].arg = arg;
        pthread_create(&threads[i], NULL, functor, (void*)&thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}