#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <math.h>


#define MIN_N -100
#define MAX_N 100
int flag = 0;
int exist = 0;
float A, B, C, D = 0.;
float a, b, c = 101;
pthread_mutex_t mutex;

void* double_b(void* id) {
    A = b * b;
    //printf("A = %f \n", A);
    pthread_mutex_lock(&mutex);
    flag += 1;
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}

void* four_ac(void* id) {
    B = 4 * a * c;
    //printf("B = %f \n", B);
    pthread_mutex_lock(&mutex);
    flag += 1;
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}

void* sqrt_bac(void* id) {
    while (flag < 2) {}
    C = A - B;
    if (C >= 0) {
        exist = 1;
        C = sqrt(C);
    }
    //printf("C = %f \n", C);
    pthread_exit(NULL);
}

void* two_a(void* id) {
    D = 2 * a;
    //printf("D = %f \n", D);
    pthread_exit(NULL);
}


int main(int argc, char* argv[]) {
    srand(time(NULL));

    while (a < MIN_N || a > MAX_N || b < MIN_N || b > MAX_N || c < MIN_N || c > MAX_N) {
        printf("Enter the value of a b c between -100 and 100: ");
        scanf("%f%f%f", &a,&b,&c);
    }


    pthread_t threads[4];

    pthread_mutex_init(&mutex, NULL);

    clock_t start = clock();
    pthread_create(&threads[0], NULL, double_b, NULL);
    pthread_create(&threads[1], NULL, four_ac, NULL);
    pthread_create(&threads[2], NULL, sqrt_bac, NULL);
    pthread_create(&threads[3], NULL, two_a, NULL);

    for (int i = 0; i < 4; ++i) {
        pthread_join(threads[i], NULL);
    }
    
    
    /*A = b * b; B = 4 * a * c;
    C = A - B;
    if (C >= 0) {
        exist = 1;
        C = sqrt(C);
    }
    D = 2 * a;
    */
    
    
    
    if (exist) {
        double x1 = (-b + C) / D;
        double x2 = (-b - C) / D;
        printf("x1 = %f, x2 = %f\n", x1, x2);
    }
    else {
        printf("no solution existing\n");
    }
    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time taken: %f seconds\n", time_taken);

    pthread_mutex_destroy(&mutex);

    return 0;
}
