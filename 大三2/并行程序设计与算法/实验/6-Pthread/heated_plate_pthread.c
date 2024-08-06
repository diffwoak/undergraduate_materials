# include <stdlib.h>
# include <stdio.h>
# include <math.h>
#include <time.h>
#include "parallel.h"

# define M 500
# define N 500

pthread_mutex_t mutex;
struct functor_args {
    double * w, * u;
    double mean;
    double my_diff;
};

void* initial_value_1(void* args) {
    struct ThreadData* data = (struct ThreadData*)args;
    struct functor_args* arg = (struct functor_args*)data->arg;
    double sum = 0;
    for (int i = data->start + 1; i < data->end + 1; i++)
    {
        arg->w[i * N + 0] = 100.0;
        arg->w[i * N + N - 1] = 100.0;
        sum = sum + arg->w[i * N + 0] + arg->w[i * N + N - 1];
    }
    pthread_mutex_lock(&mutex);
    arg->mean += sum;
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}
void* initial_value_2(void* args) {
    struct ThreadData* data = (struct ThreadData*)args;
    struct functor_args* arg = (struct functor_args*)data->arg;
    double sum = 0;
    for (int j = data->start; j < data->end; j++)
    {
        arg->w[(M - 1) * N + j] = 100.0;
        arg->w[j] = 0.0;
        sum = sum + arg->w[(M - 1) * N + j] + arg->w[j];
    }
    pthread_mutex_lock(&mutex);
    arg->mean += sum;
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}
void* initialize_solution(void* args) {
    struct ThreadData* data = (struct ThreadData*)args;
    struct functor_args* arg = (struct functor_args*)data->arg;
    for (int i = data->start+1; i < data->end+1; i++)
    {
        for (int j = 1; j < N - 1; j++)
        {
            arg->w[i * N + j] = arg->mean;
        }
    }
    pthread_exit(NULL);
}

void* save_u(void* args) {
    struct ThreadData* data = (struct ThreadData*)args;
    struct functor_args* arg = (struct functor_args*)data->arg;
    for (int i = data->start; i < data->end; i++)
    {
        for (int j = 0; j < N; j++)
        {
            arg->u[i * N + j] = arg->w[i * N + j];
        }
    }
    pthread_exit(NULL);
}
void* new_w(void* args) {
    struct ThreadData* data = (struct ThreadData*)args;
    struct functor_args* arg = (struct functor_args*)data->arg;
    for (int i = data->start + 1; i < data-> end + 1; i++)
    {
        for (int j = 1; j < N - 1; j++)
        {
            arg->w[i * N + j] = (arg->u[(i - 1) * N + j] + arg->u[(i + 1) * N + j] + arg->u[i * N + j - 1] + arg->u[i * N + j + 1]) / 4.0;
        }
    }
    pthread_exit(NULL);
}
void* update_diff(void* args) {
    struct ThreadData* data = (struct ThreadData*)args;
    struct functor_args* arg = (struct functor_args*)data->arg;
    double tmp;
    for (int i = data->start + 1; i < data->end + 1; i++)
    {
        for (int j = 1; j < N - 1; j++)
        {
            tmp = fabs(arg->w[i * N + j] - arg->u[i * N + j]);
            pthread_mutex_lock(&mutex);
            if (arg->my_diff < tmp)
            {
                arg->my_diff = tmp;
            }
            pthread_mutex_unlock(&mutex);
        }
    }
    pthread_exit(NULL);
}



/******************************************************************************/

int main ( int argc, char *argv[] )

/******************************************************************************/
/*
  Purpose:

    MAIN is the main program for HEATED_PLATE_OPENMP.

  Discussion:

    This code solves the steady state heat equation on a rectangular region.

    The sequential version of this program needs approximately
    18/epsilon iterations to complete. 


    The physical region, and the boundary conditions, are suggested
    by this diagram;

                   W = 0
             +------------------+
             |                  |
    W = 100  |                  | W = 100
             |                  |
             +------------------+
                   W = 100

    The region is covered with a grid of M by N nodes, and an N by N
    array W is used to record the temperature.  The correspondence between
    array indices and locations in the region is suggested by giving the
    indices of the four corners:

                  I = 0
          [0][0]-------------[0][N-1]
             |                  |
      J = 0  |                  |  J = N-1
             |                  |
        [M-1][0]-----------[M-1][N-1]
                  I = M-1

    The steady state solution to the discrete heat equation satisfies the
    following condition at an interior grid point:

      W[Central] = (1/4) * ( W[North] + W[South] + W[East] + W[West] )

    where "Central" is the index of the grid point, "North" is the index
    of its immediate neighbor to the "north", and so on.
   
    Given an approximate solution of the steady state heat equation, a
    "better" solution is given by replacing each interior point` by the
    average of its 4 neighbors - in other words, by using the condition
    as an ASSIGNMENT statement:

      W[Central]  <=  (1/4) * ( W[North] + W[South] + W[East] + W[West] )

    If this process is repeated often enough, the difference between successive 
    estimates of the solution will go to zero.

    This program carries out such an iteration, using a tolerance specified by
    the user, and writes the final estimate of the solution to a file that can
    be used for graphic processing.

  Licensing:

    This code is distributed under the MIT license. 

  Modified:

    18 October 2011

  Author:

    Original C version by Michael Quinn.
    This C version by John Burkardt.

  Reference:

    Michael Quinn,
    Parallel Programming in C with MPI and OpenMP,
    McGraw-Hill, 2004,
    ISBN13: 978-0071232654,
    LC: QA76.73.C15.Q55.

  Local parameters:

    Local, double DIFF, the norm of the change in the solution from one iteration
    to the next.

    Local, double MEAN, the average of the boundary values, used to initialize
    the values of the solution in the interior.

    Local, double U[M][N], the solution at the previous iteration.

    Local, double W[M][N], the solution computed at the latest iteration.
*/
{
  double diff;
  double epsilon = 0.001;
  int iterations;
  int iterations_print;
  double* u;
  double* w;
  double wtime;
  int num_threads = strtol(argv[1], NULL, 10);

  pthread_mutex_init(&mutex, NULL);
  printf ( "\n" );
  printf ( "HEATED_PLATE_OPENMP\n" );
  printf ( "  C/PTHREAD version\n" );
  printf ( "  A program to solve for the steady state temperature distribution\n" );
  printf ( "  over a rectangular plate.\n" );
  printf ( "\n" );
  printf ( "  Spatial grid of %d by %d points.\n", M, N );
  printf ( "  The iteration will be repeated until the change is <= %e\n", epsilon ); 
  printf ( "  Number of threads =              %d\n", num_threads);
  u = (double*)malloc(M * N * sizeof(double));
  w = (double*)malloc(M * N * sizeof(double));
  srand(time(NULL)); // 初始化随机数种子
  struct functor_args args;
  args.w = w;
  args.u = u;
  args.mean = 0.0;
/*
  Set the boundary values, which don't change. 
  Average the boundary values, to come up with a reasonable
  initial value for the interior.
*/
  parallel_for(0, M - 2, 1, initial_value_1, (void*)&args, num_threads);
  parallel_for(0, N, 1, initial_value_2, (void*)&args, num_threads);
/*
  OpenMP note:
  You cannot normalize MEAN inside the parallel region.  It
  only gets its correct value once you leave the parallel region.
  So we interrupt the parallel region, set MEAN, and go back in.
*/
  args.mean = args.mean / ( double ) ( 2 * M + 2 * N - 4 );
  printf ( "\n" );
  printf ( "  MEAN = %f\n", args.mean );
/* 
  Initialize the interior solution to the mean value.

*/
  parallel_for(0, M - 2, 1, initialize_solution, (void*)&args, num_threads);
/*
  iterate until the  new solution W differs from the old solution U
  by no more than EPSILON.
*/
  iterations = 0;
  iterations_print = 1;
  printf ( "\n" );
  printf ( " Iteration  Change\n" );
  printf ( "\n" );

  clock_t start = clock();
  diff = epsilon;

  while ( epsilon <= diff )
  {
/*
  Save the old solution in U.
*/
    parallel_for(0, M, 1, save_u, (void*)&args, num_threads);
/*
  Determine the new estimate of the solution at the interior points.
  The new solution W is the average of north, south, east and west neighbors.
*/
    parallel_for(0, M - 2, 1, new_w, (void*)&args, num_threads);
/*
  C and C++ cannot compute a maximum as a reduction operation.

  Therefore, we define a private variable MY_DIFF for each thread.
  Once they have all computed their values, we use a CRITICAL section
  to update DIFF.
*/
    diff = 0.0;
    args.my_diff = 0.0;
    parallel_for(0, M - 2, 1, update_diff, (void*)&args, num_threads);
    if (diff < args.my_diff)
    {
        diff = args.my_diff;
    }

    iterations++;
    if ( iterations == iterations_print )
    {
      printf ( "  %8d  %f\n", iterations, diff );
      iterations_print = 2 * iterations_print;
    }
  } 
  clock_t end = clock();
  wtime = (double)(end - start) / CLOCKS_PER_SEC;

  printf ( "\n" );
  printf ( "  %8d  %f\n", iterations, diff );
  printf ( "\n" );
  printf ( "  Error tolerance achieved.\n" );
  printf ( "  Wallclock time = %f\n", wtime );
/*
  Terminate.
*/
  pthread_mutex_destroy(&mutex);
  free(u);
  free(w);
  printf ( "\n" );
  printf ( "HEATED_PLATE_OPENMP:\n" );
  printf ( "  Normal end of execution.\n" );

  return 0;

# undef M
# undef N
}
