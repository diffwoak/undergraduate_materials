# include <cstdlib>
# include <iostream>
# include <iomanip>
#include <stdio.h>
#include <stdlib.h>
# include <cmath>
# include <ctime>
# include <omp.h>
#include <mpi.h>

using namespace std;

int main ( );
void ccopy ( int n, double x[], double y[] );
void cfft2(int n, double x[], double y[], double w[], double sgn, int comm_sz);
void cfft2_sub(int n, double w[], double sgn, int my_rank, int comm_sz);
void cffti ( int n, double w[] );
double cpu_time ( void );
double ggl ( double *ds );
void step(int n, int mj, double a[], double b[], double c[],
    double d[], double w[], double sgn, int begin, int end);
void timestamp ( );
void reorderArray(double* arr,int n); 

//****************************************************************************80

int main ( ){
    int  comm_sz;
    int  my_rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    // 同时存在的变量
    int first;
    int icase;
    int it;
    int ln2;
    int n;
    int nits = 10000;
    double sgn;
  
  if (my_rank == 0) {
      double* w, * x, * y;
      double* z;
      double z0,z1;
      double ctime,ctime1,ctime2;
      double error;
      static double seed;
      double flops; double mflops;
      double fnm1;
      timestamp();
      cout << "             N      NITS    Error         Time          Time/Call     MFLOPS\n\n";
      seed = 331.0;
      n = 1;
      for (ln2 = 1; ln2 <= 15; ln2++)
      {
          n = 2 * n;
          w = new double[n];
          x = new double[2 * n];
          y = new double[2 * n];
          z = new double[2 * n];
          int real_nums = (n / 2) < comm_sz ? (n / 2) : comm_sz;
          first = 1;
          cffti(n, w);   // len(w) = n w = [cos,sin,cos,sin...]将2pi平分n/2块
          MPI_Bcast(w, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
          for (icase = 0; icase < 2; icase++)// 计算两种初始化情况
          {    
              if (first)
              {
                  for (int i = 0; i < 2 * n; i = i + 2)
                  {
                      z0 = ggl(&seed);
                      z1 = ggl(&seed);
                      x[i] = z0;
                      z[i] = z0;
                      x[i + 1] = z1;
                      z[i + 1] = z1;
                  }
              }
              else
              {
                  for (int i = 0; i < 2 * n; i = i + 2)
                  {
                      z0 = 0.0;
                      z1 = 0.0;
                      x[i] = z0;
                      z[i] = z0;
                      x[i + 1] = z1;
                      z[i + 1] = z1;
                  }
              }

              if (first)
              {
                  sgn = +1.0;
                  reorderArray(x, n);
                  cfft2(n, x, y, w, sgn, real_nums);  // 正向FFT
                  sgn = -1.0;
                  reorderArray(y, n);
                  cfft2(n, y, x, w, sgn, real_nums);  // 反向FFT
                  fnm1 = 1.0 / (double)n;
                  error = 0.0;
                  for (int i = 0; i < 2 * n; i = i + 2)
                  {
                      error = error
                          + pow(z[i] - fnm1 * x[i], 2)
                          + pow(z[i + 1] - fnm1 * x[i + 1], 2);
                  }
                  error = sqrt(fnm1 * error);
                  cout << "  " << setw(12) << n
                      << "  " << setw(8) << nits
                      << "  " << setw(12) << error;
                  first = 0;
              }
              else
              {
                  ctime1 = cpu_time();
                  for (it = 0; it < nits; it++)
                  {
                      sgn = +1.0;
                      reorderArray(x, n);
                      cfft2(n, x, y, w, sgn, real_nums);
                      sgn = -1.0;
                      reorderArray(y, n);
                      cfft2(n, y, x, w, sgn, real_nums);
                      
                  }
                  ctime2 = cpu_time();
                  
                  ctime = ctime2 - ctime1;
                  
                  flops = 2.0 * (double)nits * (5.0 * (double)n * (double)ln2);

                  mflops = flops / 1.0E+06 / ctime;

                  cout << "  " << setw(12) << ctime
                      << "  " << setw(12) << ctime / (double)(2 * nits)
                      << "  " << setw(12) << mflops << "\n";
              }
          }
          if ((ln2 % 4) == 0)
          {
              nits = nits / 10;
          }
          if (nits < 1)
          {
              nits = 1;
          }
          delete[] w;
          delete[] x;
          delete[] y;
          delete[] z;
      }
      timestamp();
      return 0;
  }else {// 其他进程
    double* w;
    n = 1;
    for (ln2 = 1; ln2 <= 15; ln2++)
    {   
        n = 2 * n;
        int real_nums = (n / 2) < comm_sz ? (n / 2) : comm_sz;
        w = new double[n];
        MPI_Bcast(w, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (my_rank < real_nums) { 
            first = 1;
            for (icase = 0; icase < 2; icase++)
            {
                if (first)
                {
                    sgn = +1.0;
                    cfft2_sub(n, w, sgn, my_rank, real_nums);
                    sgn = -1.0;
                    cfft2_sub(n, w, sgn, my_rank, real_nums);
                    first = 0;
                }
                else
                {
                    for (it = 0; it < nits; it++)
                    {
                        sgn = +1.0;
                        cfft2_sub(n, w, sgn, my_rank, real_nums);
                        sgn = -1.0;
                        cfft2_sub(n, w, sgn, my_rank, real_nums);
                    }
                }
            }
            if ((ln2 % 4) == 0)
            {
                nits = nits / 10;
            }
            if (nits < 1)
            {
                nits = 1;
            }
        }
        delete[] w;
    }
  }
}
//****************************************************************************80
void ccopy ( int n, double x[], double y[] )
{
  int i;

  for ( i = 0; i < n; i++ )
  {
    y[i*2+0] = x[i*2+0];
    y[i*2+1] = x[i*2+1];
   }
  return;
}
//****************************************************************************80


void cfft2(int n, double x[], double y[], double w[], double sgn, int comm_sz)
{
    int j, m, mj;
    int tgle;
    int pro_nums = comm_sz;
    int counts = (n / 2) / pro_nums;
    int begin = 0, end = counts;
    m = ( int ) ( log ( ( double ) n ) / log ( 1.99 ) );
    mj = 1;
    tgle = 1;
    for (int i = 1; i < pro_nums; i++) {
        MPI_Send(&x[i * counts * 2 * 2], counts * 2 * 2, MPI_DOUBLE, i, 2, MPI_COMM_WORLD);
    }
    step(n, mj, &x[0], &x[mj * 2], &y[0], &y[mj * 2 + 0], w, sgn, begin, end);
    for (int i = 1; i < pro_nums; i++) {
        MPI_Recv(&y[i * counts * 2 * 2], counts * 2 * 2, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (n == 2)return;
    for ( j = 0; j < m - 2; j++ )
    {
        mj = mj * 2;
        pro_nums = min((n / 2) / mj, pro_nums);
        counts = (n / 2) / pro_nums;
        end = counts;
        if ( tgle )
        {
            // 根据mj重新划分,counts翻倍,comm_sz减半
            for (int i = 1; i < pro_nums; i++) {
                MPI_Send(&y[i * counts * 2 * 2], counts * 2 * 2, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
            }
            step(n, mj, &y[0], &y[mj * 2], &x[0], &x[mj * 2 + 0], w, sgn, begin, end);
            tgle = 0;
            for (int i = 1; i < pro_nums; i++) {
                MPI_Recv(&x[i * counts * 2 * 2], counts * 2 * 2, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } 
        }
        else
        {   
            for (int i = 1; i < pro_nums; i++) {
                MPI_Send(&x[i * counts * 2 * 2], counts * 2 * 2, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
            }
            step(n, mj, &x[0], &x[mj * 2], &y[0], &y[mj * 2 + 0], w, sgn, begin, end);
            tgle = 1;
            for (int i = 1; i < pro_nums; i ++) {
                MPI_Recv(&y[i * counts * 2 * 2], counts * 2 * 2, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    if ( tgle ){
        ccopy ( n, y, x );
    }
    mj = n / 2; // 只需使用主进程
    end = mj;
    step(n, mj, &x[0], &x[n], &y[0], &y[n], w, sgn, begin, end);
    return;
}
//****************************************************************************80
void cfft2_sub(int n, double w[], double sgn,int my_rank, int comm_sz)
{
    int j;
    int m = (int)(log((double)n) / log(1.99));
    int mj = 1;
    int pro_nums = comm_sz;
    int counts = (n / 2) / pro_nums;
    int begin = my_rank * counts;
    int end = begin + counts;
    double* x, * y;
    x = new double[2 * 2 * counts];
    y = new double[2 * 2 * counts];
    MPI_Recv(&x[0], counts * 2 * 2, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    step(n, mj, &x[0], &x[mj * 2 + 0], &y[0], &y[mj * 2 + 0], w, sgn, begin, end);
    MPI_Ssend(&y[0], counts * 2 * 2, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    delete[] x;
    delete[] y;
    if (n == 2)return;
    for (j = 0; j < m - 2; j++)
    {
        mj = mj * 2;
        pro_nums = min((n / 2) / mj, pro_nums);
        counts = (n / 2) / pro_nums;
        begin = my_rank * counts;
        end = begin + counts;
        if (my_rank < pro_nums) {
            x = new double[2 * 2 * counts];
            y = new double[2 * 2 * counts];
            MPI_Recv(&x[0], counts * 2 * 2, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            step(n, mj, &x[0 * 2 + 0], &x[mj * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn, begin, end);
            MPI_Ssend(&y[0], counts * 2 * 2, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            delete[] x;
            delete[] y;
        }
    }
    return;
}
//****************************************************************************80
void cffti ( int n, double w[] )
{
  double arg;
  double aw;
  int i;
  int n2;
  const double pi = 3.141592653589793;

  n2 = n / 2;
  aw = 2.0 * pi / ( ( double ) n );
  for ( i = 0; i < n2; i++ )
  {
    arg = aw * ( ( double ) i );
    w[i*2+0] = cos ( arg );
    w[i*2+1] = sin ( arg );
  }
  return;
}
//****************************************************************************80
double cpu_time ( void )

{
  double value;

  value = ( double ) clock ( ) / ( double ) CLOCKS_PER_SEC;

  return value;
}
//****************************************************************************80
double ggl ( double *seed )
{
  double d2 = 0.2147483647e10;
  double t;
  double value;

  t = *seed;
  t = fmod ( 16807.0 * t, d2 );
  *seed = t;
  value = ( t - 1.0 ) / ( d2 - 1.0 );

  return value;
}
//****************************************************************************80
void step ( int n, int mj, double a[], double b[], double c[],
  double d[], double w[], double sgn,int begin, int end )
{
  double ambr,ambu;
  int j,jw;
  int k;
  double wjw[2];
  int start = 0;
  for (j = begin / mj; j < end / mj; j++)
  {
    jw = j * mj;
    wjw[0] = w[jw*2+0]; // cos
    wjw[1] = w[jw*2+1]; // sin
    if ( sgn < 0.0 ) 
    {
      wjw[1] = - wjw[1];
    }
    
    for (k = 0; k < mj; k++)
    {
        c[(start) * 2 + 0] = a[(start) * 2 + 0] + b[(start) * 2 + 0];
        c[(start) * 2 + 1] = a[(start) * 2 + 1] + b[(start) * 2 + 1];

        ambr = a[(start) * 2 + 0] - b[(start) * 2 + 0];
        ambu = a[(start) * 2 + 1] - b[(start) * 2 + 1];

        d[(start) * 2 + 0] = wjw[0] * ambr - wjw[1] * ambu;
        d[(start) * 2 + 1] = wjw[1] * ambr + wjw[0] * ambu;
        start++;
    }
    // 如果存在多个mj
    start += mj;
  }
  return;
}
//****************************************************************************80
void timestamp ( ){
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  cout << time_buffer << "\n";

  return;
# undef TIME_SIZE
}
//****************************************************************************80
void reorderArray(double* arr, int n) {
    double* tmp1 = new double[n * 2];
    double* tmp2 = new double[n * 2];
    for (int i = 0; i < n * 2; i++) {
        tmp2[i] = arr[i];
    }
    for (int mj = 1; mj < n / 2; mj = mj * 2) {
        int index = 0;
        for (int i = 0; i < (n / 2) / mj; i++) {
            for (int j = 0; j < mj; j++) {
                tmp1[index++] = tmp2[(i * mj + j) * 2];
                tmp1[index++] = tmp2[(i * mj + j) * 2 + 1];
            }
            for (int j = 0; j < mj; j++) {
                tmp1[index++] = tmp2[(n / 2 + i * mj + j) * 2];
                tmp1[index++] = tmp2[(n / 2 + i * mj + j) * 2 + 1];
            }
        }
        for (int i = 0; i < n * 2; i++) {
            tmp2[i] = tmp1[i];
        }
    }
    for (int i = 0; i < n * 2; i++) {
        arr[i] = tmp2[i];
    }
    delete[] tmp1;
    delete[] tmp2;
}