import numpy as np
import time

def generate_matrix(rows, cols):
    return np.random.rand(rows*cols) * 100
def matrix_multiple(A,B,C,m,n,k):
    for B_i in range(n):
        tmp = []
        for B_j in range(k):
            tmp.append(B[B_j*n+B_i])
        for A_i in range(m):
            for A_j in range(k):
                C[A_i*n+B_i]+=A[A_i*k+A_j]*tmp[A_j]
    return C

def main():
    m, k, n = 0, 0, 0
    while m < 512 or m > 2048 or k < 512 or k > 2048 or n < 512 or n > 2048:
        m, k, n = map(int, input("Enter values for m, k, n (512-2048): ").split())

    A = np.random.rand(m*k) * 100
    B = np.random.rand(n*k) * 100
    # A = np.random.rand(m, k) * 100 
    # B = np.random.rand(k, n) * 100 
    C = np.zeros(m*n)
    start = time.time()  
    C = matrix_multiple(A,B,C,m,n,k)  
    # C = np.dot(A,B)
    end = time.time()  
    spend_time = end - start

    print(" Top left corner of matrix A: ")
    for i in range(6):
        for j in range(6):
            print(A[j+i*k],end=' ')
            # print(A[i][j],end=' ')
        print()
    print(" Top left corner of matrix B: ")
    for i in range(6):
        for j in range(6):
            print(B[j+i*n],end=' ')
            # print(B[i][j],end=' ')
        print()
    print(" Top left corner of matrix C: ")
    for i in range(6):
        for j in range(6):
            print(C[j+i*n],end=' ')
            # print(C[i][j],end=' ')
        print()

    print(f"\nTime taken: {spend_time:.6f} seconds")

if __name__ == "__main__":
    main()
