
#include <mpi.h>
#include <string.h>

#include "lib/common.h"
#include "lib/gemm.h"

// You can directly use aligned_alloc
// with lab2::aligned_alloc(...)

// Using declarations, if any...

// Implement an MPI parallel version of blocked GEMM. Program could be based on parallelization from lab 1.

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ], float c[kI][kJ]) {
    int numProcesses, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int aSize = kI * kK / numProcesses;
    int bSize = kK * kJ;
    int cSize = kI * kJ / numProcesses;
    int OUTER_BLOCK_SIZE = 32;
    int INNER_BLOCK_SIZE = 128;
    int BLOCK_SIZE = 8;

    float *aligned_a = (float *)lab2::aligned_alloc(1024, aSize * sizeof(float));
    float *aligned_b = (float *)lab2::aligned_alloc(1024, bSize * sizeof(float));
    float *aligned_c = (float *)lab2::aligned_alloc(1024, cSize * sizeof(float));


    MPI_Status status;
    if (rank == 0) {
        memset(aligned_c, 0, sizeof(float) * cSize);
        memcpy(aligned_b, b, sizeof(float) * bSize);
        for (int i = 1; i < numProcesses; i++) {
            MPI_Send(b, bSize, MPI_FLOAT, i, 2, MPI_COMM_WORLD);
        }
    } 
    else {
        MPI_Recv(aligned_b, bSize, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &status);
    }
    MPI_Scatter(a, aSize, MPI_FLOAT, aligned_a, aSize, MPI_FLOAT, 0, MPI_COMM_WORLD);


    int rowSize = kI / numProcesses;

    for (int i = 0; i < rowSize; i += OUTER_BLOCK_SIZE) {
        for (int k = 0; k < kK; k += BLOCK_SIZE) {
            for (int j = 0; j < kJ; j += INNER_BLOCK_SIZE) {
                int bi = i + OUTER_BLOCK_SIZE;
                int bj = j + INNER_BLOCK_SIZE;
                int bk = k + BLOCK_SIZE;

                for (int ii = i; ii < bi; ii++) {
                    for (int jj = j; jj < bj; jj++) {
                        float temp_register = 0;
                        for (int kk = k; kk < bk; kk++) {
                            temp_register += aligned_a[ii * kJ + kk] * aligned_b[kk * kJ + jj];
                        }
                        aligned_c[ii * kJ + jj] += temp_register;
                    }
                }
            }
        }
    }

    MPI_Gather(aligned_c, cSize, MPI_FLOAT, c, cSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
}