
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

    float *aBuffer = (float *)lab2::aligned_alloc(1024, aSize * sizeof(float));
    float *bBuffer = (float *)lab2::aligned_alloc(1024, bSize * sizeof(float));
    float *cBuffer = (float *)lab2::aligned_alloc(1024, cSize * sizeof(float));

        memset(cBuffer, 0, sizeof(float) * cSize);


    // int rowSize = kI / numProcesses;

    MPI_Status status;
    if (rank == 0) {
        memcpy(bBuffer, b, sizeof(float) * bSize);
        for (int i = 1; i < numProcesses; i++) {
            MPI_Send(b, bSize, MPI_FLOAT, i, 2, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(bBuffer, bSize, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &status);
    }
    MPI_Scatter(a, aSize, MPI_FLOAT, aBuffer, aSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int iBlockSize = 32;
    int jBlockSize = 128;
    int kBlockSize = 8;

    float tempBuffer;
    for (int i = 0; i < kI; i += iBlockSize) {
        for (int k = 0; k < kK; k += kBlockSize) {
            for (int j = 0; j < kJ; j += jBlockSize) {
                int bi = i + iBlockSize;
                int bj = j + jBlockSize;
                int bk = k + kBlockSize;

                for (int ii = i; ii < bi; ii++) {
                    for (int jj = j; jj < bj; jj++) {
                        tempBuffer = 0;

                        for (int kk = k; kk < bk; kk++) {
                            tempBuffer += aBuffer[ii * kJ + kk] * bBuffer[kk * kJ + jj];
                        }

                        cBuffer[ii * kJ + jj] += tempBuffer;
                    }
                }
            }
        }
    }

    MPI_Gather(cBuffer, cSize, MPI_FLOAT, c, cSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
}