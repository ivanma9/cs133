
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


    int rowSize = kI / numProcesses;

    /* ------------------------ 1 Using Scatter and Bcast ----------------------- */
    if (rank == 0) {
        memcpy(bBuffer, b, sizeof(float) * bSize);
        memset(cBuffer, 0, sizeof(float) * cSize);
    }

    MPI_Scatter(a, aSize, MPI_FLOAT, aBuffer, aSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(bBuffer, bSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* --------------------- 2 Using Isend, Irecv, and Wait --------------------- */
    // int offset = rowSize;
    // MPI_Status status;
    // MPI_Request request;
    // if (rank == 0) {
    //     memcpy(aBuffer, a, sizeof(float) * aSize);
    //     for (int i = 1; i < numProcesses; i++) {
    //         MPI_Isend(&a[offset][0], aSize, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &request);
    //         offset += rowSize;
    //     }
    // } else {
    //     MPI_Irecv(aBuffer, aSize, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request);
    //     MPI_Wait(&request, &status);
    // }

    // if (rank == 0) {
    //     memcpy(bBuffer, b, sizeof(float) * bSize);
    //     for (int i = 1; i < numProcesses; i++) {
    //         MPI_Isend(b, bSize, MPI_FLOAT, i, 2, MPI_COMM_WORLD, &request);
    //     }
    // } else {
    //     MPI_Irecv(bBuffer, bSize, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &request);
    //     MPI_Wait(&request, &status);
    // }

    /* --------------------------- X 3 Using Send, Recv --------------------------- */
    // int offset = rowSize;
    // MPI_Status status;
    // if (rank == 0) {
    //     memcpy(aBuffer, a, sizeof(float) * aSize);
    //     memcpy(bBuffer, b, sizeof(float) * bSize);
    //     for (int i = 1; i < numProcesses; i++) {
    //         MPI_Send(&a[offset][0], aSize, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
    //         MPI_Send(b, bSize, MPI_FLOAT, i, 2, MPI_COMM_WORLD);
    //     }
    // } else {
    //     MPI_Recv(aBuffer, aSize, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
    //     MPI_Recv(bBuffer, bSize, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &status);
    // }

    /* ----------------- 4 Using Send Recv, Scatter, and Gather ----------------- */
    // int offset = rowSize;
    // MPI_Status status;
    // if (rank == 0) {
    //     memcpy(bBuffer, b, sizeof(float) * bSize);
    //     for (int i = 1; i < numProcesses; i++) {
    //         MPI_Send(b, bSize, MPI_FLOAT, i, 2, MPI_COMM_WORLD);
    //     }
    // } else {
    //     MPI_Recv(bBuffer, bSize, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &status);
    // }
    // MPI_Scatter(a, aSize, MPI_FLOAT, aBuffer, aSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 186 GFlops
    int iBlockSize = 32;
    int jBlockSize = 128;
    int kBlockSize = 8;

    float tempBuffer;
    for (int i = 0; i < rowSize; i += iBlockSize) {
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

    /* ----------------------------- 1 Using gather ----------------------------- */
    MPI_Gather(cBuffer, cSize, MPI_FLOAT, c, cSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* ----------------------- 2 Using Isend, Irecv, Wait ----------------------- */
    // if (rank == 0) {
    //     offset = rowSize;
    //     memcpy(c, cBuffer, sizeof(float) * cSize);
    //     for (int i = 1; i < numProcesses; i++) {
    //         MPI_Irecv(&c[offset][0], cSize, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &request);
    //         offset += rowSize;
    //         MPI_Wait(&request, &status);
    //     }
    // } else {
    //     MPI_Isend(cBuffer, cSize, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request);
    // }

    /* --------------------------- X 3 Using Send, Recv --------------------------- */
    // if (rank == 0) {
    //     offset = rowSize;
    //     // memcpy(c, cBuffer, sizeof(float) * cSize);
    //     for (int i = 1; i < numProcesses; i++) {
    //         MPI_Recv(&c[offset][0], cSize, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &status);
    //         offset += rowSize;
    //     }
    // } else {
    //     MPI_Send(cBuffer, cSize, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    // }

    /* ----------------- 4 Using Send Recv, Scatter, and Gather ----------------- */
    // MPI_Gather(cBuffer, cSize, MPI_FLOAT, c, cSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
}