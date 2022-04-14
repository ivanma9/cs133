#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

#include "lib/gemm.h"

using std::clog;
using std::endl;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::steady_clock;

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {
  int BLOCK_SIZE = 64;
  int inner_BLOCK_SIZE = 1024;
  // matrix multiplication
#pragma omp parallel num_threads(8)
  {
#pragma omp for
    for (int i = 0; i < kI; i += BLOCK_SIZE) {
      for (int k = 0; k < kK; k += BLOCK_SIZE) {
        for (int j = 0; j < kJ; j += inner_BLOCK_SIZE) {
          int i2 = i + BLOCK_SIZE;
          int j2 = j + inner_BLOCK_SIZE;
          int k2 = k + BLOCK_SIZE;
          for (int ii = i; ii < i2; ii += 2) {
            for (int kk = k; kk < k2; kk += 2) {
              for (int jj = j; jj < j2; ++jj) {
                c[ii][jj] += a[ii][kk] * b[kk][jj];
                c[ii + 1][jj] += a[ii + 1][kk] * b[kk][jj];

                c[ii][jj] += a[ii][kk + 1] * b[kk + 1][jj];
                c[ii + 1][jj] += a[ii + 1][kk + 1] * b[kk + 1][jj];
              }
            }
          }
        }
      }
    }
  }
}

