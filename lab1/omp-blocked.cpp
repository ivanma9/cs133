// Header inclusions, if any...

#include <cmath>
#include <vector>
#include <random>
#include <cstring>
#include <chrono>

#include "lib/gemm.h"

using std::log;
using std::end;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::steady_clock;
// Using declarations, if any...

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "lib/gemm.h"

using std::clog;
using std::endl;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::steadyclock;

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {
  // setup c
  for (int i = 0; i < kI; ++i) {
    std::memset(c[i], 0, sizeof(float) * kJ);
  }
  // block size
  int BLOCK_SIZE = 64;
  // matrix multiplication
#pragma omp parallel numthreads(8)
  {
#pragma omp for
    for (int i = 0; i < kI; i += BLOCK_SIZE) {
      for (int k = 0; k < kK; k += BLOCK_SIZE) {
        for (int j = 0; j < kJ; j += BLOCK_SIZE) {
          int i2 = i + BLOCK_SIZE;
          int j2 = j + BLOCK_SIZE;
          int k2 = k + BLOCK_SIZE;
          for (int i = i; i < i2; i += 2) {
            for (int k = k; k < k2; k += 2) {
              for (int j = j; j < j2; ++j) {
                c[i][j] += a[i][k] * b[k][j];
                c[i + 1][j] += a[i + 1][k] * b[k][j];
                c[i][j] += a[i][k + 1] * b[k + 1][j];
                c[i + 1][j] += a[i + 1][k + 1] * b[k + 1][j];
              }
            }
          }
        }
      }
    }
  }
}
}
