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

void GemmParallel(const float a[kI][kK], const float b[kK][kJ],
                  float c[kI][kJ]) {
  
#pragma omp parallel num_threads(8)
{
#pragma omp parallel for
  for (int i = 0; i < kI; ++i){
    for (int k = 0; k < kK; ++k){
      for (int j = 0; j < kJ; ++j){
        c[i][j] += a[i][k] * b[k][j];        
      }
    }
  }
}
}
