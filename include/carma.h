#ifndef CACNN_CARMA_TEST_LIB_
#define CACNN_CARMA_TEST_LIB_

#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include"mkl.h"

// CARMA
void multiply ( int m, int k, int n, float *A, float *B, float *C, int max_depth );

#endif
