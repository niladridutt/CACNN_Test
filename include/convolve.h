#ifndef CACNN_CONVOLVE_TEST_LIB_
#define CACNN_CONVOLVE_TEST_LIB_

#include <stdlib.h>
#include <stdint.h>
#include <errno.h>

// Standard Convolution
int convolve_std
(
	float* in, float* out, float** filters, uint32_t C, uint32_t K, uint32_t W,
	uint32_t H, uint32_t R, uint32_t S, uint32_t sigmaW, uint32_t sigmaH
);

#endif