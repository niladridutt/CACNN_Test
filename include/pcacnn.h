#ifndef CACNN_CACNN_TEST_LIB_
#define CACNN_CACNN_TEST_LIB_

#include <stdlib.h>
#include <stdint.h>
#include <errno.h>

// Standard Convolution
int convolve_cacnn
(
	float* in, float* out, float** filters, uint32_t C, uint32_t K, uint32_t W,
	uint32_t H, uint32_t R, uint32_t S, uint32_t sigmaW, uint32_t sigmaH,
	uint32_t C_block, uint32_t K_block, uint32_t W_block, uint32_t H_block,
	uint32_t RP_block, uint32_t RPP_block, uint32_t SP_block, uint32_t SPP_block
);

#endif
