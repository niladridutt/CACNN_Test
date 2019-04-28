#include "convolve.h"
#include "omp.h"

// Standard Convolution
int convolve_std
(
	float* in, float* out, float** filters, uint32_t C, uint32_t K, uint32_t W,
	uint32_t H, uint32_t R, uint32_t S, uint32_t sigmaW, uint32_t sigmaH
)
{
	// Main Iterations
	uint32_t c, k, w, h, r, s;
	uint32_t in_H = sigmaH*(H - 1) + S;
	uint32_t in_W = sigmaW*(W - 1) + R;
	#pragma opm parallel for
	for ( c = 0; c < C; c++ )
	{
		#pragma opm parallel for
		for ( k = 0; k < K; k++ )
		{
			for ( w = 0; w < W; w++ )
			{
				for ( h = 0; h < H; h++ )
				{
					for ( r = 0; r < R; r++ )
					{
						for ( s = 0; s < S; s++ )
						{
							float u = filters[k][ c*R*S + s*R + r ];
							float v = in[ (c*in_H*in_W) + ((s + sigmaH*h)*in_W) + (r + sigmaW*w) ];
							out[ k*W*H + h*W + w ] += u*v;
						}
					}
				}
			}
		}
	}

	return 0;
}
