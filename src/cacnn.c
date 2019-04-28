#include "cacnn.h"
#include "omp.h"

// Communication Avoiding Convolution
int convolve_cacnn
(
	float* in, float* out, float** filters, uint32_t C, uint32_t K, uint32_t W,
	uint32_t H, uint32_t R, uint32_t S, uint32_t sigmaW, uint32_t sigmaH,
	uint32_t C_block, uint32_t K_block, uint32_t W_block, uint32_t H_block,
	uint32_t RP_block, uint32_t RPP_block, uint32_t SP_block, uint32_t SPP_block
)
{
	// Main Iterations
	uint32_t c,   k,   w,   h,   rp,   rpp,   sp,   spp;
	uint32_t c_b, k_b, w_b, h_b, rp_b, rpp_b, sp_b, spp_b;
	uint32_t c_p, k_p, w_p, h_p, rp_p, rpp_p, sp_p, spp_p;

	uint32_t r_bound = R/sigmaW;
	uint32_t s_bound = S/sigmaH;
	uint32_t in_H = sigmaH*(H - 1) + S;
	uint32_t in_W = sigmaW*(W - 1) + R;

	// Blocking
	#pragma omp parallel for
	for ( c_b = 0; c_b < C; c_b += C_block )
	{
		#pragma omp parallel for
		for ( k_b = 0; k_b < K; k_b += K_block )
		{
			for ( w_b = 0; w_b < W; w_b += W_block )
			{
				for ( h_b = 0; h_b < H; h_b += H_block )
				{
					for ( rp_b = 0; rp_b < r_bound; rp_b += RP_block )
					{
						for ( rpp_b = 0; rpp_b < sigmaW; rpp_b += RPP_block )
						{
							for ( sp_b = 0; sp_b < s_bound; sp_b += SP_block )
							{
								for ( spp_b = 0; spp_b < sigmaH; spp_b += SPP_block )
								{

	// Piecing
	#pragma omp parallel for
	for ( c_p = 0; c_p < C_block; c_p += 1 )
	{
		#pragma omp parallel for
		for ( k_p = 0; k_p < K_block; k_p += 1 )
		{
			for ( w_p = 0; w_p < W_block; w_p += 1 )
			{
				for ( h_p = 0; h_p < H_block; h_p += 1 )
				{
					for ( rp_p = 0; rp_p < RP_block; rp_p += 1 )
					{
						for ( rpp_p = 0; rpp_p < RPP_block; rpp_p += 1 )
						{
							for ( sp_p = 0; sp_p < SP_block; sp_p += 1 )
							{
								//#pragma omp parallel for reduction(+:u,v)
								for ( spp_p = 0; spp_p < SPP_block; spp_p += 1 )
								{
									c   = c_b   + c_p;
									k   = k_b   + k_p;
									w   = w_b   + w_p;
									h   = h_b   + h_p;
									rp  = rp_b  + rp_p;
									rpp = rpp_b + rpp_p;
									sp  = sp_b  + sp_p;
									spp = spp_b + spp_p;

									float u = filters[k][ (c*R*S) + (sigmaH*sp + spp)*(R) + (sigmaW*rp + rpp) ];
									float v = in[ (c*in_H*in_W) + (spp + sigmaH*(sp + h))*(in_W) + (rpp + sigmaW*(rp + w)) ];
									out[ k*W*H + h*W + w ] += u*v;
								}
							}
						}
					}
				}
			}
		}
	}
								}
							}
						}
					}
				}
			}
		}
	}

	return 0;
}
