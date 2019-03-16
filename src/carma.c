// Source: https://github.com/dose78/CARMA/blob/master/carma_single.c

#include "carma.h"
#define SPLIT_M 1
#define SPLIT_K 2
#define SPLIT_N 3

// Split largest dimension
int dim_to_split ( int m, int k, int n )
{
	if ( n >= k && n >= m ) return SPLIT_N;
	if ( m >= k && m >= n ) return SPLIT_M;
	return SPLIT_K;
}

void inner_multiply ( int m, int k, int n, float *A, int LDA, float *B, int LDB, float *C, int LDC, int depth, int max_depth )
{
	if ( depth >= max_depth || m <= 4 || n <= 4 || k <= 4 )
	{
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, LDA, B, LDB, 0, C, LDC);
		return;
	}

	int next_depth = depth + 1;
	int dim        = dim_to_split( m, k, n );

	if ( dim == SPLIT_N )
	{
		inner_multiply(m, k, n/2, A, LDA, B, LDB, C, LDC, next_depth, max_depth);
		inner_multiply(m, k, n/2, A, LDA, B + n/2, LDB, C + n/2, LDC, next_depth, max_depth);
	}
	else if ( dim == SPLIT_M )
	{
		inner_multiply(m/2, k, n, A, LDA, B, LDB, C, LDC, next_depth, max_depth);
		inner_multiply(m/2, k, n, A + m/2*LDA, LDA, B, LDB, C + m/2*LDC, LDC, next_depth, max_depth);
	}
	else // SPLIT_K
	{
		inner_multiply(m, k/2, n, A, LDA, B, LDB, C, LDC, next_depth, max_depth);
		inner_multiply(m, k/2, n, A + k/2, LDA, B + k/2*LDB, LDB, C, LDC, next_depth, max_depth);
	}
}

void multiply ( int m, int k, int n, float *A, float *B, float *C, int max_depth )
{
	inner_multiply( m, k, n, A, k, B, n, C, n, 0, max_depth );
}