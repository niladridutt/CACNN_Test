#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <errno.h>
#include <omp.h>
#include "lodepng.h"
//#include"mkl.h"
#include <cblas.h>
//#include "libpfc.h"
#include "constats.h"
#include "convolve.h"
#include "cacnn.h"
#include "carma.h"

#define L2_SIZE 65536
#define TRIALS  1

uint32_t __C;
uint32_t __K;
uint32_t __W;
uint32_t __H;
uint32_t __R;
uint32_t __S;
uint32_t __SIGMAW;
uint32_t __SIGMAH;

uint32_t __C_B;
uint32_t __K_B;
uint32_t __W_B;
uint32_t __H_B;
uint32_t __RP_B;
uint32_t __RPP_B;
uint32_t __SP_B;
uint32_t __SPP_B;

static inline
uint64_t RDTSC_START ( void )
{

	unsigned cycles_low, cycles_high;

	asm volatile ( "CPUID\n\t"
				   "RDTSC\n\t"
				   "mov %%edx, %0\n\t"
				   "mov %%eax, %1\n\t"
				   : "=r" (cycles_high), "=r" (cycles_low)::
				   "%rax", "%rbx", "%rcx", "%rdx");

	return ((uint64_t) cycles_high << 32) | cycles_low;
}

/**
 * CITE: http://www.intel.com/content/www/us/en/embedded/training/ia-32-ia-64-benchmark-code-execution-paper.html
 */
static inline
uint64_t RDTSCP ( void )
{
	unsigned cycles_low, cycles_high;

	asm volatile( "RDTSCP\n\t"
				  "mov %%edx, %0\n\t"
				  "mov %%eax, %1\n\t"
				  "CPUID\n\t": "=r" (cycles_high), "=r" (cycles_low)::
				  "%rax", "%rbx", "%rcx", "%rdx");

	return ((uint64_t) cycles_high << 32) | cycles_low;
}

/**
 * This function returns the average time spent collecting timestamps.
 */
static inline
uint64_t fipc_test_time_get_correction ( void )
{
	register uint64_t start;
	register uint64_t end;
	register uint64_t sum;
	register uint64_t i;

	for ( sum = 0, i = 0; i < 100000; ++i )
	{
		start = RDTSC_START();
		end   = RDTSCP();
		sum  += end - start;
	}

	return sum / i;
}

// Reads and decodes a PNG file
int open_png ( const char* filename, float** out, uint32_t* width, uint32_t* height, uint32_t* channels )
{
	int ret = 0;
	uint8_t* bytes = NULL;

	// Decode PNG File
	ret = lodepng_decode24_file( &bytes, width, height, filename );
	if ( ret )
	{
		fprintf( stderr, "Failed to load png file.\n" );
		goto exit;
	}

	*channels = 3;

	(*out) = (float*) malloc( sizeof(float) * (*width) * (*height) * (*channels) );

	if ( (*out) == NULL )
	{
		ret = -ENOMEM;
		goto exit;
	}

	// Convert to a desired data structure with floats
	int x, y, z;
	for ( y = 0; y < *height; ++y )
	{
		for ( x = 0; x < *width; ++x )
		{
			for ( z = 0; z < *channels; ++z )
			{
				(*out)[ (z * (*width) * (*height)) + (y * (*width)) + (x) ]
				   = (float)bytes[ (y * (*width) * (*channels)) + (x * (*channels)) + z ] / 255.0f;
			}
		}
	}

	exit:
		free( bytes );
		return ret;
}

// Creates filters for testing
int create_filters ( float*** out, uint32_t filter_width, uint32_t filter_height, uint32_t filter_count, uint32_t filter_channels )
{
	int ret = 0;
	uint32_t f, i, j, z;
	uint32_t filters_allocated = 0;

	(*out) = (float**) malloc( sizeof(float*) * filter_count );

	if ( (*out) == NULL )
	{
		ret = -ENOMEM;
		goto fail;
	}

	for ( f = 0; f < filter_count; f++ )
	{
		float* filter = (float*) malloc( sizeof(float) * filter_width * filter_height * filter_channels );

		if ( filter == NULL )
		{
			ret = -ENOMEM;
			goto fail;
		}

		filters_allocated++;

		for ( z = 0; z < filter_channels; ++z )
		{
			for ( i = 0; i < filter_height; ++i )
			{
				for ( j = 0; j < filter_width; ++j )
				{
					filter[ (z * filter_height * filter_width) + (i * filter_width) + j ] = 1.0f/(float)(filter_width*filter_height);//((float) rand()) / ((float) RAND_MAX );
				}
			}
		}

		(*out)[ filters_allocated - 1 ] = filter;
	}

	exit:
		return ret;
	fail:
		for ( i = 0; i < filters_allocated; ++i )
			free( (*out)[i] );
		free( (*out) );
		return ret;
}

// Creates input images for testing
int create_image ( float** out, uint32_t out_width, uint32_t out_height, uint32_t out_channels )
{
	int ret = 0;
	uint32_t i, j, z;

	(*out) = (float*) malloc( sizeof(float) * out_width * out_height * out_channels );

	if ( (*out) == NULL )
	{
		ret = -ENOMEM;
		goto fail;
	}

	for ( z = 0; z < out_channels; ++z )
	{
		for ( i = 0; i < out_height; ++i )
		{
			for ( j = 0; j < out_width; ++j )
			{
				(*out)[ (z * out_height * out_width) + (i * out_width) + j ] = ((float) rand()) / ((float) RAND_MAX );
			}
		}
	}

	exit:
		return ret;
	fail:
		free( (*out) );
		return ret;
}

// Image to Column Matrix format
int im2col
(
	float* in, uint32_t in_width, uint32_t in_height, uint32_t in_channels,
	uint32_t filter_width, uint32_t filter_height, uint32_t filter_count, uint32_t sigmaw, uint32_t sigmah,
	float** out, uint32_t* out_rows, uint32_t* out_cols
)
{
	int ret = 0;

	// Calculate out dimensions
	uint32_t out_width = ( ( ( in_width - filter_width ) / sigmaw ) + 1 );
	uint32_t out_height = ( ( ( in_height - filter_height ) / sigmah ) + 1 );
	*out_cols = in_channels * filter_width * filter_height;
	*out_rows = out_height * out_width;

	// Allocate out memory
	(*out) = (float*) malloc( sizeof(float) * (*out_rows) * (*out_cols) );

	if ( (*out) == NULL )
	{
		ret = -ENOMEM;
		goto exit;
	}

	uint32_t z, x, y, i, j;

	for ( z = 0; z < in_channels; ++z )
	{
		for ( y = 0; y < in_height - filter_height + 1; y += sigmah )
		{
			for ( x = 0; x < in_width - filter_width + 1; x += sigmaw )
			{
				for ( j = 0; j < filter_height; ++j )
				{
					for ( i = 0; i < filter_width; ++i )
					{
						(*out)[ ( ( y*out_width + x ) * (*out_cols) )
								+ ( (z*filter_width*filter_height) + (j*filter_width) + (i) ) ]
								= in[ (z*in_width*in_height) + ((y+j)*in_width) + ((x+i)) ];
					}
				}
			}
		}
	}

	exit:
		return ret;
}

// Kernels to Column Matrix format (?format name?)
int ker2col
(
	float** filters, uint32_t filter_width, uint32_t filter_height, uint32_t filter_count, uint32_t filter_channels,
	float** out, uint32_t* out_rows, uint32_t* out_cols
)
{
	int ret = 0;

	// Calculate out dimensions
	*out_rows = filter_channels * filter_width * filter_height;
	*out_cols = filter_count;

	// Allocate out memory
	(*out) = (float*) malloc( sizeof(float) * (*out_rows) * (*out_cols) );

	if ( (*out) == NULL )
	{
		ret = -ENOMEM;
		goto exit;
	}

	uint32_t f, i, j, z;

	for ( z = 0; z < filter_channels; ++ z )
	{
		for ( i = 0; i < filter_height; ++i )
		{
			for ( j = 0; j < filter_width; ++j )
			{
				for ( f = 0; f < filter_count; ++f )
				{
					(*out)[ ( (z*filter_height*filter_width + i*filter_width + j) * (*out_cols) ) + f ]
						= filters[f][ z*filter_height*filter_width + i*filter_width + j ];
				}
			}
		}
	}

	exit:
		return ret;
}

// This whole function is a hack
// just want to verify that im2col and ker2col and matrix multiply working properly
int col2im
(
	float* in, uint32_t in_rows, uint32_t in_cols,
	float** out, uint32_t* out_width, uint32_t* out_height, uint32_t* out_channels
)
{
	int ret = 0;

	// Calculate out dimensions
	*out_channels = in_cols;
	*out_width    = 252; //TODO: FIX THE HACK
	*out_height   = 252; //TODO: FIX THE HACK

	// Allocate out memory
	(*out) = (float*) malloc( sizeof(float) * (*out_channels) * (*out_width) * (*out_height) );

	if ( (*out) == NULL )
	{
		ret = -ENOMEM;
		goto exit;
	}

	uint32_t x, y;

	for ( y = 0; y < in_cols; ++y )
	{
		for ( x = 0; x < in_rows; ++x )
		{
			(*out)[ (y*(*out_width)*(*out_height)) + x ] = in[ x*in_cols + y ];
		}
	}

	exit:
		return ret;
}

// Prints a PNG file
int print_png ( const char* filename, float* out, uint32_t width, uint32_t height, uint32_t channels )
{
	int ret = 0;
	uint8_t* bytes = NULL;

	bytes = (uint8_t*) malloc( sizeof(uint8_t) * width * height * channels );

	if ( bytes == NULL )
	{
		ret = -ENOMEM;
		goto exit;
	}

	int x, y, z;
	for ( y = 0; y < height; ++y )
	{
		for ( x = 0; x < width; ++x )
		{
			for ( z = 0; z < channels; ++z )
			{
				bytes[ (y * width * channels) + (x * channels) + (z) ]
				   = (uint8_t) (out[ (z * width * height) + (y * width) + (x)  ] * 255.0f);
			}
		}
	}

	ret = lodepng_encode24_file( filename, bytes, width, height );

	if ( ret )
	{
		fprintf( stderr, "Failed to print png file.\n" );
		goto exit;
	}

	exit:
		free( bytes );
		return ret;
}

int verify ( const char* in_filename, const char* out_filename, const char* out_filename2, const char* out_filename3 )
{
	// read image -> in
	// create filters -> filters
	// convolve( in, filters ) -> out
	// write image <- out

	uint32_t i;
	int ret = 0;

	// Read Image
	uint32_t in_width;
	uint32_t in_height;
	uint32_t in_channels;
	float*   in = NULL;

	if ( open_png( in_filename, &in, &in_width, &in_height, &in_channels ) )
	{
		fprintf( stderr, "Failed to open file.\n" );
		ret = -1;
		goto fail0;
	}

	// Create Filters
	uint32_t filter_width  = __R;
	uint32_t filter_height = __S;
	uint32_t filter_count  = in_channels;
	float**  filters       = NULL;

	if ( create_filters( &filters, filter_width, filter_height, filter_count, in_channels ) )
	{
		fprintf( stderr, "Failed to create filters.\n" );
		ret = -1;
		goto fail1;
	}

	// Allocate Out
	uint32_t out_width    = (in_width  - filter_width ) / __SIGMAW + 1;
	uint32_t out_height   = (in_height - filter_height) / __SIGMAH + 1;
	uint32_t out_channels = filter_count;
	float*   out          = (float*) malloc( sizeof(float) * out_channels * out_height * out_width );

	if ( out == NULL )
	{
		fprintf( stderr, "Failed to allocate out matrix.\n" );
		ret = -ENOMEM;
		goto fail2;
	}

	// Verify convolve_std
	memset( out, 0, sizeof(float) * out_channels * out_height * out_width );
    convolve_std( in, out, filters, in_channels, out_channels, out_width, out_height, filter_width, filter_height, __SIGMAW, __SIGMAH );
	
	if ( print_png( out_filename, out, out_width, out_height, out_channels ) )
	{
		fprintf( stderr, "Failed to print png.\n" );
		ret = -1;
		goto fail3;
	}

	// Verify convolve cacnn
	memset( out, 0, sizeof(float) * out_channels * out_height * out_width );
    convolve_cacnn( in, out, filters, in_channels, out_channels, out_width,
	                out_height, filter_width, filter_height, __SIGMAW, __SIGMAH,
	                __C_B, __K_B, __W_B, __H_B, __RP_B, __RPP_B, __SP_B, __SPP_B );

	if ( print_png( out_filename2, out, out_width, out_height, out_channels ) )
	{
		fprintf( stderr, "Failed to print png.\n" );
		ret = -1;
		goto fail3;
	}

	// Verify im2col
	uint32_t in_rows;
	uint32_t in_cols;
	float*   in_matrix = NULL;

	if ( im2col( in, in_width, in_height, in_channels, filter_width, filter_height, filter_count, __SIGMAW, __SIGMAH, &in_matrix, &in_rows, &in_cols ) )
	{
		fprintf( stderr, "Failed to convert image to matrix.\n" );
		ret = -1;
		goto fail4;
	}

	// Convert filters to kernel matrix
	uint32_t ker_rows;
	uint32_t ker_cols;
	float*   ker_matrix = NULL;

	if ( ker2col( filters, filter_width, filter_height, filter_count, in_channels, &ker_matrix, &ker_rows, &ker_cols ) )
	{
		fprintf( stderr, "Failed to convert kernels to matrix.\n" );
		ret = -1;
		goto fail5;
	}

	// Multiply matrices
	uint32_t out_rows = in_rows;
	uint32_t out_cols = ker_cols;
	float*   out_matrix = NULL;

	out_matrix = (float*) malloc( sizeof(float) * out_rows * out_cols );

	if ( out_matrix == NULL )
	{
		ret = -ENOMEM;
		goto fail6;
	}

	multiply( in_rows, ker_rows, ker_cols, in_matrix, ker_matrix, out_matrix, 100 );

	uint32_t col_width;
	uint32_t col_height;
	uint32_t col_channels;
	float*   col_image = NULL;

	if ( col2im( out_matrix, out_rows, out_cols, &col_image, &col_width, &col_height, &col_channels ) )
	{
		fprintf( stderr, "Failed to revert matrix format to image format.\n" );
		ret = -1;
		goto fail5;
	}


	if ( print_png ( out_filename3, col_image, col_width, col_height, col_channels ) )
	{
		fprintf( stderr, "Failed to print reverted png.\n" );
		ret = -1;
		goto fail6;
	}

	fail6:
	fail5:
	fail4:
	fail3:
		free( out );

	fail2:
		for ( i = 0; i < filter_count; ++i )
			free( filters[i] );
		free( filters );

	fail1:
		free( in );

	fail0:
		return ret;
}

int time ( void )
{
	// Test Prep
		// Create Input
		// Create Filters
		// Allocate Out
		// Create Input_Matrix
		// Create Ker_Matrix
		// Allocate Out_Matrix
		// Create Junk Array
		// Record time correction
		// Pin thread to core
		// Create Data Space
	// For each algorithm in {im2col, convolve_std, convolve_cacnn}
		// Repeat TRIALS times
			// Load All Data (touch input, zero output)
			// Load Junk Data (clear L2)
			// Start time
			// Run Algorithm
			// Stop time
	// Print results

	int ret;
	uint32_t i = 0;
	volatile float checksum = 0.0f;

	// Test Prep
	// Create Input
	uint32_t in_width     = __SIGMAW * ( __W - 1) + __R;
	uint32_t in_height    = __SIGMAH * ( __H - 1) + __S;
	uint32_t in_channels  = __C;
	float*   in = NULL;

	if ( create_image ( &in, in_width, in_height, in_channels ) )
	{
		fprintf( stderr, "Failed to create input image.\n" );
		ret = -1;
		goto fail0;
	}

	// Create Filters
	uint32_t filter_width    = __R;
	uint32_t filter_height   = __S;
	uint32_t filter_channels = in_channels;
	uint32_t filter_count    = __K;
	float**  filters         = NULL;

	if ( create_filters( &filters, filter_width, filter_height, filter_count, filter_channels ) )
	{
		fprintf( stderr, "Failed to create filters.\n" );
		ret = -1;
		goto fail1;
	}

	// Allocate Out
	uint32_t out_width    = __W;
	uint32_t out_height   = __H;
	uint32_t out_channels = __K;
	float*   out          = (float*) malloc( sizeof(float) * out_channels * out_height * out_width );

	if ( out == NULL )
	{
		fprintf( stderr, "Failed to allocate out matrix.\n" );
		ret = -ENOMEM;
		goto fail2;
	}

	// Create Input_Matrix
	uint32_t in_rows;
	uint32_t in_cols;
	float*   in_matrix = NULL;

	if ( im2col( in, in_width, in_height, in_channels, filter_width, filter_height, filter_count, __SIGMAW, __SIGMAH, &in_matrix, &in_rows, &in_cols ) )
	{
		fprintf( stderr, "Failed to convert image to matrix.\n" );
		ret = -1;
		goto fail3;
	}

	// Create Ker_Matrix
	uint32_t ker_rows;
	uint32_t ker_cols;
	float*   ker_matrix = NULL;

	if ( ker2col( filters, filter_width, filter_height, filter_count, in_channels, &ker_matrix, &ker_rows, &ker_cols ) )
	{
		fprintf( stderr, "Failed to convert kernels to matrix.\n" );
		ret = -1;
		goto fail4;
	}

	// Allocate Out_Matrix
	uint32_t out_rows = in_rows;
	uint32_t out_cols = ker_cols;
	float*   out_matrix = NULL;

	out_matrix = (float*) malloc( sizeof(float) * out_rows * out_cols );

	if ( out_matrix == NULL )
	{
		ret = -ENOMEM;
		goto fail5;
	}

	// Create Junk Array
	float* junk_l2 = (float*) malloc( L2_SIZE*4 );

	if ( junk_l2 == NULL )
	{
		fprintf( stderr, "Failed to create junk array.\n" );
		ret = -ENOMEM;
		goto fail6;
	}

	// Record time correction
	uint64_t correction = fipc_test_time_get_correction();

	// Pin thread to core
//	pfcInit();			// Init Perf Counter
//	pfcPinThread(3);	// Pin to core number 3

	// Create Data Space
	uint64_t* data_std    = (uint64_t*) malloc( sizeof(uint64_t) * TRIALS );
	uint64_t* data_cacnn  = (uint64_t*) malloc( sizeof(uint64_t) * TRIALS );
	uint64_t* data_pcacnn  = (uint64_t*) malloc( sizeof(uint64_t) * TRIALS );
	uint64_t* data_im2col = (uint64_t*) malloc( sizeof(uint64_t) * TRIALS );


	// For each algorithm in {im2col, convolve_std, convolve_cacnn}: convolve_std
	// Repeat TRIALS times
	uint64_t a, b, c, d;
	for ( i = 0; i < TRIALS; ++i )
	{
		// Load All Data (touch input, zero output)
		for ( c = 0; c < in_channels; ++c )
			for ( b = 0; b < in_height; ++b )
				for ( a = 0; a < in_width; ++a )
					checksum += in[ c*(in_width*in_height) + b*(in_width) + a ];

		for ( d = 0; d < filter_count; ++d )
			for ( c = 0; c < filter_channels; ++c )
				for ( b = 0; b < filter_height; ++b )
					for ( a = 0; a < filter_width; ++a )
						checksum += filters[d][ c*(filter_width*filter_height) + a*filter_width + b ];

		memset( out, 0, out_channels * out_width * out_height );

		// Load Junk Data (clear L2)
		for ( a = 0; a < L2_SIZE; ++a )
			checksum += junk_l2[a];

		// Start time
		volatile uint64_t start = RDTSC_START();

		// Run Algorithm
		//#pragma omp parallel for
		for ( b = 0; b < 10; ++b )
		{
			convolve_std( in, out, filters, in_channels, out_channels, out_width, out_height, filter_width, filter_height, __SIGMAW, __SIGMAH );

		}		
		// Stop time
		volatile uint64_t end = RDTSCP();
		data_std[i] = (end - start) - correction;
	}

	// For each algorithm in {im2col, convolve_std, convolve_cacnn}: convolve_cacnn
	for ( i = 0; i < TRIALS; ++i )
	{
		// Load All Data (touch input, zero output)
		for ( c = 0; c < in_channels; ++c )
			for ( b = 0; b < in_height; ++b )
				for ( a = 0; a < in_width; ++a )
					checksum += in[ c*(in_width*in_height) + b*(in_width) + a ];

		for ( d = 0; d < filter_count; ++d )
			for ( c = 0; c < filter_channels; ++c )
				for ( b = 0; b < filter_height; ++b )
					for ( a = 0; a < filter_width; ++a )
						checksum += filters[d][ c*(filter_width*filter_height) + a*filter_width + b ];

		memset( out, 0, out_channels * out_width * out_height );

		// Load Junk Data (clear L2)
		for ( a = 0; a < L2_SIZE; ++a )
			checksum += junk_l2[a];

		// Start time
		volatile uint64_t start = RDTSC_START();

		// Run Algorithm
		#pragma omp parallel for
		for ( b = 0; b < 10; ++b )
		{
			convolve_cacnn( in, out, filters, in_channels, out_channels, out_width,
							out_height, filter_width, filter_height, __SIGMAW, __SIGMAH,
							__C_B, __K_B, __W_B, __H_B, __RP_B, __RPP_B, __SP_B, __SPP_B );
		}

		// Stop time
		volatile uint64_t end = RDTSCP();
		data_cacnn[i] = (end - start) - correction;
	}
	// For each algorithm in {im2col, convolve_std, convolve_cacnn}: im2col
	for ( i = 0; i < TRIALS; ++i )
	{
		// Load All Data (touch input, zero output)
		for ( c = 0; c < in_rows; ++c )
			for ( b = 0; b < in_cols; ++b )
					checksum += in_matrix[ c*(in_cols) + b ];

		for ( c = 0; c < ker_rows; ++c )
			for ( b = 0; b < ker_cols; ++b )
					checksum += ker_matrix[ c*(ker_cols) + b ];

		memset( out_matrix, 0, out_rows * out_cols );

		// Load Junk Data (clear L2)
		for ( a = 0; a < L2_SIZE; ++a )
			checksum += junk_l2[a];

		// Start time
		volatile uint64_t start = RDTSC_START();

		// Run Algorithm

		multiply( in_rows, ker_rows, ker_cols, in_matrix, ker_matrix, out_matrix, 10 );

		// Stop time
		volatile uint64_t end = RDTSCP();
		data_im2col[i] = (end - start) - correction;
	}

	// Print results
	printf( "Standard Convolution Timing (Cycles)\n" );
	constats_get_and_print_stats( data_std, TRIALS );

	printf( "Communication Avoiding Convolution Timing (Cycles)\n" );
	constats_get_and_print_stats( data_cacnn, TRIALS );

	printf( "Im2Col Matrix Multiplication Timing (Cycles)\n" );
	constats_get_and_print_stats( data_im2col, TRIALS );

	fail7:
		free( data_std );
		free( data_cacnn );
		free( data_im2col );
	fail6:
		free( out_matrix );
	fail5:
		free( ker_matrix );
	fail4:
		free( in_matrix );
	fail3:
		free( out );
	fail2:
		for ( i = 0; i < filter_count; ++i )
			free( filters[i] );
		free( filters );
	fail1:
		free( in );
	fail0:
		return ret;
}
/**
int count_misses ( void )
{
	// Test Prep
		// Create Input
		// Create Filters
		// Allocate Out
		// Create Input_Matrix
		// Create Ker_Matrix
		// Allocate Out_Matrix
		// Create Junk Array
		// Pin thread to core
		// Create Data Space
	// For each algorithm in {im2col, convolve_std, convolve_cacnn}
		// Repeat TRIALS times
			// Load All Data (touch input, zero output)
			// Load Junk Data (clear L2)
			// Start perf counter
			// Run Algorithm
			// Stop perf counter
	// Print results

	int ret;
	uint32_t i = 0;
	volatile float checksum = 0.0f;

	// Test Prep
	// Create Input
	uint32_t in_width     = __SIGMAW * ( __W - 1) + __R;
	uint32_t in_height    = __SIGMAH * ( __H - 1) + __S;
	uint32_t in_channels  = __C;
	float*   in = NULL;

	if ( create_image ( &in, in_width, in_height, in_channels ) )
	{
		fprintf( stderr, "Failed to create input image.\n" );
		ret = -1;
		goto fail0;
	}

	// Create Filters
	uint32_t filter_width    = __R;
	uint32_t filter_height   = __S;
	uint32_t filter_channels = in_channels;
	uint32_t filter_count    = __K;
	float**  filters         = NULL;

	if ( create_filters( &filters, filter_width, filter_height, filter_count, filter_channels ) )
	{
		fprintf( stderr, "Failed to create filters.\n" );
		ret = -1;
		goto fail1;
	}

	// Allocate Out
	uint32_t out_width    = __W;
	uint32_t out_height   = __H;
	uint32_t out_channels = __K;
	float*   out          = (float*) malloc( sizeof(float) * out_channels * out_height * out_width );

	if ( out == NULL )
	{
		fprintf( stderr, "Failed to allocate out matrix.\n" );
		ret = -ENOMEM;
		goto fail2;
	}

	// Create Input_Matrix
	uint32_t in_rows;
	uint32_t in_cols;
	float*   in_matrix = NULL;

	if ( im2col( in, in_width, in_height, in_channels, filter_width, filter_height, filter_count, __SIGMAW, __SIGMAH, &in_matrix, &in_rows, &in_cols ) )
	{
		fprintf( stderr, "Failed to convert image to matrix.\n" );
		ret = -1;
		goto fail3;
	}

	// Create Ker_Matrix
	uint32_t ker_rows;
	uint32_t ker_cols;
	float*   ker_matrix = NULL;

	if ( ker2col( filters, filter_width, filter_height, filter_count, in_channels, &ker_matrix, &ker_rows, &ker_cols ) )
	{
		fprintf( stderr, "Failed to convert kernels to matrix.\n" );
		ret = -1;
		goto fail4;
	}

	// Allocate Out_Matrix
	uint32_t out_rows = in_rows;
	uint32_t out_cols = ker_cols;
	float*   out_matrix = NULL;

	out_matrix = (float*) malloc( sizeof(float) * out_rows * out_cols );

	if ( out_matrix == NULL )
	{
		ret = -ENOMEM;
		goto fail5;
	}

	// Create Junk Array
	float* junk_l2 = (float*) malloc( L2_SIZE*4 );

	if ( junk_l2 == NULL )
	{
		fprintf( stderr, "Failed to create junk array.\n" );
		ret = -ENOMEM;
		goto fail6;
	}

	// Record time correction
	uint64_t correction = fipc_test_time_get_correction();

	// Pin thread to core
	pfcInit();			// Init Perf Counter
	pfcPinThread(3);	// Pin to core number 3

	// Create Data Space
	uint64_t* data_std    = (uint64_t*) malloc( sizeof(uint64_t) * TRIALS );
	uint64_t* data_cacnn  = (uint64_t*) malloc( sizeof(uint64_t) * TRIALS );
	uint64_t* data_im2col = (uint64_t*) malloc( sizeof(uint64_t) * TRIALS );

	uint64_t* data2_std    = (uint64_t*) malloc( sizeof(uint64_t) * TRIALS );
	uint64_t* data2_cacnn  = (uint64_t*) malloc( sizeof(uint64_t) * TRIALS );
	uint64_t* data2_im2col = (uint64_t*) malloc( sizeof(uint64_t) * TRIALS );

	// Init Perf Counters
	PFC_CNT  CNT[7] = {0,0,0,0,0,0,0};
	PFC_CFG  CFG[7] = {0,0,0,0,0,0,0};
	CFG[3] = pfcParseCfg( "l2_rqsts.demand_data_rd_miss" );
	CFG[4] = pfcParseCfg( "llc.miss" );
	pfcWrCfgs(0, 7, CFG);
	pfcWrCnts(0, 7, CNT);
	memset(CNT, 0, sizeof(CNT));

	// For each algorithm in {im2col, convolve_std, convolve_cacnn}: convolve_std
	// Repeat TRIALS times
	uint64_t a, b, c, d;
	for ( i = 0; i < TRIALS; ++i )
	{
		// Load All Data (touch input, zero output)
		for ( c = 0; c < in_channels; ++c )
			for ( b = 0; b < in_height; ++b )
				for ( a = 0; a < in_width; ++a )
					checksum += in[ c*(in_width*in_height) + b*(in_width) + a ];

		for ( d = 0; d < filter_count; ++d )
			for ( c = 0; c < filter_channels; ++c )
				for ( b = 0; b < filter_height; ++b )
					for ( a = 0; a < filter_width; ++a )
						checksum += filters[d][ c*(filter_width*filter_height) + a*filter_width + b ];

		memset( out, 0, out_channels * out_width * out_height );

		// Load Junk Data (clear L2)
		for ( a = 0; a < L2_SIZE; ++a )
			checksum += junk_l2[a];

		// Start Perf Counter
		pfcWrCnts(0, 7, CNT);
		memset(CNT, 0, sizeof(CNT));
		PFCSTART(CNT);

		// Run Algorithm
		convolve_std( in, out, filters, in_channels, out_channels, out_width, out_height, filter_width, filter_height, __SIGMAW, __SIGMAH );

		// Stop Perf Counter
		PFCEND(CNT);
		pfcRemoveBias(CNT, 1);
		data_std[i]  = CNT[3];
		data2_std[i] = CNT[4];
	}

	// For each algorithm in {im2col, convolve_std, convolve_cacnn}: convolve_cacnn
	for ( i = 0; i < TRIALS; ++i )
	{
		// Load All Data (touch input, zero output)
		for ( c = 0; c < in_channels; ++c )
			for ( b = 0; b < in_height; ++b )
				for ( a = 0; a < in_width; ++a )
					checksum += in[ c*(in_width*in_height) + b*(in_width) + a ];

		for ( d = 0; d < filter_count; ++d )
			for ( c = 0; c < filter_channels; ++c )
				for ( b = 0; b < filter_height; ++b )
					for ( a = 0; a < filter_width; ++a )
						checksum += filters[d][ c*(filter_width*filter_height) + a*filter_width + b ];

		memset( out, 0, out_channels * out_width * out_height );

		// Load Junk Data (clear L2)
		for ( a = 0; a < L2_SIZE; ++a )
			checksum += junk_l2[a];

		// Start Perf Counter
		pfcWrCnts(0, 7, CNT);
		memset(CNT, 0, sizeof(CNT));
		PFCSTART(CNT);

		// Run Algorithm
		convolve_cacnn( in, out, filters, in_channels, out_channels, out_width,
	    	            out_height, filter_width, filter_height, __SIGMAW, __SIGMAH,
	        	        __C_B, __K_B, __W_B, __H_B, __RP_B, __RPP_B, __SP_B, __SPP_B );

		// Stop Perf Counter
		PFCEND(CNT);
		pfcRemoveBias(CNT, 1);
		data_cacnn[i]  = CNT[3];
		data2_cacnn[i] = CNT[4];
	}

	// For each algorithm in {im2col, convolve_std, convolve_cacnn}: im2col
	for ( i = 0; i < TRIALS; ++i )
	{
		// Load All Data (touch input, zero output)
		for ( c = 0; c < in_rows; ++c )
			for ( b = 0; b < in_cols; ++b )
					checksum += in_matrix[ c*(in_cols) + b ];

		for ( c = 0; c < ker_rows; ++c )
			for ( b = 0; b < ker_cols; ++b )
					checksum += ker_matrix[ c*(ker_cols) + b ];

		memset( out_matrix, 0, out_rows * out_cols );

		// Load Junk Data (clear L2)
		for ( a = 0; a < L2_SIZE; ++a )
			checksum += junk_l2[a];

		// Start Perf Counter
		pfcWrCnts(0, 7, CNT);
		memset(CNT, 0, sizeof(CNT));
		PFCSTART(CNT);

		// Run Algorithm
		multiply( in_rows, ker_rows, ker_cols, in_matrix, ker_matrix, out_matrix, 10 );

		// Stop Perf Counter
		PFCEND(CNT);
		pfcRemoveBias(CNT, 1);
		data_im2col[i]  = CNT[3];
		data2_im2col[i] = CNT[4];
	}

	// Print results
	printf( "Standard Convolution L2 Misses\n" );
	constats_get_and_print_stats( data_std, TRIALS );

	printf( "Communication Avoiding Convolution L2 Misses\n" );
	constats_get_and_print_stats( data_cacnn, TRIALS );

	printf( "Im2Col Matrix Multiplication L2 Misses\n" );
	constats_get_and_print_stats( data_im2col, TRIALS );

	printf( "Standard Convolution LLC Misses\n" );
	constats_get_and_print_stats( data2_std, TRIALS );

	printf( "Communication Avoiding Convolution LLC Misses\n" );
	constats_get_and_print_stats( data2_cacnn, TRIALS );

	printf( "Im2Col Matrix Multiplication LLC Misses\n" );
	constats_get_and_print_stats( data2_im2col, TRIALS );

	fail7:
		free( data_std );
		free( data_cacnn );
		free( data_im2col );
		free( data2_std );
		free( data2_cacnn );
		free( data2_im2col );
	fail6:
		free( out_matrix );
	fail5:
		free( ker_matrix );
	fail4:
		free( in_matrix );
	fail3:
		free( out );
	fail2:
		for ( i = 0; i < filter_count; ++i )
			free( filters[i] );
		free( filters );
	fail1:
		free( in );
	fail0:
		return ret;
}
*/
int main ( int argc, const char* argv[] )
{
	printf("hello \n");
	if ( argc < 17 )
	{
		fprintf( stderr, "Need to specifiy parameters.\n" );
		return -1;
	}

	 __C       = atoi(argv[1]);
	 __K       = atoi(argv[2]);
	 __W       = atoi(argv[3]);
	 __H       = atoi(argv[4]);
	 __R       = atoi(argv[5]);
	 __S       = atoi(argv[6]);
	 __SIGMAW  = atoi(argv[7]);
	 __SIGMAH  = atoi(argv[8]);

	 __C_B     = atoi(argv[9]);
	 __K_B     = atoi(argv[10]);
	 __W_B     = atoi(argv[11]);
	 __H_B     = atoi(argv[12]);
	 __RP_B    = atoi(argv[13]);
	 __RPP_B   = atoi(argv[14]);
	 __SP_B    = atoi(argv[15]);
	 __SPP_B   = atoi(argv[16]);
	
	//verify( "data/dog.png", "ver_convolve.png", "ver_cacnn.png", "ver_carma.png" );
	time();
//	count_misses();
	return 0;
}
