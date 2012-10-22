#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "util.h"

#define BLOCK_DIM		16		// Dimension that doesn't have ghost cells
#define	BLOCK_DIM_G		8		// Dimension that has ghost cells

__device__ kernel_const_t kc;
__global__ void gpu_hypterm_merged_kernel(
	global_const_t *g,	// i:
	double *cons,		// i:
	double *q,			// i:
	double *flux		// o:
){

}
#undef	s_q
#undef 	s_qpres
#undef	s_cons

void gpu_hypterm_merged(
	global_const_t h_const, 	// i: Global struct containing applicatino parameters
	global_const_t *d_const,	// i: Device pointer to global struct containing application paramters
	double *d_cons,				// i:
	double *d_q,				// i:
	double *d_flux				// o: flux
){
	int i, grid_dim;
	kernel_const_t h_kc;
	dim3 block_dim(BLOCK_DIM_G, BLOCK_DIM, 1);

    cudaMemcpyToSymbol(kc, &h_kc, sizeof(kernel_const_t), cudaMemcpyHostToDevice);

	gpu_hypterm_merged_kernel<<<grid_dim, block_dim>>>(d_const, d_cons, d_q, d_flux);

}
