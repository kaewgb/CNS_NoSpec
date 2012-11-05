#include <cuda.h>
#include <stdio.h>
#include "header.h"
#include "util.cuh"
#include "util.h"
#define	N	8

global_const_t h_const;
global_const_t *d_const_ptr;

__global__ void gpu_setzero(double *ptr){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	ptr[i] = 0.0;
}
__global__ void testadd(double *a){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	atomicAdd(a+i, 55.5);
}

int main(){
	int i,j,k;
	int dim[3] = {N, N, N};
	double ***a, *d_a;

	allocate_3D(a, dim);
	gpu_allocate_3D(d_a, dim);

	int block_dim = 512;
	int grid_dim = CEIL(block_dim, 512);
	gpu_setzero<<<grid_dim, block_dim>>>(d_a);
	testadd<<<grid_dim, block_dim>>>(d_a);

	gpu_copy_to_host_3D(a, d_a, dim);

	FOR(k, 0, N){
		FOR(j, 0, N){
			FOR(i, 0, N)
				printf("%8.3lf ", a[k][j][i]);
			printf("\n");
		}
	}

	free_3D(a, dim);
	gpu_free_3D(d_a);
	return 0;
}
