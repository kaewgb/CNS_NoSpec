#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "util.h"

#define	BLOCK_DIM	16

__global__ void gpu_readwrite(){

}

void readwrite(){
	dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
	dim3 grid_dim(CEIL(n, BLOCK_DIM), CEIL(n, BLOCK_DIM));

	gpu_readwrite<<<grid_dim, block_dim>>>();
}
