#include <stdio.h>
#include <cuda.h>
#include "header.h"

global_const_t h_const;
global_const_t *d_const_ptr;
__constant__ global_const_t d_const;

int main(int argc, char *argv[]){

	int i;

	// Prepare Global Constants
	FILE *fin = fopen("../testcases/general_input", "r");
	fscanf(fin, "%d", &h_const.ng);
	fscanf(fin, "%d %d %d", &h_const.lo[0], &h_const.lo[1], &h_const.lo[2]);
	fscanf(fin, "%d %d %d", &h_const.hi[0], &h_const.hi[1], &h_const.hi[2]);
	fscanf(fin, "%le %le %le", &h_const.dx[0], &h_const.dx[1], &h_const.dx[2]);
	fscanf(fin, "%le", &h_const.cfl);
	fscanf(fin, "%le", &h_const.eta);
	fscanf(fin, "%le", &h_const.alam);
	fclose(fin);
	FOR(i, 0, 3){
		h_const.dim[i] 		= h_const.hi[i] - h_const.lo[i] + 1;
		h_const.dim_ng[i] 	= h_const.hi[i] - h_const.lo[i] + 1 + h_const.ng + h_const.ng;
	}
	h_const.comp_offset_ng = h_const.dim_ng[0] * h_const.dim_ng[1] * h_const.dim_ng[2];

	cudaMemcpyToSymbol(d_const, &h_const, sizeof(global_const_t));
	cudaGetSymbolAddress((void **) &d_const_ptr, d_const);

	// Calling Test Kernels
//	ctoprim_test(h_const, d_const_ptr);
//	diffterm_test(h_const, d_const_ptr);
	hypterm_test(h_const, d_const_ptr);

//	advance_test();

	return 0;

}

