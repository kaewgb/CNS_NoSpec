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
		h_const.dim_g[i] 	= h_const.hi[i] - h_const.lo[i] + 1 + h_const.ng + h_const.ng;
	}
	h_const.comp_offset_g  = h_const.dim_g[0] * h_const.dim_g[1] * h_const.dim_g[2];
	h_const.comp_offset    = h_const.dim[0]   * h_const.dim[1]   * h_const.dim[2];
	h_const.plane_offset_g = h_const.dim_g[1] * h_const.dim_g[2];
	h_const.plane_offset   = h_const.dim[1]   * h_const.dim[2];

	FOR(i, 0, 3)
		h_const.dxinv[i] = 1.0E0/h_const.dx[i];

	h_const.ALP	=  0.8E0;
	h_const.BET	= -0.2E0;
	h_const.GAM	=  4.0E0/105.0E0;
	h_const.DEL	= -1.0E0/280.0E0;

	h_const.OneThird	= 1.0E0/3.0E0;
	h_const.TwoThirds	= 2.0E0/3.0E0;
	h_const.FourThirds	= 4.0E0/3.0E0;

	h_const.CENTER		= -205.0E0/72.0E0;
	h_const.OFF1		=  8.0E0/5.0E0;
	h_const.OFF2 		= -0.2E0;
	h_const.OFF3		=  8.0E0/315.0E0;
	h_const.OFF4		= -1.0E0/560.0E0;

	cudaMemcpyToSymbol(d_const, &h_const, sizeof(global_const_t));
	cudaGetSymbolAddress((void **) &d_const_ptr, d_const);

	// Calling Test Kernels
//	ctoprim_test(h_const, d_const_ptr);
	diffterm_test(h_const, d_const_ptr);
//	hypterm_test(h_const, d_const_ptr);

//	advance_test();

	return 0;

}

