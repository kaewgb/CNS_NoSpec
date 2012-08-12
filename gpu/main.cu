#include <stdio.h>
#include <cuda.h>
#include "header.h"
#include "helper_functions.h"

global_const_t h_const;
global_const_t *d_const_ptr;
__constant__ global_const_t d_const;
__constant__ kernel_const_t kc;

int main(int argc, char *argv[]){

	int i;
	char *dest;

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

	h_const.OneThird		= 1.0E0/3.0E0;
	h_const.TwoThirds		= 2.0E0/3.0E0;
	h_const.FourThirds		= 4.0E0/3.0E0;
	h_const.OneQuarter    	= 1.E0/4.E0;
    h_const.ThreeQuarters 	= 3.E0/4.E0;

	h_const.CENTER		= -205.0E0/72.0E0;
	h_const.OFF1		=  8.0E0/5.0E0;
	h_const.OFF2 		= -0.2E0;
	h_const.OFF3		=  8.0E0/315.0E0;
	h_const.OFF4		= -1.0E0/560.0E0;

	cudaGetSymbolAddress((void **) &(h_const.kc), kc);

	cudaMemcpyToSymbol(d_const, &h_const, sizeof(global_const_t));
	cudaGetSymbolAddress((void **) &d_const_ptr, d_const);

	dest = (char *)d_const_ptr + ((char *)&h_const.lo - (char *)&h_const);
	cudaMemcpy((int *) dest, h_const.lo, 3*sizeof(int), cudaMemcpyHostToDevice);
	dest = (char *)d_const_ptr + ((char *)&h_const.hi - (char *)&h_const);
	cudaMemcpy((int *) dest, h_const.hi, 3*sizeof(int), cudaMemcpyHostToDevice);
	dest = (char *)d_const_ptr + ((char *)&h_const.dim - (char *)&h_const);
	cudaMemcpy((int *) dest, h_const.dim, 3*sizeof(int), cudaMemcpyHostToDevice);
	dest = (char *)d_const_ptr + ((char *)&h_const.dim_g - (char *)&h_const);
	cudaMemcpy((int *) dest, h_const.dim_g, 3*sizeof(int), cudaMemcpyHostToDevice);

	dest = (char *)d_const_ptr + ((char *)&h_const.dx - (char *)&h_const);
	cudaMemcpy((int *) dest, h_const.dx, 3*sizeof(double), cudaMemcpyHostToDevice);
	dest = (char *)d_const_ptr + ((char *)&h_const.dxinv - (char *)&h_const);
	cudaMemcpy((int *) dest, h_const.dxinv, 3*sizeof(double), cudaMemcpyHostToDevice);

	printf("alloc size = %d\n", h_const.dim_g[0]*h_const.dim_g[1]*h_const.dim_g[2] * 30);
	// Calling Test Kernels
//	ctoprim_test(h_const, d_const_ptr);
//	diffterm_test(h_const, d_const_ptr);
//	hypterm_test(h_const, d_const_ptr);
//	fill_boundary_test(h_const, d_const_ptr);

//	advance_hybrid_test(h_const, d_const_ptr);
	advance_test(h_const, d_const_ptr);

	int dim[3], dim_g[3], nc=NC;
	FOR(i, 0, 3){
		dim[i] = h_const.dim[i];
		dim_g[i] = h_const.dim_g[i];
	}

	double ****U, ****Unew, ****Q, ****D, ****F;
	double ****U2, ****Unew2, ****Q2, ****D2, ****F2;

	// Allocation
	allocate_4D(U,  	dim_g, 	nc);
	allocate_4D(Unew,  	dim_g, 	nc);
	allocate_4D(Q,  	dim_g, 	nc+1);
	allocate_4D(D,  	dim, 	nc);
	allocate_4D(F, 		dim, 	nc);

	allocate_4D(U2,  	dim_g, 	nc);
	allocate_4D(Unew2,  dim_g, 	nc);
	allocate_4D(Q2,  	dim_g, 	nc+1);
	allocate_4D(D2,  	dim, 	nc);
	allocate_4D(F2,		dim, 	nc);

//	advance_test(U, Unew, Q, D, F);
//	advance_hybrid_test(h_const, d_const_ptr, U2, Unew2, Q2, D2, F2);

//	check_4D_array("Q",	Q, Q2, dim_g, nc+1);
//	int k;
//	FOR(k, 0, 10)
//		printf("%le\n", D2[0][0][0][k]);
//	check_4D_array("D", D, D2, dim, nc);


	free_4D(D, 		dim, 	nc);
	free_4D(D2,		dim, 	nc);

	return 0;

}

