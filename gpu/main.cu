#include <stdio.h>
#include <cuda.h>
#include "header.h"
#include "helper_functions.h"

global_const_t h_const;
global_const_t *d_const_ptr;
__constant__ global_const_t d_const;
__constant__ kernel_const_t kc;

int main(int argc, char *argv[]){

	//!
	//! Variable Declaration
	//!
	char *dest;
	FILE *fin, *fout;
	int i, l;
	double dt, total_time;
	double ****U, ****Unew, ****Q, ****D, ****F;
	double *d_U, *d_Unew, *d_Q, *d_D, *d_F;

	cudaGetSymbolAddress((void **) &d_const_ptr, d_const);
	cudaGetSymbolAddress((void **) &(h_const.kc), kc);

	//!
	//! Prepare Global Constants
	//!
	read_configurations(h_const, d_const_ptr);

	//!
	//! Allocation
	//!
	allocate_variables(U, Unew, Q, D, F, d_U, d_Unew, d_Q, d_D, d_F);

	//!
	//! Advance
	//!
	fin = fopen("../testcases/multistep_input", "r");
	FOR(l, 0, h_const.nc)
		read_3D(fin, U, h_const.dim_g, l);
	fclose(fin);

	total_time = -get_time();
	gpu_copy_from_host_4D(d_U, U, h_const.dim_g, h_const.nc);
	FOR(i, 0, h_const.nsteps)
		gpu_advance(h_const, d_const_ptr, d_U, d_Unew, d_Q, d_D, d_F, dt);
	gpu_copy_to_host_4D(U, d_U, h_const.dim_g, h_const.nc);
	total_time += get_time();
	printf("Total time: %lf\n", total_time);


	fout = fopen("output", "w");
	fprintf(fout, "%d\n", h_const.nc);
	fprintf(fout, "%d %d %d\n", h_const.dim_g[0], h_const.dim_g[1], h_const.dim_g[2]);
	print_4D(fout, U, h_const.dim_g, h_const.nc);
	fclose(fout);

//	gpu_copy_to_host_4D(Q, d_Q, h_const.dim_g, h_const.nc+1);
//
//	fout = fopen("ctoprim.out", "w");
//	fprintf(fout, "%d\n", h_const.nc+1);
//	fprintf(fout, "%d %d %d\n", h_const.dim_g[0], h_const.dim_g[1], h_const.dim_g[2]);
//	print_4D(fout, Q, h_const.dim_g, h_const.nc+1);
//	fclose(fout);
//
//	gpu_copy_to_host_4D(D, d_D, h_const.dim, h_const.nc);
//
//	fout = fopen("diffterm.out", "w");
//	fprintf(fout, "%d\n", h_const.nc);
//	fprintf(fout, "%d %d %d\n", h_const.dim[0], h_const.dim[1], h_const.dim[2]);
//	print_4D(fout, D, h_const.dim, h_const.nc);
//	fclose(fout);

	//!
	//!	Free Allocations
	//!
	free_variables(U, Unew, Q, D, F, d_U, d_Unew, d_Q, d_D, d_F);

	return 0;

}

