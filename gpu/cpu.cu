#include <stdio.h>
#include <cuda.h>
#include "header.h"
#include "util.h"

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
	read_configurations(h_const, argc, argv);
	copy_configurations(h_const, d_const_ptr);

	//!
	//! Allocation
	//!
	allocate_variables(U, Unew, Q, D, F, d_U, d_Unew, d_Q, d_D, d_F, false, true);

	//!
	//! Advance
	//!
	fin = fopen(h_const.input_file_name, "r");
	FOR(l, 0, h_const.nc)
		read_3D(fin, U, h_const.dim_g, l);
	fclose(fin);
//	init_data(h_const, U);

	total_time = -get_time();
	FOR(i, 0, h_const.nsteps)
		advance(h_const, U, Unew, Q, D, F, dt);

	total_time += get_time();
	printf("Total time: %lf\n", total_time);


	fout = fopen("correct", "w");
	fprintf(fout, "%d\n", h_const.nc);
	fprintf(fout, "%d %d %d\n", h_const.dim_g[0], h_const.dim_g[1], h_const.dim_g[2]);
	print_4D(fout, U, h_const.dim_g, h_const.nc);
	fclose(fout);


	//!
	//!	Free Allocations
	//!
	free_variables(U, Unew, Q, D, F, d_U, d_Unew, d_Q, d_D, d_F, false, true);

	return 0;

}

