#include <stdio.h>
#include <cuda.h>
#include "header.h"
#include "util.h"

global_const_t h_const;
global_const_t *d_const_ptr;
__constant__ global_const_t d_const;
__constant__ kernel_const_t kc;

enum { CTOPRIM, DIFFTERM, HYPTERM, READWRITE };

int main(int argc, char *argv[]){

	//!
	//! Variable Declaration
	//!
	char *dest;
	FILE *fin, *fout;
	int i, l, kernel;
	float elapsed_time;
	double dt, total_time;
	double ****U, ****Unew, ****Q, ****D, ****F;
	double *d_U, *d_Unew, *d_Q, *d_D, *d_F;
    cudaEvent_t start, stop;

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
	allocate_variables(U, Unew, Q, D, F, d_U, d_Unew, d_Q, d_D, d_F);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	//!
	//! Advance
	//!
	fin = fopen(h_const.input_file_name, "r");
	FOR(l, 0, h_const.nc)
		read_3D(fin, U, h_const.dim_g, l);
	fclose(fin);
//	init_data(h_const, U);

    kernel = READWRITE;
    gpu_copy_from_host_4D(d_U, U, h_const.pitch_g, h_const.nc);

    switch(kernel){
        case CTOPRIM:   break;
        case DIFFTERM:  break;
        case HYPTERM:   break;
        case READWRITE:
            cudaEventRecord(start, 0);
            total_time = -get_time();
            FOR(i, 0, h_const.nsteps)
                gpu_advance(h_const, d_const_ptr, d_U, d_Unew, d_Q, d_D, d_F, dt);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
    }
    cudaEventElapsedTime(&elapsed_time, start, stop);
	gpu_copy_to_host_4D(U, d_U, h_const.pitch_g, h_const.nc);
	printf("Total time: %f\n", elapsed_time);


	fout = fopen("output", "w");
	fprintf(fout, "%d\n", h_const.nc);
	fprintf(fout, "%d %d %d\n", h_const.dim_g[0], h_const.dim_g[1], h_const.dim_g[2]);
	print_4D(fout, U, h_const.dim_g, h_const.nc);
	fclose(fout);

	//!
	//!	Free Allocations
	//!
	free_variables(U, Unew, Q, D, F, d_U, d_Unew, d_Q, d_D, d_F);
	cudaEventDestroy(start);
	cudaEventDestroy(stop;

	return 0;

}


