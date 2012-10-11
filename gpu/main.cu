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
	int i, dim[3], dim_g[3], nc;
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
	//! Calling Test Kernels
	//!
//	ctoprim_test(h_const, d_const_ptr);
//	diffterm_test(h_const, d_const_ptr);
//	hypterm_test(h_const, d_const_ptr);
//	fill_boundary_test(h_const, d_const_ptr);

//	advance_hybrid_test(h_const, d_const_ptr);
	advance_multistep_test(h_const, d_const_ptr, U, Unew, Q, D, F, d_U, d_Unew, d_Q, d_D, d_F);

//	advance_cpu_multistep_test(h_const, U, Unew, Q, D, F);
//	advance_hybrid_test(h_const, d_const_ptr, U2, Unew2, Q2, D2, F2);
//	check_4D_array("D", D, D2, dim, nc);


	//!
	//!	Free Allocations
	//!
	free_variables(U, Unew, Q, D, F, d_U, d_Unew, d_Q, d_D, d_F);

	return 0;

}

