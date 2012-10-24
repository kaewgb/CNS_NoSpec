#ifndef	UTIL_INCLUDED
#define UTIL_INCLUDED

#include "util_cpu.h"

// Allocations
void gpu_allocate_3D(double *&d_ptr, int dim[]);
void gpu_copy_from_host_3D(double *dev, double ***host, int dim[]);
void gpu_copy_to_host_3D(double ***host, double *dev, int dim[]);
void gpu_free_3D(double *d_ptr);

void gpu_allocate_4D(double *&d_ptr, int dim[], int dl);
void gpu_copy_from_host_4D(double *dev, double ****host, int dim[], int dl);
void gpu_copy_to_host_4D(double ****host, double *dev, int dim[], int dl);
void gpu_free_4D(double *d_ptr);

void gpu_fill_boundary(
	global_const_t &h_const,	// i:	Global Constants
	global_const_t *d_const,	// i:	Device Pointer to Global Constants
	double *d_ptr		 		// i/o: Device Pointer
);

// Read and check
void copy_configurations(global_const_t h_const, global_const_t *d_const_ptr);
void allocate_variables(
	double ****&U, double ****&Unew, double ****&Q, double ****&D, double ****&F,
	double *&d_U, double *&d_Unew, double *&d_Q, double *&d_D, double *&d_F,
	bool gpu=true, bool cpu=false
);
void free_variables(
	double ****U, double ****Unew, double ****Q, double ****D, double ****F,
	double *d_U, double *d_Unew, double *d_Q, double *d_D, double *d_F,
	bool gpu=true, bool cpu=false
);

#endif
