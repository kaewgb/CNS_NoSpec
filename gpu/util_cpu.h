#ifndef UTIL_CPU_INCLUDED
#define UTIL_CPU_INCLUDED

#include <math.h>
#define FEQ(x, y)	((fabs(x-y)<0.0001)? true:false)

// Allocations
void allocate_4D(double ****&ptr, int dim[], int dl);
void allocate_3D(double ***&ptr, int dim[]);
void free_4D(double ****ptr, int dim[], int dl);
void free_3D(double ***ptr, int dim[]);

// I/O and check
void read_configurations(global_const_t &h_const, int argc, char *argv[]);
void read_3D(FILE *f, double ****ptr, int dim[], int l);
void print_4D(FILE *f, double ****ptr, int dim[], int dl);
void print_3D(FILE *f, double ***ptr, int dim[]);
void check_double(double a, double b, const char *name);
void check_4D_array( const char *name, double ****a, double ****a2, int dim[],  int la);

// Data manipulations
void fill_boundary(
	global_const_t h,	// Application parameters
	double ****U		// Array
);
void init_data(global_const_t h, double ****cons);
void set_3D(double val, double ***ptr, int dim[]);
void number_3D(double ***ptr, int dim[]);

// Time
double get_time();

#endif
