#include <stdio.h>
#include <math.h>
#define FEQ(x, y)	((fabs(x-y)<1.0E-9)? true:false)

// Allocations
void allocate_variables(double ****&U, double ****&Unew, double ****&Q, double ****&D, double ****&F);
void allocate_4D(double ****&ptr, int dim[], int dl);
void allocate_3D(double ***&ptr, int dim[]);
void free_variables(double ****U, double ****Unew, double ****Q, double ****D, double ****F);
void free_4D(double ****ptr, int dim[]);
void free_3D(double ***ptr, int dim[]);

// Utilities
void read_configurations(global_const_t &h_const);
void fill_boundary(
	double ****U,	// Array
	int dim[],		// Dimensions (ghost cells excluded)
	int dim_ng[]	// Dimensions (ghost cells included)
);
double get_time();

// Read and check
void read_3D(FILE *f, double ****ptr, int dim[], int l);
void print_4D(FILE *f, double ****ptr, int dim[], int l);
void check_double(double a, double b, const char *name);
void check_lo_hi_ng_dx( int lo[],  int hi[],  int ng,  double dx[],
									  int lo2[], int hi2[], int ng2, double dx2[] );
void check_4D_array( const char *name, double ****a, double ****a2, int dim[],  int la);


