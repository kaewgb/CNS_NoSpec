#include <math.h>
#define FEQ(x, y)	((fabs(x-y)<0.01)? true:false)

// Allocations
void gpu_allocate_3D(double *&d_ptr, int dim[]);
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

void allocate_4D(double ****&ptr, int dim[], int dl);
void allocate_3D(double ***&ptr, int dim[]);
void free_4D(double ****ptr, int dim[], int dl);
void free_3D(double ***ptr, int dim[]);

// Read and check
void read_3D(FILE *f, double ****ptr, int dim[], int l);
void check_double(double a, double b, const char *name);
void check_lo_hi_ng_dx( int lo[],  int hi[],  int ng,  double dx[],
									  int lo2[], int hi2[], int ng2, double dx2[] );
void check_4D_array( const char *name, double ****a, double ****a2, int dim[],  int la);

void fill_boundary(
	double ****U,	// Array
	int dim[],		// Dimensions (ghost cells excluded)
	int dim_ng[]	// Dimensions (ghost cells included)
);

