#ifndef HEADER_INCLUDED
#define HEADER_INCLUDED

// CONSTRUCTS
#define DO(x, y, z)		for(x=y; x<=(int)(z); x++)
#define FOR(x, y, z)	for(x=y; x<(int)(z); x++)

// MACROS
#define SQR(x)          ((x)*(x))
#define MIN(x, y)       ((x < y)? (x):(y))
#define MAX(x, y)       ((x > y)? (x):(y))
#define CEIL(x, div)	(((x) + (div)-1)/(div))

// CONSTANTS
#define	DIM		3
#define	NC		5
#define	NG		4
#define	NBOXES	1

enum {
	irho=0,
	imx,
	imy,
	imz,
	iene
};
enum {
	s_imx=0,
	s_imy,
	s_imz,
	s_iene
};
enum {
	qu=1,
	qv,
	qw,
	qpres,
	qt
};
enum {
	s_qu=0,
	s_qv,
	s_qw,
	s_qt,
	s_qend
};

enum diffterm_enum {
	UX, 	VX, 	WX,
	UY, 	VY, 	WY,
	UZ, 	VZ, 	WZ,
	UXX,	UYY,	UZZ,	VYX,	WZX,
	VXX,	VYY, 	VZZ, 	UXY,	WZY,
	WXX,	WYY,	WZZ, 	UXZ, 	VYZ,
	TXX,	TYY,	TZZ,
	MAX_TEMP
};

typedef struct kernel_const {
	int gridDim_x;
	int gridDim_y;
	int gridDim_z;
	int gridDim_plane_xy;
	int gridDim_plane_xz;
	int gridDim_plane_yz;
	int blockDim_x_g;
	int blockDim_y_g;
	int blockDim_z_g;

}kernel_const_t;

typedef struct global_const {

	char *input_file_name;
	char *output_file_name;

	int ng;						// Number of ghost cells
	int nc;						// Number of components
	int ncells;
	int lo[3];
	int hi[3];
	int dim[3];					// Dimension
	int dim_g[3];				// Dimension, including ghost cells
	int comp_offset_g;
	int comp_offset;
	int plane_offset_g;
	int plane_offset;
	int nsteps;					// Number of simulation iterations

	double dt;
	double dx[3];
	double dxinv[3];
	double cfl;
	double eta;
	double alam;
	double *temp[MAX_TEMP];	// Temporary arrays for diffterm
	double ***cpu_temp[WZ+1];

	double ALP;
	double BET;
	double GAM;

	double DEL;
	double OneThird;
	double TwoThirds;
	double FourThirds;
	double OneQuarter;
	double ThreeQuarters;

	double CENTER;
	double OFF1;
	double OFF2;
	double OFF3;
	double OFF4;

	kernel_const_t *kc;

} global_const_t;

//#define ALP		( 0.8E0)
//#define BET	 	(-0.2E0)
//#define GAM	 	( 4.0E0/105.0E0)
//#define DEL	 	(-1.0E0/280.0E0)

// FUNCTIONS
extern void ctoprim_test(
	global_const_t h_const, // i: Global struct containing applicatino parameters
	global_const_t *d_const	// i: Device pointer to global struct containing application paramters
);
extern void ctoprim (
	global_const_t c,
    double ****u,   // i: u[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][5]
    double ****q, 	// o: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
    double &courno  // i/o
);
extern void ctoprim (
	global_const_t c,
    double ****u,   // i: u[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][5]
    double ****q 	// o: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
);
extern void gpu_ctoprim(
	global_const_t h_const,		// i: Global struct containing application parameters
    global_const_t *d_const,	// i: Device pointer to global struct containing application parameters
    double *u_d,   				// i: u[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][5]
    double *q_d, 				// o: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
    double &courno  			// i/o
);
extern void gpu_ctoprim(
	global_const_t h_const,		// i: Global struct containing application parameters
    global_const_t *d_const,	// i: Device pointer to global struct containing application parameters
    double *u_d,   				// i: u[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][5]
    double *q_d 				// o: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
);


extern void hypterm_test(
	global_const_t h_const, // i: Global struct containing applicatino parameters
	global_const_t *d_const	// i: Device pointer to global struct containing application paramters
);
extern void hypterm(
	global_const_t h,
	double ****cons,	//i: cons[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[0]-lo[0]+2*ng][5];
	double ****q,		//i: cons[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[0]-lo[0]+2*ng][6];
	double ****flux		//o: flux[hi[0]-lo[0]][hi[1]-lo[1]][hi[2]-lo[2]][5]
);
extern void gpu_hypterm_merged(
	global_const_t h_const, 	// i: Global struct containing applicatino parameters
	global_const_t *d_const,	// i: Device pointer to global struct containing application paramters
	double *d_cons,				// i:
	double *d_q,				// i:
	double *d_flux				// o: flux
);
extern void gpu_hypterm(
	global_const_t h_const, 	// i: Global struct containing application parameters
	global_const_t *d_const,	// i: Device pointer to global struct containing application paramters
	double *d_cons,				// i:
	double *d_q,				// i:
	double *d_flux				// o: flux
);


extern void diffterm(
	global_const_t c,		// i: global application parameters
	double ****q,			// i: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
	double ****difflux		// i/o: difflux[hi[0]-lo[0]][hi[1]-lo[1]][hi[2]-lo[2]][5]
);
extern void gpu_diffterm(
	global_const_t h_const, 	// i: Global struct containing application parameters
	global_const_t *d_const,	// i: Device pointer to global struct containing application paramters
	double *d_q,				// i:
	double *flux				// o:
);


extern void advance_test();
extern void advance_cpu_test(
	global_const_t h_const,
	double ****U,
	double ****Unew,
	double ****Q,
	double ****D,
	double ****F
);
void advance_cpu_multistep_test(
	global_const_t h_const,
	double ****U,
	double ****Unew,
	double ****Q,
	double ****D,
	double ****F
);
extern void advance_test(
	global_const_t h_const, 	// i: Global struct containing application parameters
	global_const_t *d_const,	// i: Device pointer to global struct containing application paramters
	double ****U,
	double ****Unew,
	double ****Q,
	double ****D,
	double ****F,
	double *d_U,
	double *d_Unew,
	double *d_Q,
	double *d_D,
	double *d_F
);
void advance_multistep_test(
	global_const_t h_const, 	// i: Global struct containing application parameters
	global_const_t *d_const,	// i: Device pointer to global struct containing application paramters
	double ****U,
	double ****Unew,
	double ****Q,
	double ****D,
	double ****F,
	double *d_U,
	double *d_Unew,
	double *d_Q,
	double *d_D,
	double *d_F
);

extern void advance(
	global_const_t h_const,
	double ****U,		// i/o
	double ****Unew,	// o
	double ****Q,		// o
	double ****D,		// o
	double ****F,		// o
	double &dt			// o
);
extern void gpu_advance(
	global_const_t &h_const,	// i: Global constants
	global_const_t *d_const,	// i: Device pointer to global constants
	double *d_U,				// i/o
	double *d_Unew,
	double *d_Q,
	double *d_D,
	double *d_F,
	double &dt					// o
);
extern void new_advance_hybrid(
	global_const_t h_const,
	global_const_t *d_const,
	double ****U,	// i/o
	double ****Unew,
	double ****Q,
	double ****D,
	double ****F,
	double *d_U,	// i/o
	double *d_Unew,	// i/o
	double *d_Q,	// i/o
	double *d_D,	// i/o
	double *d_F,	// i/o
	double &dt		// o
);

extern void fill_boundary_test(
	global_const_t h_const, // i: Global struct containing application parameters
	global_const_t *d_const	// i: Device pointer to global struct containing application paramters
);


extern void gpu_Unew(
	global_const_t h_const,	// i: Global Constants
	global_const_t *d_const,	// i: Device Pointer to Global Constants
	double *d_Unew,		 		// o: New U
	double *d_U,				// i: Old U
	double *d_D,				// i: difflux
	double *d_F,				// i: flux
	double dt,					// i: dt
	int phase					// i: phase
);
#endif
