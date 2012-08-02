#ifndef HEADER_INCLUDED
#define HEADER_INCLUDED

// CONSTRUCTS
#define DO(x, y, z)		for(x=y; x<=(int)(z); x++)
#define FOR(x, y, z)	for(x=y; x<(int)(z); x++)

// MACROS
#define SQR(x)          ((x)*(x))
#define MAX(x, y)       ((x > y)? (x):(y))
#define CEIL(x, div)	(((x) + (div)-1)/(div))

// CONSTANTS
#define	DIM		3
#define	NC		5
#define	NG		4
#define	NCELLS	16
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

typedef struct global_const {
	int ng;
	int lo[3];
	int hi[3];
	int dim[3];
	int dim_g[3];
	int comp_offset_g;
	int comp_offset;
	int plane_offset_g;
	int plane_offset;
	int gridDim_x;
	int gridDim_y;
	int gridDim_z;
	int gridDim_plane_xy;
	int gridDim_plane_xz;
	int gridDim_plane_yz;
	int blockDim_x_g;
	int blockDim_y_g;
	int blockDim_z_g;
	double dx[3];
	double dxinv[3];
	double cfl;
	double eta;
	double alam;
	double *temp[MAX_TEMP];

	double ALP;
	double BET;
	double GAM;

	double DEL;
	double OneThird;
	double TwoThirds;
	double FourThirds;

	double CENTER;
	double OFF1;
	double OFF2;
	double OFF3;
	double OFF4;

} global_const_t;

static const double ALP	=  0.8E0;
static const double BET	= -0.2E0;
static const double GAM	=  4.0E0/105.0E0;
static const double DEL	= -1.0E0/280.0E0;

// FUNCTIONS
extern void ctoprim_test(
	global_const_t h_const, // i: Global struct containing applicatino parameters
	global_const_t *d_const	// i: Device pointer to global struct containing application paramters
);
extern void ctoprim (
    int lo[],       // i: lo[3]
    int hi[],       // i: hi[3]
    double ****u,   // i: u[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][5]
    double ****q, 	// o: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
    double dx[],    // i: dx[3]
    int ng,         // i
    double &courno   // i/o
);

extern void hypterm_test(
	global_const_t h_const, // i: Global struct containing applicatino parameters
	global_const_t *d_const	// i: Device pointer to global struct containing application paramters
);
extern void hypterm(
	int lo[],			//i: lo[3]
	int hi[],			//i: hi[3]
	int ng,				//i
	double dx[],		//i: dx[3]
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

extern void diffterm_test(
	global_const_t h_const, // i: Global struct containing applicatino parameters
	global_const_t *d_const	// i: Device pointer to global struct containing application paramters
);
extern void diffterm (
	int lo[],			// i: lo[3]
	int hi[],			// i: hi[3]
	int ng,				// i
	double dx[],		// i: dx[3]
	double ****q,		// i: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
	double ****difflux,	// i/o: difflux[hi[0]-lo[0]][hi[1]-lo[1]][hi[2]-lo[2]][5]
	double eta,			// i
	double alam,		// i
	double ***ux, double ***vx, double ***wx,
	double ***uy, double ***vy, double ***wy,
	double ***uz, double ***vz, double ***wz,
	double ***vyx_, double ***wzx_,
	double ***uxy_, double ***wzy_,
	double ***uxz_, double ***vyz_
);

extern void advance_test();
extern void advance(
	double ****U[],	// i/o
	double &dt,		// o
	double dx[],	// i: dx[U.dim]
	double cfl,		// i
	double eta,		// i
	double alam		// i
);
#endif
