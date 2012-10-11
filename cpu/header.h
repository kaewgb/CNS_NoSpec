#ifndef HEADER_INCLUDED
#define HEADER_INCLUDED

// CONSTRUCTS
#define DO(x, y, z)		for(x=y; x<=(int)(z); x++)
#define FOR(x, y, z)	for(x=y; x<(int)(z); x++)

// MACROS
#define SQR(x)          ((x)*(x))
#define MAX(x, y)       ((x > y)? (x):(y))

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
	qu=1,
	qv,
	qw,
	qpres,
	qfive
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
	int nc;
	int ncells;
	int lo[3];
	int hi[3];
	int dim[3];
	int dim_g[3];
	int comp_offset_g;
	int comp_offset;
	int plane_offset_g;
	int plane_offset;
	int nsteps;

	double dt;
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
	double OneQuarter;
	double ThreeQuarters;

	double CENTER;
	double OFF1;
	double OFF2;
	double OFF3;
	double OFF4;

} global_const_t;

static const double ALP	=  0.8E0;
static const double BET	= -0.2E0;
static const double GAM	=  4.0E0/105.0E0;
static const double DEL	=  -1.0E0/280.0E0;
//static const double ALP	=  0.80000000000000004; // 0.8E0;
//static const double BET	= -0.20000000000000001;	//-0.2E0;
//static const double GAM	=  3.80952380952380987E-002;	// 4.0E0/105.0E0;
//static const double DEL	= -3.57142857142857132E-003;	// -1.0E0/280.0E0;

// FUNCTIONS
extern void ctoprim_test();
extern void ctoprim (
    int lo[],       // i: lo[3]
    int hi[],       // i: hi[3]
    double ****u,   // i: u[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][5]
    double ****q, 	// o: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
    double dx[],    // i: dx[3]
    int ng,         // i
    double &courno  // i/o
);
extern void ctoprim (
    int lo[],       // i: lo[3]
    int hi[],       // i: hi[3]
    double ****u,   // i: u[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][5]
    double ****q, 	// o: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
    double dx[],    // i: dx[3]
    int ng         // i
);

extern void hypterm_test();
extern void hypterm(
	int lo[],			//i: lo[3]
	int hi[],			//i: hi[3]
	int ng,				//i
	double dx[],		//i: dx[3]
	double ****cons,	//i: cons[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[0]-lo[0]+2*ng][5];
	double ****q,		//i: cons[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[0]-lo[0]+2*ng][6];
	double ****flux		//o: flux[hi[0]-lo[0]][hi[1]-lo[1]][hi[2]-lo[2]][5]
);

extern void diffterm_test();
extern void diffterm (
	int lo[],			// i: lo[3]
	int hi[],			// i: hi[3]
	int ng,				// i
	double dx[],		// i: dx[3]
	double ****q,		// i: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
	double ****difflux,	// i/o: difflux[hi[0]-lo[0]][hi[1]-lo[1]][hi[2]-lo[2]][5]
	double eta,			// i
	double alam			// i
);

extern void advance_test();
extern void advance(
	double ****U,	// i/o
	double &dt,		// o
	double dx[],	// i: dx[U.dim]
	double cfl,		// i
	double eta,		// i
	double alam		// i
);
extern void new_advance(
	double ****U,		// i/o
	double ****Unew,
	double ****Q,
	double ****D,
	double ****F,
	double &dt			// o
);
#endif
