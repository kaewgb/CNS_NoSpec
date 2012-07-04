#ifndef HEADER_INCLUDED
#define HEADER_INCLUDED

// LIBRARIES
#include <stdio.h>

// CONSTRUCTS
#define DO(x, y, z)		for(x=y; x<=(int)(z); x++)
#define FOR(x, y, z)	for(x=y; x<(int)(z); x++)

// MACROS
#define SQR(x)          ((x)*(x))
#define MAX(x, y)       ((x > y)? (x):(y))

// CONSTANTS
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

static const double ALP	=  0.8E0;
static const double BET	= -0.2E0;
static const double GAM	=  4.0E0/105.0E0;
static const double DEL	= -1.0E0/280.0E0;

// FUNCTIONS
extern void ctoprim (
    int lo[],       // i: lo[3]
    int hi[],       // i: hi[3]
    double ****u,   // i: u[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][5]
    double ****q, 	// o: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
    double dx[],    // i: dx[3]
    int ng,         // i
    double &courno   // i/o
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
#endif