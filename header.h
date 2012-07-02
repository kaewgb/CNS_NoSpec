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
	qpres
};

static const double ALP	=  0.8E0;
static const double BET	= -0.2E0;
static const double GAM	=  4.0E0/105.0E0;
static const double DEL	= -1.0E0/280.0E0;


#endif
