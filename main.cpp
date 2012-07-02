#include <stdio.h>
#include "header.h"

//typedef struct multifab_t{
//	int a;
//}multifab;
//typedef struct layout_t{
//	int a;
//}layout;
//
//int ncomp(multifab U);
//int nghost(multifab U);
//layout get_layout(multifab U);
//void multifab_fill_boundary(multifab U);
//void multifab_build(multifab U, layout la, int nc, int ng);
//
//void advance(multifab U,
//			 double dt, double dx,
//			 double cfl, double eta, double alam){
//
//	int lo[3], hi[3], i, j, k, m, n, nc, ng;
//	double courno, courno_proc;
//	layout la;
//	multifab D, F, Unew, Q;
//	double ****up, ****dp, ****fp, ****unp, ****qp;
//
//	//
//	// Some arithmetic constants.
//	//
//	const double OneThird 		= 1.0/3.0;
//	const double TwoThirds 		= 2.0/3.0;
//	const double OneQuarter 	= 1.0/4.0;
//	const double ThreeQuarters	= 3.0/4.0;
//
//	nc = ncomp(U);
//	ng = nghost(U);
//	la = get_layout(U);
//
//	//
//	// Sync U prior to calculating D & F.
//	//
//	multifab_fill_boundary(U);
//	multifab_build(D, 	 la, nc, 	0);
//	multifab_build(F,	 la, nc, 	0);
//	multifab_build(Q,	 la, nc+1, 	ng);
//	multifab_build(Unew, la, nc,	ng);
//
//	//
//	// Calculate primitive variables based on U.
//	// Also calculate courno so we can set "dt".
//	//
//	courno_proc = 1.0d-50;
//	DO(n, nboxes(Q)){
//
//	}
//
//}

int main(int argc, char *argv[]){
//	advance();
	return 0;
}
