#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "helper_functions.h"

#define	q(i,j,k,l)		q[i][j][k][l]
#define cons(i,j,k,l)	cons[i][j][k][l]
#define flux(i,j,k,l)	flux[i-ng][j-ng][k-ng][l]
#define dxinv(i)		dxinv[i-1]

void hypterm(
	int lo[],			//i: lo[3]
	int hi[],			//i: hi[3]
	int ng,				//i
	double dx[],		//i: dx[3]
	double ****cons,	//i: cons[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[0]-lo[0]+2*ng][5];
	double ****q,		//i: cons[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[0]-lo[0]+2*ng][6];
	double ****flux		//o: flux[hi[0]-lo[0]][hi[1]-lo[1]][hi[2]-lo[2]][5]
){

	int i, j, k;
	double unp1,unp2,unp3,unp4,unm1,unm2,unm3,unm4;
	double dxinv[3];

	FOR(i, 0, 3)
		dxinv[i] = 1.0E0/dx[i];

//	#pragma omp parallel for private(i,j,k,unp1,unp2,unp3,unp4,unm1,unm2,unm3,unm4)
	DO(i, lo[0], hi[0]){
		DO(j, lo[1], hi[1]){
			DO(k, lo[2], hi[2]){

				unp1 = q(i+1,j,k,qu);
				unp2 = q(i+2,j,k,qu);
				unp3 = q(i+3,j,k,qu);
				unp4 = q(i+4,j,k,qu);

				unm1 = q(i-1,j,k,qu);
				unm2 = q(i-2,j,k,qu);
				unm3 = q(i-3,j,k,qu);
				unm4 = q(i-4,j,k,qu);

				flux(i,j,k,irho)= -
					   (ALP*(cons(i+1,j,k,imx)-cons(i-1,j,k,imx))
					  + BET*(cons(i+2,j,k,imx)-cons(i-2,j,k,imx))
					  + GAM*(cons(i+3,j,k,imx)-cons(i-3,j,k,imx))
					  + DEL*(cons(i+4,j,k,imx)-cons(i-4,j,k,imx)))*dxinv(1);

				flux(i,j,k,imx)= -
					   (ALP*(cons(i+1,j,k,imx)*unp1-cons(i-1,j,k,imx)*unm1
					  + (q(i+1,j,k,qpres)-q(i-1,j,k,qpres)))
					  + BET*(cons(i+2,j,k,imx)*unp2-cons(i-2,j,k,imx)*unm2
					  + (q(i+2,j,k,qpres)-q(i-2,j,k,qpres)))
					  + GAM*(cons(i+3,j,k,imx)*unp3-cons(i-3,j,k,imx)*unm3
					  + (q(i+3,j,k,qpres)-q(i-3,j,k,qpres)))
					  + DEL*(cons(i+4,j,k,imx)*unp4-cons(i-4,j,k,imx)*unm4
					  + (q(i+4,j,k,qpres)-q(i-4,j,k,qpres))))*dxinv(1);

				flux(i,j,k,imy)= -
					   (ALP*(cons(i+1,j,k,imy)*unp1-cons(i-1,j,k,imy)*unm1)
					  + BET*(cons(i+2,j,k,imy)*unp2-cons(i-2,j,k,imy)*unm2)
					  + GAM*(cons(i+3,j,k,imy)*unp3-cons(i-3,j,k,imy)*unm3)
					  + DEL*(cons(i+4,j,k,imy)*unp4-cons(i-4,j,k,imy)*unm4))*dxinv(1);

				flux(i,j,k,imz)= -
					   (ALP*(cons(i+1,j,k,imz)*unp1-cons(i-1,j,k,imz)*unm1)
					  + BET*(cons(i+2,j,k,imz)*unp2-cons(i-2,j,k,imz)*unm2)
					  + GAM*(cons(i+3,j,k,imz)*unp3-cons(i-3,j,k,imz)*unm3)
					  + DEL*(cons(i+4,j,k,imz)*unp4-cons(i-4,j,k,imz)*unm4))*dxinv(1);

				flux(i,j,k,iene)= -
					   (ALP*(cons(i+1,j,k,iene)*unp1-cons(i-1,j,k,iene)*unm1
					  + (q(i+1,j,k,qpres)*unp1-q(i-1,j,k,qpres)*unm1))
					  + BET*(cons(i+2,j,k,iene)*unp2-cons(i-2,j,k,iene)*unm2
					  + (q(i+2,j,k,qpres)*unp2-q(i-2,j,k,qpres)*unm2))
					  + GAM*(cons(i+3,j,k,iene)*unp3-cons(i-3,j,k,iene)*unm3
					  + (q(i+3,j,k,qpres)*unp3-q(i-3,j,k,qpres)*unm3))
					  + DEL*(cons(i+4,j,k,iene)*unp4-cons(i-4,j,k,iene)*unm4
					  + (q(i+4,j,k,qpres)*unp4-q(i-4,j,k,qpres)*unm4)))*dxinv(1);

			}
		}
	}

//	#pragma omp parallel for private(i,j,k,unp1,unp2,unp3,unp4,unm1,unm2,unm3,unm4)
	DO(i, lo[0], hi[0]){
		DO(j, lo[1], hi[1]){
			DO(k, lo[2], hi[2]){

				unp1 = q(i,j+1,k,qv);
				unp2 = q(i,j+2,k,qv);
				unp3 = q(i,j+3,k,qv);
				unp4 = q(i,j+4,k,qv);

				unm1 = q(i,j-1,k,qv);
				unm2 = q(i,j-2,k,qv);
				unm3 = q(i,j-3,k,qv);
				unm4 = q(i,j-4,k,qv);

				flux(i,j,k,irho)=flux(i,j,k,irho) -
					   (ALP*(cons(i,j+1,k,imy)-cons(i,j-1,k,imy))
					  + BET*(cons(i,j+2,k,imy)-cons(i,j-2,k,imy))
					  + GAM*(cons(i,j+3,k,imy)-cons(i,j-3,k,imy))
					  + DEL*(cons(i,j+4,k,imy)-cons(i,j-4,k,imy)))*dxinv(2);

				flux(i,j,k,imx)=flux(i,j,k,imx) -
					   (ALP*(cons(i,j+1,k,imx)*unp1-cons(i,j-1,k,imx)*unm1)
					  + BET*(cons(i,j+2,k,imx)*unp2-cons(i,j-2,k,imx)*unm2)
					  + GAM*(cons(i,j+3,k,imx)*unp3-cons(i,j-3,k,imx)*unm3)
					  + DEL*(cons(i,j+4,k,imx)*unp4-cons(i,j-4,k,imx)*unm4))*dxinv(2);

				flux(i,j,k,imy)=flux(i,j,k,imy) -
					   (ALP*(cons(i,j+1,k,imy)*unp1-cons(i,j-1,k,imy)*unm1
					  + (q(i,j+1,k,qpres)-q(i,j-1,k,qpres)))
					  + BET*(cons(i,j+2,k,imy)*unp2-cons(i,j-2,k,imy)*unm2
					  + (q(i,j+2,k,qpres)-q(i,j-2,k,qpres)))
					  + GAM*(cons(i,j+3,k,imy)*unp3-cons(i,j-3,k,imy)*unm3
					  + (q(i,j+3,k,qpres)-q(i,j-3,k,qpres)))
					  + DEL*(cons(i,j+4,k,imy)*unp4-cons(i,j-4,k,imy)*unm4
					  + (q(i,j+4,k,qpres)-q(i,j-4,k,qpres))))*dxinv(2);

				flux(i,j,k,imz)=flux(i,j,k,imz) -
					   (ALP*(cons(i,j+1,k,imz)*unp1-cons(i,j-1,k,imz)*unm1)
					  + BET*(cons(i,j+2,k,imz)*unp2-cons(i,j-2,k,imz)*unm2)
					  + GAM*(cons(i,j+3,k,imz)*unp3-cons(i,j-3,k,imz)*unm3)
					  + DEL*(cons(i,j+4,k,imz)*unp4-cons(i,j-4,k,imz)*unm4))*dxinv(2);

				flux(i,j,k,iene)=flux(i,j,k,iene) -
					   (ALP*(cons(i,j+1,k,iene)*unp1-cons(i,j-1,k,iene)*unm1
					  + (q(i,j+1,k,qpres)*unp1-q(i,j-1,k,qpres)*unm1))
					  + BET*(cons(i,j+2,k,iene)*unp2-cons(i,j-2,k,iene)*unm2
					  + (q(i,j+2,k,qpres)*unp2-q(i,j-2,k,qpres)*unm2))
					  + GAM*(cons(i,j+3,k,iene)*unp3-cons(i,j-3,k,iene)*unm3
					  + (q(i,j+3,k,qpres)*unp3-q(i,j-3,k,qpres)*unm3))
					  + DEL*(cons(i,j+4,k,iene)*unp4-cons(i,j-4,k,iene)*unm4
					  + (q(i,j+4,k,qpres)*unp4-q(i,j-4,k,qpres)*unm4)))*dxinv(2);

			}
		}
	}

//	#pragma omp parallel for private(i,j,k,unp1,unp2,ump3,unp4,unm1,unm2,unm3,unm4)
	DO(i, lo[0], hi[0]){
		DO(j, lo[1], hi[1]){
			DO(k, lo[2], hi[2]){

				unp1 = q(i,j,k+1,qw);
				unp2 = q(i,j,k+2,qw);
				unp3 = q(i,j,k+3,qw);
				unp4 = q(i,j,k+4,qw);

				unm1 = q(i,j,k-1,qw);
				unm2 = q(i,j,k-2,qw);
				unm3 = q(i,j,k-3,qw);
				unm4 = q(i,j,k-4,qw);

				flux(i,j,k,irho)=flux(i,j,k,irho) -
					   (ALP*(cons(i,j,k+1,imz)-cons(i,j,k-1,imz))
					  + BET*(cons(i,j,k+2,imz)-cons(i,j,k-2,imz))
					  + GAM*(cons(i,j,k+3,imz)-cons(i,j,k-3,imz))
					  + DEL*(cons(i,j,k+4,imz)-cons(i,j,k-4,imz)))*dxinv(3);

				flux(i,j,k,imx)=flux(i,j,k,imx) -
					   (ALP*(cons(i,j,k+1,imx)*unp1-cons(i,j,k-1,imx)*unm1)
					  + BET*(cons(i,j,k+2,imx)*unp2-cons(i,j,k-2,imx)*unm2)
					  + GAM*(cons(i,j,k+3,imx)*unp3-cons(i,j,k-3,imx)*unm3)
					  + DEL*(cons(i,j,k+4,imx)*unp4-cons(i,j,k-4,imx)*unm4))*dxinv(3);

				flux(i,j,k,imy)=flux(i,j,k,imy) -
					   (ALP*(cons(i,j,k+1,imy)*unp1-cons(i,j,k-1,imy)*unm1)
					  + BET*(cons(i,j,k+2,imy)*unp2-cons(i,j,k-2,imy)*unm2)
					  + GAM*(cons(i,j,k+3,imy)*unp3-cons(i,j,k-3,imy)*unm3)
					  + DEL*(cons(i,j,k+4,imy)*unp4-cons(i,j,k-4,imy)*unm4))*dxinv(3);

				flux(i,j,k,imz)=flux(i,j,k,imz) -
					   (ALP*(cons(i,j,k+1,imz)*unp1-cons(i,j,k-1,imz)*unm1
					  + (q(i,j,k+1,qpres)-q(i,j,k-1,qpres)))
					  + BET*(cons(i,j,k+2,imz)*unp2-cons(i,j,k-2,imz)*unm2
					  + (q(i,j,k+2,qpres)-q(i,j,k-2,qpres)))
					  + GAM*(cons(i,j,k+3,imz)*unp3-cons(i,j,k-3,imz)*unm3
					  + (q(i,j,k+3,qpres)-q(i,j,k-3,qpres)))
					  + DEL*(cons(i,j,k+4,imz)*unp4-cons(i,j,k-4,imz)*unm4
					  + (q(i,j,k+4,qpres)-q(i,j,k-4,qpres))))*dxinv(3);

				flux(i,j,k,iene)=flux(i,j,k,iene) -
					   (ALP*(cons(i,j,k+1,iene)*unp1-cons(i,j,k-1,iene)*unm1
					  + (q(i,j,k+1,qpres)*unp1-q(i,j,k-1,qpres)*unm1))
					  + BET*(cons(i,j,k+2,iene)*unp2-cons(i,j,k-2,iene)*unm2
					  + (q(i,j,k+2,qpres)*unp2-q(i,j,k-2,qpres)*unm2))
					  + GAM*(cons(i,j,k+3,iene)*unp3-cons(i,j,k-3,iene)*unm3
					  + (q(i,j,k+3,qpres)*unp3-q(i,j,k-3,qpres)*unm3))
					  + DEL*(cons(i,j,k+4,iene)*unp4-cons(i,j,k-4,iene)*unm4
					  + (q(i,j,k+4,qpres)*unp4-q(i,j,k-4,qpres)*unm4)))*dxinv(3);

			}
		}
	}
}

void hypterm_test(){

	int lo[3], hi[3], ng;
	double dx[3];
	double ****cons, ****q, ****flux;

	int lo2[3], hi2[3], ng2;
	double dx2[3];
	double ****cons2, ****q2, ****flux2;

	int i, l;
	int dim[3], dim2[3];

	FILE *fin = fopen("../testcases/hypterm_input", "r");
	FILE *fout = fopen("../testcases/hypterm_output", "r");
	if(fin == NULL || fout == NULL){
		printf("Invalid input!\n");
		exit(1);
	}

	// Scanning input
	fscanf(fin, "%d %d %d\n", &lo[0], &lo[1], &lo[2]);
	fscanf(fin, "%d %d %d\n", &hi[0], &hi[1], &hi[2]);
	fscanf(fin, "%d\n", &ng);
	fscanf(fin, "%le %le %le\n", &dx[0], &dx[1], &dx[2]);

	lo[0] += ng; 	lo[1] += ng; 	lo[2] += ng;
	hi[0] += ng; 	hi[1] += ng; 	hi[2] += ng;

	FOR(i, 0, 3){
		dim2[i] = hi[i]-lo[i]+1;
		dim[i]  = dim2[i] + 2*ng;
	}

	allocate_4D(cons, 	dim, 	5);		// [40][40][40][5]
	allocate_4D(q, 		dim, 	6);		// [40][40][40][6]
	allocate_4D(flux, 	dim2, 	5);		// [32][32][32][5]
	allocate_4D(cons2, 	dim, 	5);		// [40][40][40][5]
	allocate_4D(q2, 	dim, 	6);	 	// [40][40][40][6]
	allocate_4D(flux2, 	dim2, 	5);		// [40][40][40][5]

	FOR(l, 0, 5)
		read_3D(fin, cons,  dim,  l);
	FOR(l, 0, 6)
		read_3D(fin, q,		dim,  l);
	FOR(l, 0, 5)
		read_3D(fin, flux,  dim2, l);
	fclose(fin);

	printf("Applying hypterm()...\n");
	hypterm(lo, hi, ng, dx, cons, q, flux);

	// Scanning output to check
	fscanf(fout, "%d %d %d\n", &lo2[0], &lo2[1], &lo2[2]);
	fscanf(fout, "%d %d %d\n", &hi2[0], &hi2[1], &hi2[2]);
	fscanf(fout, "%d\n", &ng2);
	fscanf(fout, "%le %le %le\n", &dx2[0], &dx2[1], &dx2[2]);

	FOR(l, 0, 5)
		read_3D(fout, cons2, dim,  l);
	FOR(l, 0, 6)
		read_3D(fout, q2,	 dim,  l);
	FOR(l, 0, 5)
		read_3D(fout, flux2,  dim2, l);
	fclose(fout);

	// Checking...
	check_lo_hi_ng_dx(lo, hi, ng, dx, lo2, hi2, ng2, dx2);
	check_4D_array("cons", cons, cons2, dim,  5);
	check_4D_array("q",    q, 	 q2,	dim,  6);
	check_4D_array("flux", flux, flux2, dim2, 5);

	free_4D(cons, dim);		free_4D(q, dim);	free_4D(flux, dim2);
	free_4D(cons2, dim);	free_4D(q2, dim);	free_4D(flux2, dim2);

	printf("Correct!\n");
}
