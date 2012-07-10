#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "header.h"
#include "helper_functions.h"

#define u(i,j,k,l)  u[i][j][k][l-1]
#define q(i,j,k,l)  q[i][j][k][l-1]
#define dx(i)		dx[i-1]

__global__ void gpu_ctoprim_kernel(
    int lo[],       // i: lo[3]
    int hi[],       // i: hi[3]
    double *u_d,   // i: u[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][5]
    double *q_d, 	// o: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
    double dx[],    // i: dx[3]
    int ng,         // i
    double *courno   // i/o
){

}
void gpu_ctoprim(
    int lo[],       // i: lo[3]
    int hi[],       // i: hi[3]
    double *u_d,   	// i: u[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][5]
    double *q_d, 	// o: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
    double dx[],    // i: dx[3]
    int ng,         // i
    double &courno  // i/o
){
	int i, dim_ng[3];
	FOR(i, 0, 3)
		dim_ng[i] = hi[i]-lo[i]+1 + ng+ng;

}
void ctoprim (
    int lo[],       // i: lo[3]
    int hi[],       // i: hi[3]
    double ****u,   // i: u[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][5]
    double ****q, 	// o: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
    double dx[],    // i: dx[3]
    int ng,         // i
    double &courno   // i/o
){
    int i, j, k;
    double c, eint, courx, coury, courz, courmx, courmy, courmz, rhoinv;

    const double GAMMA  = 1.4E0;
    const double CV     = 8.3333333333E6;

//    #pragma omp parallel for private(i, j, k, eint, rhoinv)
    DO(i, lo[0]-ng, hi[0]+ng){
        DO(j, lo[1]-ng, hi[1]+ng){
            DO(k, lo[2]-ng, hi[2]+ng){

				rhoinv     = 1.0E0/u(i,j,k,1);
				q(i,j,k,1) = u(i,j,k,1);
				q(i,j,k,2) = u(i,j,k,2)*rhoinv;
				q(i,j,k,3) = u(i,j,k,3)*rhoinv;
				q(i,j,k,4) = u(i,j,k,4)*rhoinv;

				eint = u(i,j,k,5)*rhoinv - 0.5E0*(SQR(q(i,j,k,2)) + SQR(q(i,j,k,3)) + SQR(q(i,j,k,4)));

				q(i,j,k,5) = (GAMMA-1.0E0)*eint*u(i,j,k,1);
				q(i,j,k,6) = eint/CV;

            }
        }
    }

    if(courno != -1.0){	// Just my way to check of courno is present, i.e., is passed to the function
//        #pragma omp parallel for private(i, j, k, c, courx, coury, courz) reduction(max: courmx, courmy, courmz)
        DO(i, lo[0], hi[0]){
            DO(j, lo[1], hi[1]){
                DO(k, lo[2], hi[2]){

					c     = sqrt(GAMMA*q(i,j,k,5)/q(i,j,k,1));
					courx = ( c+fabs(q(i,j,k,2)) ) / dx(1);
					coury = ( c+fabs(q(i,j,k,3)) ) / dx(2);
					courz = ( c+fabs(q(i,j,k,4)) ) / dx(3);

					courmx = MAX( courmx, courx );
					courmy = MAX( courmy, coury );
					courmz = MAX( courmz, courz );

                }
            }
        }
    }

    //
    // Compute running max of Courant number over grids.
    //
    courno = MAX(MAX(courmx, courmy), MAX(courmz, courno));
}

void ctoprim_test(){

	int i, l, dummy, dim_ng[3];
	int lo[3], hi[3];
	int ng=4;
	double ****u, ****q;
	double dx[3], courno;

	int ng2;
	int lo2[3], hi2[3];
	double ****u2, ****q2;
	double *u_d, *q_d;
	double dx2[3], courno2;

	FILE *fin = fopen("../testcases/ctoprim_input", "r");
	FILE *fout = fopen("../testcases/ctoprim_output", "r");
	if(fin == NULL || fout == NULL){
		printf("Invalid input file\n");
		exit(1);
	}

	// Scanning input
	fscanf(fin, "%d %d %d\n", &lo[0], &lo[1], &lo[2]);
	fscanf(fin, "%d %d %d\n", &hi[0], &hi[1], &hi[2]);

	lo[0] += ng; 	lo[1] += ng; 	lo[2] += ng;
	hi[0] += ng; 	hi[1] += ng; 	hi[2] += ng;

	FOR(i, 0, 3)
		dim_ng[i] = hi[i]-lo[i]+1 + 2*ng;

	allocate_4D(u, 	dim_ng, 5); 	// [40][40][40][5]
	allocate_4D(q, 	dim_ng, 6); 	// [40][40][40][6]
	allocate_4D(u2, dim_ng, 5); 	// [40][40][40][5]
	allocate_4D(q2, dim_ng, 6); 	// [40][40][40][6]

	gpu_allocate_4D(u_d, dim_ng, 5);
	gpu_allocate_4D(q_d, dim_ng, 6);

	FOR(l, 0, 5)
		read_3D(fin, u, dim_ng, l);
	FOR(l, 0, 6)
		read_3D(fin, q, dim_ng, l);

	fscanf(fin, "%le %le %le\n", &dx[0], &dx[1], &dx[2]);
	fscanf(fin, "%d\n", &dummy);
	fscanf(fin, "%le\n", &courno);
	fclose(fin);

	gpu_copy_from_host_4D(u_d, u, dim_ng, 5);
	gpu_copy_from_host_4D(q_d, q, dim_ng, 6);

	printf("Applying ctoprim()...\n");
//	gpu_ctoprim(lo, hi, u_d, q_d, dx, ng, courno);
//	ctoprim(lo, hi, u, q, dx, ng, courno);

	gpu_copy_to_host_4D(u, u_d, dim_ng, 5);
	gpu_copy_to_host_4D(q, q_d, dim_ng, 6);

	// Scanning output to check
	fscanf(fout, "%d %d %d\n", &lo2[0], &lo2[1], &lo2[2]);
	fscanf(fout, "%d %d %d\n", &hi2[0], &hi2[1], &hi2[2]);
	FOR(l, 0, 5)
		read_3D(fout, u2, dim_ng, l);
	FOR(l, 0, 6)
		read_3D(fout, q2, dim_ng, l);

	fscanf(fout, "%le %le %le\n", &dx2[0], &dx2[1], &dx2[2]);
	fscanf(fout, "%d\n", &ng2);
	fscanf(fout, "%le\n", &courno2);
	fclose(fout);

	// Checking...
	check_lo_hi_ng_dx(lo, hi, ng, dx, lo2, hi2, ng2, dx2);
	check_double(courno, courno2, "courno");
	check_4D_array("u", u, u2, dim_ng, 5);
	check_4D_array("q", q, q2, dim_ng, 6);
	printf("Correct!\n");

	gpu_free_4D(u_d);
	gpu_free_4D(q_d);

	free_4D(u,  dim_ng);		free_4D(q,  dim_ng);
	free_4D(u2, dim_ng);		free_4D(q2, dim_ng);
}

