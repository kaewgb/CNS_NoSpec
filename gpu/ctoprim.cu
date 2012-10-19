#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "thrust/reduce.h"
#include "thrust/device_ptr.h"
#include "header.h"
#include "helper_functions.h"

#define	BLOCK_DIM	512

__device__ double d_courno;
__constant__ double GAMMA  = 1.4E0;
__constant__ double CV     = 8.3333333333E6;

#undef 	SQR
#define SQR(x)          (__dmul_rn((x),(x)))
__global__ void gpu_ctoprim_kernel(
	global_const_t *g,	// i: Application parameters
    double *u,   		// i: u[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][5]
    double *q, 			// o: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
    double *courno  	// i/o
){

	int i, j, k, idx, loffset;
	int numthreads = BLOCK_DIM;
	double rhoinv, eint, c, courx, coury, courz;

	idx = blockIdx.x * blockDim.x + threadIdx.x;
	i = idx / (g->dim_g[2] * g->dim_g[1]);
	j = (idx / g->dim_g[2]) % g->dim_g[1];
	k = idx % g->dim_g[2];

	loffset = g->dim_g[0] * g->dim_g[1] * g->dim_g[2];

	// Calculate Q
	if( idx < loffset ){

		rhoinv 				= 1.0E0/u[idx];				//u(i,j,k,1) = u[0][i][j][k]
		q[idx] 				= u[idx]; 					//u(i,j,k,1) = u[0][i][j][k]
		q[idx+loffset] 		= u[idx+loffset]*rhoinv; 	//u(i,j,k,2) = u[1][i][j][k]
		q[idx+2*loffset] 	= u[idx+2*loffset]*rhoinv; 	//u(i,j,k,3) = u[2][i][j][k]
		q[idx+3*loffset] 	= u[idx+3*loffset]*rhoinv; 	//u(i,j,k,4) = u[3][i][j][k]

		eint = u[idx+4*loffset]*rhoinv - 0.5E0*(SQR(q[idx+loffset]) + SQR(q[idx+2*loffset]) + SQR(q[idx+3*loffset]));
//		eint = __dadd_rn(__dmul_rn(u[idx+4*loffset], rhoinv),
//						-__dmul_rn(0.5E0, __dadd_rn(__dadd_rn(SQR(q[idx+loffset]), SQR(q[idx+2*loffset])), SQR(q[idx+3*loffset]))));


		q[idx+4*loffset] = (GAMMA-1.0E0)*eint*u[idx];
		q[idx+5*loffset] = eint/CV;

		// Calculate new courno (excluding ng)
		if(	g->ng <= i && i <= g->hi[0]+g->ng &&
			g->ng <= j && j <= g->hi[1]+g->ng &&
			g->ng <= k && k <= g->hi[2]+g->ng ){

			c 		= sqrt(GAMMA*q[idx+4*loffset]/q[idx]);
			courx 	= (c+fabs(q[idx+loffset]))	/g->dx[0];
			coury	= (c+fabs(q[idx+2*loffset]))/g->dx[1];
			courz	= (c+fabs(q[idx+3*loffset]))/g->dx[2];

//			c		= sqrt(__dmul_rn(GAMMA, q[idx+4*loffset])/q[idx]);
//			courx	= __dadd_rn(c, fabs(q[idx+loffset]))/g->dx[0];
//			coury	= __dadd_rn(c, fabs(q[idx+2*loffset]))/g->dx[1];
//			courz	= __dadd_rn(c, fabs(q[idx+3*loffset]))/g->dx[2];

			courno[idx] = MAX(courx, MAX(coury, courz));
		}
		else
			courno[idx] = -1.0;		//TODO: make it minus infinity
	}
}
__global__ void gpu_ctoprim_kernel(
	global_const_t *g,	// i: Application parameters
    double *u,   		// i: u[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][5]
    double *q 			// o: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
){

	int i, j, k, idx, loffset;
	int numthreads = BLOCK_DIM;
	double rhoinv, eint, c, courx, coury, courz;

	idx = blockIdx.x * blockDim.x + threadIdx.x;
	i = idx / (g->dim_g[2] * g->dim_g[1]);
	j = (idx / g->dim_g[2]) % g->dim_g[1];
	k = idx % g->dim_g[2];

	loffset = g->dim_g[0] * g->dim_g[1] * g->dim_g[2];

	// Calculate Q
	if( idx < loffset ){

		rhoinv 				= 1.0E0/u[idx];				//u(i,j,k,1) = u[0][i][j][k]
		q[idx] 				= u[idx]; 					//u(i,j,k,1) = u[0][i][j][k]
		q[idx+loffset] 		= u[idx+loffset]*rhoinv; 	//u(i,j,k,2) = u[1][i][j][k]
		q[idx+2*loffset] 	= u[idx+2*loffset]*rhoinv; 	//u(i,j,k,3) = u[2][i][j][k]
		q[idx+3*loffset] 	= u[idx+3*loffset]*rhoinv; 	//u(i,j,k,4) = u[3][i][j][k]

		eint = u[idx+4*loffset]*rhoinv - 0.5E0*(SQR(q[idx+loffset]) + SQR(q[idx+2*loffset]) + SQR(q[idx+3*loffset]));
//		eint = __dadd_rn(__dmul_rn(u[idx+4*loffset], rhoinv),
//						-__dmul_rn(0.5E0, __dadd_rn(__dadd_rn(SQR(q[idx+loffset]), SQR(q[idx+2*loffset])), SQR(q[idx+3*loffset]))));


		q[idx+4*loffset] = (GAMMA-1.0E0)*eint*u[idx];
		q[idx+5*loffset] = eint/CV;
	}
}

void gpu_ctoprim(
	global_const_t h_const,		// i: Global struct containing application parameters
    global_const_t *d_const,	// i: Device pointer to global struct containing application parameters
    double *u_d,   				// i: u[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][5]
    double *q_d, 				// o: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
    double &courno  			// i/o
){
	int i, len;
	double *d_cour;

	len = h_const.dim_g[0] * h_const.dim_g[1] * h_const.dim_g[2];
	int grid_dim = (len + BLOCK_DIM-1) / BLOCK_DIM;
	int block_dim = BLOCK_DIM;

	// Allocate temporary memory to find maximum courno
	cudaMalloc((void **) &d_cour, len * sizeof(double));

	// TODO: edit parameters
	gpu_ctoprim_kernel<<<grid_dim, block_dim>>>(d_const, u_d, q_d, d_cour);

	// Find max & update courno
	// TODO: make it minus infinity
	thrust::device_ptr<double> dev_ptr(d_cour);
	courno = thrust::reduce(dev_ptr, dev_ptr + len, (double) -1.0, thrust::maximum<double>());

	// Free temporary memory
	cudaFree(d_cour);

}
void gpu_ctoprim(
	global_const_t h_const,		// i: Global struct containing application parameters
    global_const_t *d_const,	// i: Device pointer to global struct containing application parameters
    double *u_d,   				// i: u[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][5]
    double *q_d					// o: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
){
	int i, len;

	len = h_const.dim_g[0] * h_const.dim_g[1] * h_const.dim_g[2];
	int grid_dim = (len + BLOCK_DIM-1) / BLOCK_DIM;
	int block_dim = BLOCK_DIM;

	// TODO: edit parameters
	gpu_ctoprim_kernel<<<grid_dim, block_dim>>>(d_const, u_d, q_d);

}
#undef 	SQR
#define SQR(x)      ((x)*(x))
#define u(i,j,k,l)  u[l-1][i][j][k]
#define q(i,j,k,l)  q[l-1][i][j][k]
#define dx(i)		h.dx[i-1]
#define dxinv(i)	h.dxinv[i-1]
void ctoprim (
	global_const_t h,
    double ****u,   // i: u[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][5]
    double ****q, 	// o: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
    double &courno  // i/o
){
    int i, j, k;
    double c, eint, courx, coury, courz, courmx, courmy, courmz, rhoinv;

    const double GAMMA  = 1.4E0;
    const double CV     = 8.3333333333E6;

//    #pragma omp parallel for private(i, j, k, eint, rhoinv)
    DO(i, h.lo[0]-h.ng, h.hi[0]+h.ng){
        DO(j, h.lo[1]-h.ng, h.hi[1]+h.ng){
            DO(k, h.lo[2]-h.ng, h.hi[2]+h.ng){

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

//    #pragma omp parallel for private(i, j, k, c, courx, coury, courz) reduction(max: courmx, courmy, courmz)
	DO(i, h.lo[0], h.hi[0]){
		DO(j, h.lo[1], h.hi[1]){
			DO(k, h.lo[2], h.hi[2]){

				c     = sqrt(GAMMA*q(i,j,k,5)/q(i,j,k,1));

				courx = ( c+fabs(q(i,j,k,2)) ) / dx(1); // I tried to change to * dxinv(1) but the results diverge.. (max diff = 5E-8)
				coury = ( c+fabs(q(i,j,k,3)) ) / dx(2);
				courz = ( c+fabs(q(i,j,k,4)) ) / dx(3);

				courmx = MAX( courmx, courx );
				courmy = MAX( courmy, coury );
				courmz = MAX( courmz, courz );

			}
		}
	}

    //
    // Compute running max of Courant number over grids.
    //
    courno = MAX(MAX(courmx, courmy), MAX(courmz, courno));
}

void ctoprim (
	global_const_t h,
    double ****u,   // i: u[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][5]
    double ****q 	// o: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
){
    int i, j, k;
    double c, eint, courx, coury, courz, courmx, courmy, courmz, rhoinv;

    const double GAMMA  = 1.4E0;
    const double CV     = 8.3333333333E6;

//    #pragma omp parallel for private(i, j, k, eint, rhoinv)
    DO(i, h.lo[0]-h.ng, h.hi[0]+h.ng){
        DO(j, h.lo[1]-h.ng, h.hi[1]+h.ng){
            DO(k, h.lo[2]-h.ng, h.hi[2]+h.ng){

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
}
#undef u
#undef q
#undef dx

void ctoprim_test(
	global_const_t h_const, // i: Global struct containing application parameters
	global_const_t *d_const	// i: Device pointer to global struct containing application paramters
){

	int i, l, dummy, dim_g[3];
	int lo[3], hi[3];
	int ng=4;
	double ****u, ****q;
	double dx[3], courno;

	int ng2;
	int lo2[3], hi2[3];
	double ****u2, ****q2;
	double *d_u, *d_q;
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
		dim_g[i] = hi[i]-lo[i]+1 + 2*ng;
	printf("dim_g: %d %d %d\n", dim_g[0], dim_g[1], dim_g[2]);

	allocate_4D(u, 	dim_g, 5); 	// [40][40][40][5]
	allocate_4D(q, 	dim_g, 6); 	// [40][40][40][6]
	allocate_4D(u2, dim_g, 5); 	// [40][40][40][5]
	allocate_4D(q2, dim_g, 6); 	// [40][40][40][6]

	gpu_allocate_4D(d_u, dim_g, 5);
	gpu_allocate_4D(d_q, dim_g, 6);

	// TODO: rearrange array to [l][i][j][k]
	FOR(l, 0, 5)
		read_3D(fin, u, dim_g, l);
	FOR(l, 0, 6)
		read_3D(fin, q, dim_g, l);

	fscanf(fin, "%le %le %le\n", &dx[0], &dx[1], &dx[2]);
	fscanf(fin, "%d\n", &dummy);
	fscanf(fin, "%le\n", &courno);
	fclose(fin);

	gpu_copy_from_host_4D(d_u, u, dim_g, 5);
	gpu_copy_from_host_4D(d_q, q, dim_g, 6);

	printf("Applying ctoprim()...\n");
	gpu_ctoprim(h_const, d_const, d_u, d_q, courno);
//	ctoprim(lo, hi, u, q, dx, ng, courno);
	gpu_copy_to_host_4D(u, d_u, dim_g, 5);
	gpu_copy_to_host_4D(q, d_q, dim_g, 6);

	// Scanning output to check
	fscanf(fout, "%d %d %d\n", &lo2[0], &lo2[1], &lo2[2]);
	fscanf(fout, "%d %d %d\n", &hi2[0], &hi2[1], &hi2[2]);
	FOR(l, 0, 5)
		read_3D(fout, u2, dim_g, l);
	FOR(l, 0, 6)
		read_3D(fout, q2, dim_g, l);

	fscanf(fout, "%le %le %le\n", &dx2[0], &dx2[1], &dx2[2]);
	fscanf(fout, "%d\n", &ng2);
	fscanf(fout, "%le\n", &courno2);
	fclose(fout);

	// Checking...
	double u_max=-1.0;
	int j,k;
	FOR(l, 0, 5){
		FOR(i, 0, dim_g[0]){
			FOR(j, 0, dim_g[1]){
				FOR(k, 0, dim_g[2]){
					if(u_max < u[l][i][j][k])
						u_max = u[l][i][j][k];
				}
			}
		}
	}
	printf("u_max = %le\n", u_max);

	check_lo_hi_ng_dx(lo, hi, ng, dx, lo2, hi2, ng2, dx2);
	check_double(courno, courno2, "courno");
	check_4D_array("u", u, u2, dim_g, 5);
	check_4D_array("q", q, q2, dim_g, 6);
	printf("Correct!\n");

	gpu_free_4D(d_u);
	gpu_free_4D(d_q);

	free_4D(u,  dim_g, 5);		free_4D(q,  dim_g, 6);
	free_4D(u2, dim_g, 5);		free_4D(q2, dim_g, 6);
}

