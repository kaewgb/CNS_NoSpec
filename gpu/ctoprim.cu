#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "thrust/reduce.h"
#include "thrust/device_ptr.h"
#include "header.h"
#include "util.h"

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

	int i, j, k, idx, cour_idx, loffset;
	int numthreads = BLOCK_DIM;
	double rhoinv, eint, c, courx, coury, courz;

	cour_idx = blockIdx.x * blockDim.x + threadIdx.x;
	k =  cour_idx / (g->dim_g[0] * g->dim_g[1]);
	j = (cour_idx / g->dim_g[0]) % g->dim_g[1];
	i =  cour_idx % g->dim_g[0];
	idx = k*g->plane_offset_g_padded + j*g->pitch_g[0] + i;

	loffset = g->comp_offset_g_padded;

	// Calculate Q
	if( idx < loffset ){

		rhoinv 				= 1.0E0/u[idx];				//u(i,j,k,1) = u[0][i][j][k]
		q[idx] 				= u[idx]; 					//u(i,j,k,1) = u[0][i][j][k]
		q[idx+loffset] 		= u[idx+loffset]*rhoinv; 	//u(i,j,k,2) = u[1][i][j][k]
		q[idx+2*loffset] 	= u[idx+2*loffset]*rhoinv; 	//u(i,j,k,3) = u[2][i][j][k]
		q[idx+3*loffset] 	= u[idx+3*loffset]*rhoinv; 	//u(i,j,k,4) = u[3][i][j][k]

		eint = u[idx+4*loffset]*rhoinv - 0.5E0*(SQR(q[idx+loffset]) + SQR(q[idx+2*loffset]) + SQR(q[idx+3*loffset]));

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

			courno[cour_idx] = MAX(courx, MAX(coury, courz));
		}
		else
			courno[cour_idx] = -1.0;		//TODO: make it minus infinity
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
	k =  idx / (g->dim_g[0] * g->dim_g[1]);
	j = (idx / g->dim_g[0]) % g->dim_g[1];
	i =  idx % g->dim_g[0];
    idx = k*g->plane_offset_g_padded + j*g->pitch_g[0] + i;

	loffset = g->comp_offset_g_padded;

	// Calculate Q
	if( idx < loffset ){

		rhoinv 				= 1.0E0/u[idx];				//u(i,j,k,1) = u[0][i][j][k]
		q[idx] 				= u[idx]; 					//u(i,j,k,1) = u[0][i][j][k]
		q[idx+loffset] 		= u[idx+loffset]*rhoinv; 	//u(i,j,k,2) = u[1][i][j][k]
		q[idx+2*loffset] 	= u[idx+2*loffset]*rhoinv; 	//u(i,j,k,3) = u[2][i][j][k]
		q[idx+3*loffset] 	= u[idx+3*loffset]*rhoinv; 	//u(i,j,k,4) = u[3][i][j][k]

		eint = u[idx+4*loffset]*rhoinv - 0.5E0*(SQR(q[idx+loffset]) + SQR(q[idx+2*loffset]) + SQR(q[idx+3*loffset]));

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

	len = h_const.dim_g[0] * h_const.dim_g[1] * h_const.dim_g[2];
	int grid_dim = (len + BLOCK_DIM-1) / BLOCK_DIM;
	int block_dim = BLOCK_DIM;

	gpu_ctoprim_kernel<<<grid_dim, block_dim>>>(d_const, u_d, q_d, h_const.temp[0]);

	// Find max & update courno
	// TODO: make it minus infinity
	thrust::device_ptr<double> dev_ptr(h_const.temp[0]);
	courno = thrust::reduce(dev_ptr, dev_ptr + len, (double) -INFINITY, thrust::maximum<double>());

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
	DO(k, h.lo[2]-h.ng, h.hi[2]+h.ng){
        DO(j, h.lo[1]-h.ng, h.hi[1]+h.ng){
            DO(i, h.lo[0]-h.ng, h.hi[0]+h.ng){

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
	DO(k, h.lo[2], h.hi[2]){
		DO(j, h.lo[1], h.hi[1]){
			DO(i, h.lo[0], h.hi[0]){

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
    DO(k, h.lo[2]-h.ng, h.hi[2]+h.ng){
        DO(j, h.lo[1]-h.ng, h.hi[1]+h.ng){
			DO(i, h.lo[0]-h.ng, h.hi[0]+h.ng){
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


