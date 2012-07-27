#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "helper_functions.h"

#define	q(i,j,k,l)		q[l][i][j][k]
#define cons(i,j,k,l)	cons[l][i][j][k]
#define flux(i,j,k,l)	flux[l][i-ng][j-ng][k-ng]
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

				if(i==ng && j==ng && k==ng){
					printf("here\n");
					printf("%le %le\n%le %le\n%le %le\n%le %le\n",
								cons(i+1,j,k,imx), cons(i-1,j,k,imx),
								cons(i+2,j,k,imx), cons(i-2,j,k,imx),
								cons(i+3,j,k,imx), cons(i-3,j,k,imx),
								cons(i+4,j,k,imx), cons(i-4,j,k,imx));
				}

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
	printf("flux[0][0][0][0] = %le\n", flux[0][0][0][0]);

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
	printf("flux[0][0][0][0] = %le\n", flux[0][0][0][0]);

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
	printf("flux[0][0][0][0] = %le\n", flux[0][0][0][0]);
}

#undef	q(i,j,k,l)
#undef 	cons(i,j,k,l)
#undef 	flux(i,j,k,l)
#undef 	dxinv(i)


#define BLOCK_DIM_X		8
#define	BLOCK_DIM_Y		16
#define	s_q(i)			s_q[threadIdx.x+g->ng+i][threadIdx.y]
#define	s_qpres(i)		s_qpres[threadIdx.x+g->ng+i][threadIdx.y]
#define	s_cons(i, comp)	s_cons[comp][threadIdx.x+g->ng+i][threadIdx.y]

__global__ void gpu_hypterm_kernel(
	global_const_t *g,	// i:
	double *cons,		// i:
	double *q,			// i:
	double *flux		// o:
){
	int idx,bi,bj,bk;
	int si,sj,sk,tidx,tidy;
	double dxinv, unp1, unp2, unp3, unp4, unm1, unm2, unm3, unm4;
	double flux_irho, flux_imx, flux_imy, flux_imz, flux_iene;

	__shared__ double       s_q[BLOCK_DIM_X+NG+NG][BLOCK_DIM_Y];
	__shared__ double   s_qpres[BLOCK_DIM_X+NG+NG][BLOCK_DIM_Y];
	__shared__ double s_cons[4][BLOCK_DIM_X+NG+NG][BLOCK_DIM_Y];

	// Load to shared mem
	// TODO: boundary check
	bi = (blockIdx.x % (g->gridDim_plane_xy)) / g->gridDim_y;
	bj = (blockIdx.x % (g->gridDim_plane_xy)) % g->gridDim_y;
	bk =  blockIdx.x / (g->gridDim_plane_xy);
	si = bi*blockDim.x+threadIdx.x;
	sj = bj*blockDim.y+threadIdx.y;
	sk = bk;
	tidx = threadIdx.x;
	tidy = threadIdx.y;
	while( tidx < g->blockDim_x_g ){

        idx = si*g->plane_offset_g + sk*g->dim_g[2] + sj;

		           s_q[tidx][tidy]  =     q[idx + qu*g->comp_offset_g];
			   s_qpres[tidx][tidy]	=     q[idx + qpres*g->comp_offset_g];
		 s_cons[s_imx][tidx][tidy] 	=  cons[idx + s_imx*g->comp_offset_g];
		 s_cons[s_imy][tidx][tidy] 	=  cons[idx + s_imy*g->comp_offset_g];
		 s_cons[s_imz][tidx][tidy] 	=  cons[idx + s_imz*g->comp_offset_g];
		s_cons[s_iene][tidx][tidy] 	=  cons[idx + s_iene*g->comp_offset_g];

		tidx += blockDim.x;
		si   += blockDim.x;
	}
	__syncthreads();

	dxinv = 1.0E0/g->dx[0];
	unp1 = s_q(1); //q(i+1,j,k,qu);
	unp2 = s_q(2); //q(i+2,j,k,qu);
	unp3 = s_q(3); //q(i+3,j,k,qu);
	unp4 = s_q(4); //q(i+4,j,k,qu);

	unm1 = s_q(-1); //q(i-1,j,k,qu);
	unm2 = s_q(-2); //q(i-2,j,k,qu);
	unm3 = s_q(-3); //q(i-3,j,k,qu);
	unm4 = s_q(-4); //q(i-4,j,k,qu);

	flux_irho = - ( ALP*(s_cons(1, s_imx)-s_cons(-1, s_imx))
				  + BET*(s_cons(2, s_imx)-s_cons(-2, s_imx))
				  + GAM*(s_cons(3, s_imx)-s_cons(-3, s_imx))
				  + DEL*(s_cons(4, s_imx)-s_cons(-4, s_imx)))*dxinv;

//	flux_irho = s_cons(1, s_imx);


	flux_imx  = - ( ALP*(s_cons(1, s_imx)*unp1-s_cons(-1, s_imx)*unm1
				  + (s_qpres(1)-s_qpres(-1)))
				  + BET*(s_cons(2, s_imx)*unp2-s_cons(-2, s_imx)*unm2
				  + (s_qpres(2)-s_qpres(-2)))
				  + GAM*(s_cons(3, s_imx)*unp3-s_cons(-3, s_imx)*unm3
				  + (s_qpres(3)-s_qpres(-3)))
				  + DEL*(s_cons(4, s_imx)*unp4-s_cons(-4, s_imx)*unm4
				  + (s_qpres(4)-s_qpres(-4))))*dxinv;

	flux_imy  = - ( ALP*(s_cons(1, s_imy)*unp1-s_cons(-1, s_imy)*unm1)
				  + BET*(s_cons(2, s_imy)*unp2-s_cons(-2, s_imy)*unm2)
				  + GAM*(s_cons(3, s_imy)*unp3-s_cons(-3, s_imy)*unm3)
				  + DEL*(s_cons(4, s_imy)*unp4-s_cons(-4, s_imy)*unm4))*dxinv;

	flux_imz  = - ( ALP*(s_cons(1, s_imz)*unp1-s_cons(-1, s_imz)*unm1)
				  + BET*(s_cons(2, s_imz)*unp2-s_cons(-2, s_imz)*unm2)
				  + GAM*(s_cons(3, s_imz)*unp3-s_cons(-3, s_imz)*unm3)
				  + DEL*(s_cons(4, s_imz)*unp4-s_cons(-4, s_imz)*unm4))*dxinv;

	flux_iene = - ( ALP*(s_cons(1, s_iene)*unp1-s_cons(-1, s_iene)*unm1
				  + (s_qpres(1)*unp1-s_qpres(-1)*unm1))
				  + BET*(s_cons(2, s_iene)*unp2-s_cons(-2, s_iene)*unm2
				  + (s_qpres(2)*unp2-s_qpres(-2)*unm2))
				  + GAM*(s_cons(3, s_iene)*unp3-s_cons(-3, s_iene)*unm3
				  + (s_qpres(3)*unp3-s_qpres(-3)*unm3))
				  + DEL*(s_cons(4, s_iene)*unp4-s_cons(-4, s_iene)*unm4
				  + (s_qpres(4)*unp4-s_qpres(-4)*unm4)))*dxinv;

	// Load to shared mem
	// s_q -> qv
	// j is the 1st dim (moving)
	// dim map = (j, i, k)
	// TODO: boundary check
	__syncthreads();
	griddim_x = (g->dim[0] + blockDim.x -1)/blockDim.x;
	griddim_y = (g->dim[1] + blockDim.y -1)/blockDim.y;
	bi = (blockIdx.x % (griddim_x*griddim_y)) / g->dim[1];
	bj = (blockIdx.x % (griddim_x*griddim_y)) % g->dim[1];
	bk =  blockIdx.x / (griddim_x*griddim_y);
	si = bi*blockDim.x+threadIdx.x;
	sj = bj*blockDim.y+threadIdx.y;
	sk = bk;
	tidx = threadIdx.x;
	tidy = threadIdx.y;
	__syncthreads();
	while( tidx < blockDim.x+g->ng+g->ng && tidy < blockDim.y ){

		           s_q[tidx][tidy]  =     q[(sj+tidy)*g->dim[1]*g->dim[2] + (si+tidx)*g->dim[2] + sk + qv*g->comp_offset_g];
			   s_qpres[tidx][tidy]	= 	  q[(sj+tidy)*g->dim[1]*g->dim[2] + (si+tidx)*g->dim[2] + sk + qpres*g->comp_offset_g];
		 s_cons[s_imx][tidx][tidy] 	=  cons[(sj+tidy)*g->dim[1]*g->dim[2] + (si+tidx)*g->dim[2] + sk + s_imx*g->comp_offset_g];
		 s_cons[s_imy][tidx][tidy] 	=  cons[(sj+tidy)*g->dim[1]*g->dim[2] + (si+tidx)*g->dim[2] + sk + s_imy*g->comp_offset_g];
		 s_cons[s_imz][tidx][tidy] 	=  cons[(sj+tidy)*g->dim[1]*g->dim[2] + (si+tidx)*g->dim[2] + sk + s_imz*g->comp_offset_g];
		s_cons[s_iene][tidx][tidy] 	=  cons[(sj+tidy)*g->dim[1]*g->dim[2] + (si+tidx)*g->dim[2] + sk + s_iene*g->comp_offset_g];

		tidx += blockDim.x;
		tidy += blockDim.y;
	}

	dxinv = 1.0E0/g->dx[1];
	unp1 = s_q(1); 		//q(i,j+1,k,qv);
	unp2 = s_q(2); 		//q(i,j+2,k,qv);
	unp3 = s_q(3); 		//q(i,j+3,k,qv);
	unp4 = s_q(4); 		//q(i,j+4,k,qv);

	unm1 = s_q(-1); 	//q(i,j-1,k,qv);
	unm2 = s_q(-2); 	//q(i,j-2,k,qv);
	unm3 = s_q(-3); 	//q(i,j-3,k,qv);
	unm4 = s_q(-4); 	//q(i,j-4,k,qv);

	flux_irho -=   ( ALP*(s_cons(1, s_imy)-s_cons(-1, s_imy))
				  + BET*(s_cons(2, s_imy)-s_cons(-2, s_imy))
				  + GAM*(s_cons(3, s_imy)-s_cons(-3, s_imy))
				  + DEL*(s_cons(4, s_imy)-s_cons(-4, s_imy)))*dxinv;

	flux_imx -=   ( ALP*(s_cons(1, s_imx)*unp1-s_cons(-1, s_imx)*unm1)
				  + BET*(s_cons(2, s_imx)*unp2-s_cons(-2, s_imx)*unm2)
				  + GAM*(s_cons(3, s_imx)*unp3-s_cons(-3, s_imx)*unm3)
				  + DEL*(s_cons(4, s_imx)*unp4-s_cons(-4, s_imx)*unm4))*dxinv;

	flux_imy -=   ( ALP*(s_cons(1, s_imy)*unp1-s_cons(-1, s_imy)*unm1
				  + (s_qpres(1)-s_qpres(-1)))
				  + BET*(s_cons(2, s_imy)*unp2-s_cons(-2, s_imy)*unm2
				  + (s_qpres(2)-s_qpres(-2)))
				  + GAM*(s_cons(3, s_imy)*unp3-s_cons(-3, s_imy)*unm3
				  + (s_qpres(3)-s_qpres(-3)))
				  + DEL*(s_cons(4, s_imy)*unp4-s_cons(-4, s_imy)*unm4
				  + (s_qpres(4)-s_qpres(-4))))*dxinv;

	flux_imz -=   ( ALP*(s_cons(1, s_imz)*unp1-s_cons(-1, s_imz)*unm1)
				  + BET*(s_cons(2, s_imz)*unp2-s_cons(-2, s_imz)*unm2)
				  + GAM*(s_cons(3, s_imz)*unp3-s_cons(-3, s_imz)*unm3)
				  + DEL*(s_cons(4, s_imz)*unp4-s_cons(-4, s_imz)*unm4))*dxinv;

	flux_iene -=  ( ALP*(s_cons(1, s_iene)*unp1-s_cons(-1, s_iene)*unm1
				  + (s_qpres(1)*unp1-s_qpres(-1)*unm1))
				  + BET*(s_cons(2, s_iene)*unp2-s_cons(-2, s_iene)*unm2
				  + (s_qpres(2)*unp2-s_qpres(-2)*unm2))
				  + GAM*(s_cons(3, s_iene)*unp3-s_cons(-3, s_iene)*unm3
				  + (s_qpres(3)*unp3-s_qpres(-3)*unm3))
				  + DEL*(s_cons(4, s_iene)*unp4-s_cons(-4, s_iene)*unm4
				  + (s_qpres(4)*unp4-s_qpres(-4)*unm4)))*dxinv;



	// Load to shared mem
	// s_q -> qw
	// k is the 1st dim
	// TODO: check boundary
	__syncthreads();
	griddim_x = (g->dim[0] + blockDim.x -1)/blockDim.x;
	griddim_y = (g->dim[1] + blockDim.y -1)/blockDim.y;
	bi = (blockIdx.x % (griddim_x*griddim_y)) / g->dim[1];
	bj = (blockIdx.x % (griddim_x*griddim_y)) % g->dim[1];
	bk =  blockIdx.x / (griddim_x*griddim_y);
	si = bi*blockDim.x+threadIdx.x;
	sj = bj*blockDim.y+threadIdx.y;
	sk = bk;
	tidx = threadIdx.x;
	tidy = threadIdx.y;
	__syncthreads();
	while( tidx < blockDim.x+g->ng+g->ng && tidy < blockDim.y ){

		           s_q[tidx][tidy]  =     q[sk*g->dim[1]*g->dim[2] + (sj+tidy)*g->dim[2] + (si+tidx) + qw*g->comp_offset_g];
			   s_qpres[tidx][tidy]	=     q[sk*g->dim[1]*g->dim[2] + (sj+tidy)*g->dim[2] + (si+tidx) + qpres*g->comp_offset_g];
		 s_cons[s_imx][tidx][tidy] 	=  cons[sk*g->dim[1]*g->dim[2] + (sj+tidy)*g->dim[2] + (si+tidx) + s_imx*g->comp_offset_g];
		 s_cons[s_imy][tidx][tidy] 	=  cons[sk*g->dim[1]*g->dim[2] + (sj+tidy)*g->dim[2] + (si+tidx) + s_imy*g->comp_offset_g];
		 s_cons[s_imz][tidx][tidy] 	=  cons[sk*g->dim[1]*g->dim[2] + (sj+tidy)*g->dim[2] + (si+tidx) + s_imz*g->comp_offset_g];
		s_cons[s_iene][tidx][tidy] 	=  cons[sk*g->dim[1]*g->dim[2] + (sj+tidy)*g->dim[2] + (si+tidx) + s_iene*g->comp_offset_g];

		tidx += blockDim.x;
		tidy += blockDim.y;
	}

	dxinv = 1.0E0/g->dx[2];
	unp1 = s_q(1);	//q(i,j,k+1,qw);
	unp2 = s_q(2);	//q(i,j,k+2,qw);
	unp3 = s_q(3);	//q(i,j,k+3,qw);
	unp4 = s_q(4);	//q(i,j,k+4,qw);

	unm1 = s_q(-1);	//q(i,j,k-1,qw);
	unm2 = s_q(-2);	//q(i,j,k-2,qw);
	unm3 = s_q(-3);	//q(i,j,k-3,qw);
	unm4 = s_q(-4);	//q(i,j,k-4,qw);

	flux_irho -=  ( ALP*(s_cons(1,s_imz)-s_cons(-1,s_imz))
				  + BET*(s_cons(2,s_imz)-s_cons(-2,s_imz))
				  + GAM*(s_cons(3,s_imz)-s_cons(-3,s_imz))
				  + DEL*(s_cons(4,s_imz)-s_cons(-4,s_imz)))*dxinv;

	flux_imx -=   ( ALP*(s_cons(1,s_imx)*unp1-s_cons(-1,s_imx)*unm1)
				  + BET*(s_cons(2,s_imx)*unp2-s_cons(-2,s_imx)*unm2)
				  + GAM*(s_cons(3,s_imx)*unp3-s_cons(-3,s_imx)*unm3)
				  + DEL*(s_cons(4,s_imx)*unp4-s_cons(-4,s_imx)*unm4))*dxinv;

	flux_imy -=   ( ALP*(s_cons(1,s_imy)*unp1-s_cons(-1,s_imy)*unm1)
				  + BET*(s_cons(2,s_imy)*unp2-s_cons(-2,s_imy)*unm2)
				  + GAM*(s_cons(3,s_imy)*unp3-s_cons(-3,s_imy)*unm3)
				  + DEL*(s_cons(4,s_imy)*unp4-s_cons(-4,s_imy)*unm4))*dxinv;

	flux_imz -=   ( ALP*(s_cons(1,s_imz)*unp1-s_cons(-1,s_imz)*unm1
				  + (s_qpres(1)-s_qpres(-1)))
				  + BET*(s_cons(2,s_imz)*unp2-s_cons(-2,s_imz)*unm2
				  + (s_qpres(2)-s_qpres(-2)))
				  + GAM*(s_cons(3,s_imz)*unp3-s_cons(-3,s_imz)*unm3
				  + (s_qpres(3)-s_qpres(-3)))
				  + DEL*(s_cons(4,s_imz)*unp4-s_cons(-4,s_imz)*unm4
				+ (s_qpres(4)-s_qpres(-4))))*dxinv;

	flux_iene -= ( ALP*(s_cons(1,s_iene)*unp1-s_cons(-1,s_iene)*unm1
				  + (s_qpres(1)*unp1-s_qpres(-1)*unm1))
				  + BET*(s_cons(2,s_iene)*unp2-s_cons(-2,s_iene)*unm2
				  + (s_qpres(2)*unp2-s_qpres(-2)*unm2))
				  + GAM*(s_cons(3,s_iene)*unp3-s_cons(-3,s_iene)*unm3
				  + (s_qpres(3)*unp3-s_qpres(-3)*unm3))
				  + DEL*(s_cons(4,s_iene)*unp4-s_cons(-4,s_iene)*unm4
				  + (s_qpres(4)*unp4-s_qpres(-4)*unm4)))*dxinv;

	// Update changes
	bi = (blockIdx.x % (g->gridDim_plane_xy)) / g->gridDim_y;
	bj = (blockIdx.x % (g->gridDim_plane_xy)) % g->gridDim_y;
	bk =  blockIdx.x / (g->gridDim_plane_xy);
	si = bi*blockDim.x+threadIdx.x + g->ng;
	sj = bj*blockDim.y+threadIdx.y + g->ng;
	sk = bk + g-ng;

    idx = si*g->plane_offset_g + sk*g->dim_g[2] + sj;

//	flux[idx + irho*g->comp_offset_g] = flux_irho;
	flux[idx + imx *g->comp_offset_g] = flux_imx;
	flux[idx + imy *g->comp_offset_g] = flux_imy;
	flux[idx + imz *g->comp_offset_g] = flux_imz;
	flux[idx + iene*g->comp_offset_g] = flux_iene;

	if(idx == 0)
		flux[idx + irho*g->comp_offset_g] = flux_irho;

}
void gpu_hypterm(
	global_const_t h_const, 	// i: Global struct containing applicatino parameters
	global_const_t *d_const,	// i: Device pointer to global struct containing application paramters
	double *d_cons,				// i:
	double *d_q,				// i:
	double *d_flux				// o: flux
){
	int i, len, dim[3];
	int grid_dim, grid_dim_x, grid_dim_y;
	FOR(i, 0, 3)
		dim[i] = h_const.dim[i];

	// TODO: Make sure it supports non-square box
	len = dim[0] * dim[1] * dim[2];
	grid_dim_x = (dim[0]+BLOCK_DIM_X-1)/BLOCK_DIM_X;
	grid_dim_y = (dim[1]+BLOCK_DIM_Y-1)/BLOCK_DIM_Y;
	grid_dim = grid_dim_x * grid_dim_y * dim[2];

	dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
    h_const.gridDim_x = grid_dim_x;
    h_const.gridDim_y = grid_dim_y;
    h_const.gridDim_z = dim[2];
    h_const.gridDim_plane_xy = grid_dim_x * grid_dim_y;
    h_const.blockDim_x_g = BLOCK_DIM_X + h_const.ng + h_const.ng;
    cudaMemcpy(d_const, &h_const, sizeof(global_const_t), cudaMemcpyHostToDevice);

	gpu_hypterm_kernel<<<grid_dim, block_dim>>>(d_const, d_cons, d_q, d_flux);

}

void hypterm_test(
	global_const_t h_const, // i: Global struct containing applicatino parameters
	global_const_t *d_const	// i: Device pointer to global struct containing application paramters
){

	int lo[3], hi[3], ng;
	double dx[3];
	double ****cons, ****q, ****flux;

	int lo2[3], hi2[3], ng2;
	double dx2[3];
	double ****cons2, ****q2, ****flux2;

	double *d_cons, *d_q, *d_flux;

	int i, l;
	int dim_g[3], dim[3];

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
		dim[i] = hi[i]-lo[i]+1;
		dim_g[i]  = dim[i] + 2*ng;
	}

	allocate_4D(cons, 	dim_g, 	5);		// [40][40][40][5]
	allocate_4D(q, 		dim_g, 	6);		// [40][40][40][6]
	allocate_4D(flux, 	dim, 	5);		// [32][32][32][5]
	allocate_4D(cons2, 	dim_g, 	5);		// [40][40][40][5]
	allocate_4D(q2, 	dim_g, 	6);	 	// [40][40][40][6]
	allocate_4D(flux2, 	dim, 	5);		// [40][40][40][5]

	gpu_allocate_4D(d_cons, dim_g, 5);
	gpu_allocate_4D(d_q,	dim_g, 6);
	gpu_allocate_4D(d_flux, dim,   5);

	FOR(l, 0, 5)
		read_3D(fin, cons,  dim_g,  l);
	FOR(l, 0, 6)
		read_3D(fin, q,		dim_g,  l);
	FOR(l, 0, 5)
		read_3D(fin, flux,  dim, l);
	fclose(fin);

	gpu_copy_from_host_4D(d_cons, cons, dim_g, 5);
	gpu_copy_from_host_4D(d_q, 	  q, 	dim_g, 6);
	gpu_copy_from_host_4D(d_flux, flux, dim  , 5);

	printf("Applying hypterm()...\n");
//	hypterm(lo, hi, ng, dx, cons, q, flux);
	gpu_hypterm(h_const, d_const, d_cons, d_q, d_flux);

	gpu_copy_to_host_4D(cons, d_cons, dim_g, 5);
	gpu_copy_to_host_4D(q   , d_q   , dim_g, 6);
	gpu_copy_to_host_4D(flux, d_flux, dim  , 5);

	// Scanning output to check

	fscanf(fout, "%d %d %d\n", &lo2[0], &lo2[1], &lo2[2]);
	fscanf(fout, "%d %d %d\n", &hi2[0], &hi2[1], &hi2[2]);
	fscanf(fout, "%d\n", &ng2);
	fscanf(fout, "%le %le %le\n", &dx2[0], &dx2[1], &dx2[2]);

	printf("read solutions\n");
	FOR(l, 0, 5)
		read_3D(fout, cons2, dim_g,  l);
	FOR(l, 0, 6)
		read_3D(fout, q2,	 dim_g,  l);
	FOR(l, 0, 5)
		read_3D(fout, flux2,  dim, l);
	fclose(fout);

	// Checking...
	printf("checking answers..\n");
	check_lo_hi_ng_dx(lo, hi, ng, dx, lo2, hi2, ng2, dx2);
	check_4D_array("cons", cons, cons2, dim_g,  5);
	check_4D_array("q",    q, 	 q2,	dim_g,  6);
	check_4D_array("flux", flux, flux2, dim, 5);

	gpu_free_4D(d_cons);
	gpu_free_4D(d_q);
	gpu_free_4D(d_flux);

	free_4D(cons,  dim_g, 5);	free_4D(q,  dim_g, 6);	free_4D(flux,  dim, 5);
	free_4D(cons2, dim_g, 5);	free_4D(q2, dim_g, 6);	free_4D(flux2, dim, 5);

	printf("Correct!\n");
}
