#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "util.cuh"
#include "util.h"

#define BLOCK_SMALL		8
#define	BLOCK_LARGE		16
#define	s_q(i)			s_q[threadIdx.y][threadIdx.x+g->ng+(i)]
#define	s_qpres(i)		s_qpres[threadIdx.y][threadIdx.x+g->ng+(i)]
#define	s_cons(i, comp)	s_cons[comp][threadIdx.y][threadIdx.x+g->ng+(i)]

__global__ void gpu_hypterm_x_stencil_kernel(
	global_const_t *g,	// i:
	double *cons,		// i:
	double *q,			// i:
	double *flux		// o:
){
	int idx,si,sj,sk,tidx,tidy;
	double dxinv, unp1, unp2, unp3, unp4, unm1, unm2, unm3, unm4;
	double flux_irho, flux_imx, flux_imy, flux_imz, flux_iene;

	__shared__ double       s_q[BLOCK_LARGE][BLOCK_SMALL+NG+NG];
	__shared__ double   s_qpres[BLOCK_LARGE][BLOCK_SMALL+NG+NG];
	__shared__ double s_cons[4][BLOCK_LARGE][BLOCK_SMALL+NG+NG];

	// Load to shared mem
	si = blockIdx.x*blockDim.x+threadIdx.x;
	sj = blockIdx.y*blockDim.y+threadIdx.y;
	sk = blockIdx.z*blockDim.z+threadIdx.z;

	tidx = threadIdx.x;
	tidy = threadIdx.y;
	while( tidx < BLOCK_SMALL+NG+NG && si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){

		idx = (sk+g->ng)*g->plane_offset_g_padded + (sj+g->ng)*g->pitch_g[0] + si;

				   s_q[tidy][tidx]  =     q[idx + qu*g->comp_offset_g_padded];
			   s_qpres[tidy][tidx]	=     q[idx + qpres*g->comp_offset_g_padded];
		 s_cons[s_imx][tidy][tidx] 	=  cons[idx + imx*g->comp_offset_g_padded];
		 s_cons[s_imy][tidy][tidx] 	=  cons[idx + imy*g->comp_offset_g_padded];
		 s_cons[s_imz][tidy][tidx] 	=  cons[idx + imz*g->comp_offset_g_padded];
		s_cons[s_iene][tidy][tidx] 	=  cons[idx + iene*g->comp_offset_g_padded];

		tidx += blockDim.x;
		si   += blockDim.x;
	}
	__syncthreads();

	si = blockIdx.x*blockDim.x+threadIdx.x;
	if(si < g->dim[0] && sj < g->dim[1] && sk < g->dim[2]){

		dxinv = 1.0E0/g->dx[0];
		unp1 = s_q(1); //q(i+1,j,k,qu);
		unp2 = s_q(2); //q(i+2,j,k,qu);
		unp3 = s_q(3); //q(i+3,j,k,qu);
		unp4 = s_q(4); //q(i+4,j,k,qu);

		unm1 = s_q(-1); //q(i-1,j,k,qu);
		unm2 = s_q(-2); //q(i-2,j,k,qu);
		unm3 = s_q(-3); //q(i-3,j,k,qu);
		unm4 = s_q(-4); //q(i-4,j,k,qu);

		flux_irho = - ( g->ALP*(s_cons(1, s_imx)-s_cons(-1, s_imx))
					  + g->BET*(s_cons(2, s_imx)-s_cons(-2, s_imx))
					  + g->GAM*(s_cons(3, s_imx)-s_cons(-3, s_imx))
					  + g->DEL*(s_cons(4, s_imx)-s_cons(-4, s_imx)))*dxinv;

		flux_imx  = - ( g->ALP*(s_cons(1, s_imx)*unp1-s_cons(-1, s_imx)*unm1
					  + (s_qpres(1)-s_qpres(-1)))
					  + g->BET*(s_cons(2, s_imx)*unp2-s_cons(-2, s_imx)*unm2
					  + (s_qpres(2)-s_qpres(-2)))
					  + g->GAM*(s_cons(3, s_imx)*unp3-s_cons(-3, s_imx)*unm3
					  + (s_qpres(3)-s_qpres(-3)))
					  + g->DEL*(s_cons(4, s_imx)*unp4-s_cons(-4, s_imx)*unm4
					  + (s_qpres(4)-s_qpres(-4))))*dxinv;

		flux_imy  = - ( g->ALP*(s_cons(1, s_imy)*unp1-s_cons(-1, s_imy)*unm1)
					  + g->BET*(s_cons(2, s_imy)*unp2-s_cons(-2, s_imy)*unm2)
					  + g->GAM*(s_cons(3, s_imy)*unp3-s_cons(-3, s_imy)*unm3)
					  + g->DEL*(s_cons(4, s_imy)*unp4-s_cons(-4, s_imy)*unm4))*dxinv;

		flux_imz  = - ( g->ALP*(s_cons(1, s_imz)*unp1-s_cons(-1, s_imz)*unm1)
					  + g->BET*(s_cons(2, s_imz)*unp2-s_cons(-2, s_imz)*unm2)
					  + g->GAM*(s_cons(3, s_imz)*unp3-s_cons(-3, s_imz)*unm3)
					  + g->DEL*(s_cons(4, s_imz)*unp4-s_cons(-4, s_imz)*unm4))*dxinv;

		flux_iene = - ( g->ALP*(s_cons(1, s_iene)*unp1-s_cons(-1, s_iene)*unm1
					  + (s_qpres(1)*unp1-s_qpres(-1)*unm1))
					  + g->BET*(s_cons(2, s_iene)*unp2-s_cons(-2, s_iene)*unm2
					  + (s_qpres(2)*unp2-s_qpres(-2)*unm2))
					  + g->GAM*(s_cons(3, s_iene)*unp3-s_cons(-3, s_iene)*unm3
					  + (s_qpres(3)*unp3-s_qpres(-3)*unm3))
					  + g->DEL*(s_cons(4, s_iene)*unp4-s_cons(-4, s_iene)*unm4
					  + (s_qpres(4)*unp4-s_qpres(-4)*unm4)))*dxinv;

		// Update changes
		idx = sk*g->plane_offset_padded + sj*g->pitch[0] + si;

//		atomicAdd(flux + idx + irho*g->comp_offset_padded, flux_irho);
//		atomicAdd(flux + idx + imx *g->comp_offset_padded, flux_imx);
//		atomicAdd(flux + idx + imy *g->comp_offset_padded, flux_imy);
//		atomicAdd(flux + idx + imz *g->comp_offset_padded, flux_imz);
//		atomicAdd(flux + idx + iene*g->comp_offset_padded, flux_iene);
		flux[idx + irho*g->comp_offset_padded] = flux_irho;
		flux[idx + imx *g->comp_offset_padded] = flux_imx;
		flux[idx + imy *g->comp_offset_padded] = flux_imy;
		flux[idx + imz *g->comp_offset_padded] = flux_imz;
		flux[idx + iene*g->comp_offset_padded] = flux_iene;
	}
}
#undef	s_q
#undef 	s_qpres
#undef	s_cons

#define	s_q(i)			s_q[threadIdx.y+g->ng+(i)][threadIdx.x]
#define	s_qpres(i)		s_qpres[threadIdx.y+g->ng+(i)][threadIdx.x]
#define	s_cons(i, comp)	s_cons[comp][threadIdx.y+g->ng+(i)][threadIdx.x]

__global__ void gpu_hypterm_y_stencil_kernel(
	global_const_t *g,	// i:
	double *cons,		// i:
	double *q,			// i:
	double *flux		// o:
){
	int idx,si,sj,sk,tidx,tidy;
	double dxinv, unp1, unp2, unp3, unp4, unm1, unm2, unm3, unm4;
	double flux_irho, flux_imx, flux_imy, flux_imz, flux_iene;

	__shared__ double       s_q[BLOCK_SMALL+NG+NG][BLOCK_LARGE];
	__shared__ double   s_qpres[BLOCK_SMALL+NG+NG][BLOCK_LARGE];
	__shared__ double s_cons[4][BLOCK_SMALL+NG+NG][BLOCK_LARGE];

	// Load to shared mem
	si = blockIdx.x*blockDim.x+threadIdx.x;
	sj = blockIdx.y*blockDim.y+threadIdx.y;
	sk = blockIdx.z*blockDim.z+threadIdx.z;

	tidx = threadIdx.x;
	tidy = threadIdx.y;
	while( tidy < BLOCK_SMALL+NG+NG && si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){

		idx = (sk+g->ng)*g->plane_offset_g_padded + sj*g->pitch_g[0] + (si+g->ng);

				   s_q[tidy][tidx]  =     q[idx + qv*g->comp_offset_g_padded];
			   s_qpres[tidy][tidx]	=     q[idx + qpres*g->comp_offset_g_padded];
		 s_cons[s_imx][tidy][tidx] 	=  cons[idx + imx*g->comp_offset_g_padded];
		 s_cons[s_imy][tidy][tidx] 	=  cons[idx + imy*g->comp_offset_g_padded];
		 s_cons[s_imz][tidy][tidx] 	=  cons[idx + imz*g->comp_offset_g_padded];
		s_cons[s_iene][tidy][tidx] 	=  cons[idx + iene*g->comp_offset_g_padded];

		tidy += blockDim.y;
		sj   += blockDim.y;
	}
	__syncthreads();

	sj = blockIdx.y*blockDim.y+threadIdx.y;
	if(si < g->dim[0] && sj < g->dim[1] && sk < g->dim[2]){

		dxinv = 1.0E0/g->dx[1];
		unp1 = s_q(1); 		//q(i,j+1,k,qv);
		unp2 = s_q(2); 		//q(i,j+2,k,qv);
		unp3 = s_q(3); 		//q(i,j+3,k,qv);
		unp4 = s_q(4); 		//q(i,j+4,k,qv);

		unm1 = s_q(-1); 	//q(i,j-1,k,qv);
		unm2 = s_q(-2); 	//q(i,j-2,k,qv);
		unm3 = s_q(-3); 	//q(i,j-3,k,qv);
		unm4 = s_q(-4); 	//q(i,j-4,k,qv);

		flux_irho =  ( g->ALP*(s_cons(1, s_imy)-s_cons(-1, s_imy))
					  + g->BET*(s_cons(2, s_imy)-s_cons(-2, s_imy))
					  + g->GAM*(s_cons(3, s_imy)-s_cons(-3, s_imy))
					  + g->DEL*(s_cons(4, s_imy)-s_cons(-4, s_imy)))*dxinv;

		flux_imx =   ( g->ALP*(s_cons(1, s_imx)*unp1-s_cons(-1, s_imx)*unm1)
					  + g->BET*(s_cons(2, s_imx)*unp2-s_cons(-2, s_imx)*unm2)
					  + g->GAM*(s_cons(3, s_imx)*unp3-s_cons(-3, s_imx)*unm3)
					  + g->DEL*(s_cons(4, s_imx)*unp4-s_cons(-4, s_imx)*unm4))*dxinv;

		flux_imy =   ( g->ALP*(s_cons(1, s_imy)*unp1-s_cons(-1, s_imy)*unm1
					  + (s_qpres(1)-s_qpres(-1)))
					  + g->BET*(s_cons(2, s_imy)*unp2-s_cons(-2, s_imy)*unm2
					  + (s_qpres(2)-s_qpres(-2)))
					  + g->GAM*(s_cons(3, s_imy)*unp3-s_cons(-3, s_imy)*unm3
					  + (s_qpres(3)-s_qpres(-3)))
					  + g->DEL*(s_cons(4, s_imy)*unp4-s_cons(-4, s_imy)*unm4
					  + (s_qpres(4)-s_qpres(-4))))*dxinv;

		flux_imz =   ( g->ALP*(s_cons(1, s_imz)*unp1-s_cons(-1, s_imz)*unm1)
					  + g->BET*(s_cons(2, s_imz)*unp2-s_cons(-2, s_imz)*unm2)
					  + g->GAM*(s_cons(3, s_imz)*unp3-s_cons(-3, s_imz)*unm3)
					  + g->DEL*(s_cons(4, s_imz)*unp4-s_cons(-4, s_imz)*unm4))*dxinv;

		flux_iene =  ( g->ALP*(s_cons(1, s_iene)*unp1-s_cons(-1, s_iene)*unm1
					  + (s_qpres(1)*unp1-s_qpres(-1)*unm1))
					  + g->BET*(s_cons(2, s_iene)*unp2-s_cons(-2, s_iene)*unm2
					  + (s_qpres(2)*unp2-s_qpres(-2)*unm2))
					  + g->GAM*(s_cons(3, s_iene)*unp3-s_cons(-3, s_iene)*unm3
					  + (s_qpres(3)*unp3-s_qpres(-3)*unm3))
					  + g->DEL*(s_cons(4, s_iene)*unp4-s_cons(-4, s_iene)*unm4
					  + (s_qpres(4)*unp4-s_qpres(-4)*unm4)))*dxinv;

		// Update changes
		idx = sk*g->plane_offset_padded + sj*g->pitch[0] + si;

//		if(flux[idx + irho*g->comp_offset_padded]!=4.0){
//			printf("si=%d, sj=%d, sk=%d, flux[%d]=%le\n", si, sj, sk, idx + irho*g->comp_offset_padded, flux[idx + irho*g->comp_offset_padded]);
//		}
//		if(flux[idx + imx*g->comp_offset_padded]!=4.0){
//			printf("si=%d, sj=%d, sk=%d, flux[%d]=%le\n", si, sj, sk, idx + imx*g->comp_offset_padded, flux[idx + imx*g->comp_offset_padded]);
//		}
//		if(flux[idx + imy*g->comp_offset_padded]!=4.0){
//			printf("si=%d, sj=%d, sk=%d, flux[%d]=%le\n", si, sj, sk, idx + imy*g->comp_offset_padded, flux[idx + imy*g->comp_offset_padded]);
//		}
//		if(flux[idx + imz*g->comp_offset_padded]!=4.0){
//			printf("si=%d, sj=%d, sk=%d, flux[%d]=%le\n", si, sj, sk, idx + imz*g->comp_offset_padded, flux[idx + imz*g->comp_offset_padded]);
//		}
//		if(flux[idx + iene*g->comp_offset_padded]!=4.0){
//			printf("si=%d, sj=%d, sk=%d, flux[%d]=%le\n", si, sj, sk, idx + iene*g->comp_offset_padded, flux[idx + iene*g->comp_offset_padded]);
//		}
//		flux[idx + irho*g->comp_offset_padded] -= flux_irho + 4.0;
//		flux[idx + imx *g->comp_offset_padded] -= flux_imx + 4.0;
//		flux[idx + imy *g->comp_offset_padded] -= flux_imy + 4.0;
//		flux[idx + imz *g->comp_offset_padded] -= flux_imz + 4.0 ;
//		flux[idx + iene*g->comp_offset_padded] -= flux_iene + 4.0;

		flux[idx + irho*g->comp_offset_padded] -= flux_irho;
		flux[idx + imx *g->comp_offset_padded] -= flux_imx;
		flux[idx + imy *g->comp_offset_padded] -= flux_imy;
		flux[idx + imz *g->comp_offset_padded] -= flux_imz;
		flux[idx + iene*g->comp_offset_padded] -= flux_iene;
	}
}
#undef	s_q
#undef 	s_qpres
#undef	s_cons

#define	s_q(i)			s_q[threadIdx.z+g->ng+(i)][threadIdx.x]
#define	s_qpres(i)		s_qpres[threadIdx.z+g->ng+(i)][threadIdx.x]
#define	s_cons(i, comp)	s_cons[comp][threadIdx.z+g->ng+(i)][threadIdx.x]

__global__ void gpu_hypterm_z_stencil_kernel(
	global_const_t *g,	// i:
	double *cons,		// i:
	double *q,			// i:
	double *flux		// o:
){
	int idx,si,sj,sk,tidx,tidz;
	double dxinv, unp1, unp2, unp3, unp4, unm1, unm2, unm3, unm4;
	double flux_irho, flux_imx, flux_imy, flux_imz, flux_iene;

	__shared__ double       s_q[BLOCK_SMALL+NG+NG][BLOCK_LARGE];
	__shared__ double   s_qpres[BLOCK_SMALL+NG+NG][BLOCK_LARGE];
	__shared__ double s_cons[4][BLOCK_SMALL+NG+NG][BLOCK_LARGE];

	// Load to shared mem
	si = blockIdx.x*blockDim.x+threadIdx.x;
	sj = blockIdx.y*blockDim.y+threadIdx.y;
	sk = blockIdx.z*blockDim.z+threadIdx.z;

	tidx = threadIdx.x;
	tidz = threadIdx.z;
	while( tidz < BLOCK_SMALL+NG+NG && si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){

		idx = sk*g->plane_offset_g_padded + (sj+g->ng)*g->pitch_g[0] + (si+g->ng);

				   s_q[tidz][tidx]  =     q[idx + qw*g->comp_offset_g_padded];
			   s_qpres[tidz][tidx]	=     q[idx + qpres*g->comp_offset_g_padded];
		 s_cons[s_imx][tidz][tidx] 	=  cons[idx + imx*g->comp_offset_g_padded];
		 s_cons[s_imy][tidz][tidx] 	=  cons[idx + imy*g->comp_offset_g_padded];
		 s_cons[s_imz][tidz][tidx] 	=  cons[idx + imz*g->comp_offset_g_padded];
		s_cons[s_iene][tidz][tidx] 	=  cons[idx + iene*g->comp_offset_g_padded];

		tidz += blockDim.z;
		sk   += blockDim.z;
	}
	__syncthreads();

	sk = blockIdx.z*blockDim.z+threadIdx.z;
	if(si < g->dim[0] && sj < g->dim[1] && sk < g->dim[2]){

		dxinv = 1.0E0/g->dx[2];
		unp1 = s_q(1);	//q(i,j,k+1,qw);
		unp2 = s_q(2);	//q(i,j,k+2,qw);
		unp3 = s_q(3);	//q(i,j,k+3,qw);
		unp4 = s_q(4);	//q(i,j,k+4,qw);

		unm1 = s_q(-1);	//q(i,j,k-1,qw);
		unm2 = s_q(-2);	//q(i,j,k-2,qw);
		unm3 = s_q(-3);	//q(i,j,k-3,qw);
		unm4 = s_q(-4);	//q(i,j,k-4,qw);

		flux_irho =  ( g->ALP*(s_cons(1,s_imz)-s_cons(-1,s_imz))
					  + g->BET*(s_cons(2,s_imz)-s_cons(-2,s_imz))
					  + g->GAM*(s_cons(3,s_imz)-s_cons(-3,s_imz))
					  + g->DEL*(s_cons(4,s_imz)-s_cons(-4,s_imz)))*dxinv;

		flux_imx =   ( g->ALP*(s_cons(1,s_imx)*unp1-s_cons(-1,s_imx)*unm1)
					  + g->BET*(s_cons(2,s_imx)*unp2-s_cons(-2,s_imx)*unm2)
					  + g->GAM*(s_cons(3,s_imx)*unp3-s_cons(-3,s_imx)*unm3)
					  + g->DEL*(s_cons(4,s_imx)*unp4-s_cons(-4,s_imx)*unm4))*dxinv;

		flux_imy =   ( g->ALP*(s_cons(1,s_imy)*unp1-s_cons(-1,s_imy)*unm1)
					  + g->BET*(s_cons(2,s_imy)*unp2-s_cons(-2,s_imy)*unm2)
					  + g->GAM*(s_cons(3,s_imy)*unp3-s_cons(-3,s_imy)*unm3)
					  + g->DEL*(s_cons(4,s_imy)*unp4-s_cons(-4,s_imy)*unm4))*dxinv;

		flux_imz =   ( g->ALP*(s_cons(1,s_imz)*unp1-s_cons(-1,s_imz)*unm1
					  + (s_qpres(1)-s_qpres(-1)))
					  + g->BET*(s_cons(2,s_imz)*unp2-s_cons(-2,s_imz)*unm2
					  + (s_qpres(2)-s_qpres(-2)))
					  + g->GAM*(s_cons(3,s_imz)*unp3-s_cons(-3,s_imz)*unm3
					  + (s_qpres(3)-s_qpres(-3)))
					  + g->DEL*(s_cons(4,s_imz)*unp4-s_cons(-4,s_imz)*unm4
					  + (s_qpres(4)-s_qpres(-4))))*dxinv;

		flux_iene   = ( g->ALP*(s_cons(1,s_iene)*unp1-s_cons(-1,s_iene)*unm1
					  + (s_qpres(1)*unp1-s_qpres(-1)*unm1))
					  + g->BET*(s_cons(2,s_iene)*unp2-s_cons(-2,s_iene)*unm2
					  + (s_qpres(2)*unp2-s_qpres(-2)*unm2))
					  + g->GAM*(s_cons(3,s_iene)*unp3-s_cons(-3,s_iene)*unm3
					  + (s_qpres(3)*unp3-s_qpres(-3)*unm3))
					  + g->DEL*(s_cons(4,s_iene)*unp4-s_cons(-4,s_iene)*unm4
					  + (s_qpres(4)*unp4-s_qpres(-4)*unm4)))*dxinv;

		// Update changes
		idx = sk*g->plane_offset_padded + sj*g->pitch[0] + si;

		flux[idx + irho*g->comp_offset_padded] -= flux_irho;
		flux[idx + imx *g->comp_offset_padded] -= flux_imx;
		flux[idx + imy *g->comp_offset_padded] -= flux_imy;
		flux[idx + imz *g->comp_offset_padded] -= flux_imz;
		flux[idx + iene*g->comp_offset_padded] -= flux_iene;
	}
}
#undef	s_q
#undef 	s_qpres
#undef	s_cons


__global__ void gpu_hypterm_xy_stencil_kernel(
	global_const_t *g,	// i:
	double *cons,		// i:
	double *q,			// i:
	double *flux		// o:
){
	int idx,si,sj,sk,tidx,tidy;
	double dxinv, unp1, unp2, unp3, unp4, unm1, unm2, unm3, unm4;
	double flux_irho, flux_imx, flux_imy, flux_imz, flux_iene;

	__shared__ double      s_qu[BLOCK_SMALL+NG+NG][BLOCK_SMALL+NG+NG];
	__shared__ double      s_qv[BLOCK_SMALL+NG+NG][BLOCK_SMALL+NG+NG];
	__shared__ double   s_qpres[BLOCK_SMALL+NG+NG][BLOCK_SMALL+NG+NG];
	__shared__ double s_cons[4][BLOCK_SMALL+NG+NG][BLOCK_SMALL+NG+NG];

	// Load to shared mem
	sk = blockIdx.z*blockDim.z+threadIdx.z;
	for(sj=blockIdx.y*blockDim.y+threadIdx.y, tidy=threadIdx.y; tidy < BLOCK_SMALL+NG+NG; sj+=blockDim.y, tidy+=blockDim.y){
		for(si=blockIdx.x*blockDim.x+threadIdx.x, tidx=threadIdx.x; tidx < BLOCK_SMALL+NG+NG; si+=blockDim.x, tidx+=blockDim.x){
			if( si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){

				idx = (sk+g->ng)*g->plane_offset_g_padded + sj*g->pitch_g[0] + si;

                          s_qu[tidy][tidx]  =     q[idx + qu*g->comp_offset_g_padded];
                          s_qv[tidy][tidx]  =     q[idx + qv*g->comp_offset_g_padded];
					   s_qpres[tidy][tidx]	=     q[idx + qpres*g->comp_offset_g_padded];
				 s_cons[s_imx][tidy][tidx] 	=  cons[idx + imx*g->comp_offset_g_padded];
				 s_cons[s_imy][tidy][tidx] 	=  cons[idx + imy*g->comp_offset_g_padded];
				 s_cons[s_imz][tidy][tidx] 	=  cons[idx + imz*g->comp_offset_g_padded];
				s_cons[s_iene][tidy][tidx] 	=  cons[idx + iene*g->comp_offset_g_padded];
			}
		}
	}
	__syncthreads();

	si = blockIdx.x*blockDim.x+threadIdx.x;
	sj = blockIdx.y*blockDim.y+threadIdx.y;
	if(si < g->dim[0] && sj < g->dim[1] && sk < g->dim[2]){

#define	s_q(i)			s_qu[threadIdx.y+g->ng][threadIdx.x+g->ng+(i)]
#define	s_qpres(i)		s_qpres[threadIdx.y+g->ng][threadIdx.x+g->ng+(i)]
#define	s_cons(i, comp)	s_cons[comp][threadIdx.y+g->ng][threadIdx.x+g->ng+(i)]

		dxinv = 1.0E0/g->dx[0];
		unp1 = s_q(1); //q(i+1,j,k,qu);
		unp2 = s_q(2); //q(i+2,j,k,qu);
		unp3 = s_q(3); //q(i+3,j,k,qu);
		unp4 = s_q(4); //q(i+4,j,k,qu);

		unm1 = s_q(-1); //q(i-1,j,k,qu);
		unm2 = s_q(-2); //q(i-2,j,k,qu);
		unm3 = s_q(-3); //q(i-3,j,k,qu);
		unm4 = s_q(-4); //q(i-4,j,k,qu);

		flux_irho = - ( g->ALP*(s_cons(1, s_imx)-s_cons(-1, s_imx))
					  + g->BET*(s_cons(2, s_imx)-s_cons(-2, s_imx))
					  + g->GAM*(s_cons(3, s_imx)-s_cons(-3, s_imx))
					  + g->DEL*(s_cons(4, s_imx)-s_cons(-4, s_imx)))*dxinv;

		flux_imx  = - ( g->ALP*(s_cons(1, s_imx)*unp1-s_cons(-1, s_imx)*unm1
					  + (s_qpres(1)-s_qpres(-1)))
					  + g->BET*(s_cons(2, s_imx)*unp2-s_cons(-2, s_imx)*unm2
					  + (s_qpres(2)-s_qpres(-2)))
					  + g->GAM*(s_cons(3, s_imx)*unp3-s_cons(-3, s_imx)*unm3
					  + (s_qpres(3)-s_qpres(-3)))
					  + g->DEL*(s_cons(4, s_imx)*unp4-s_cons(-4, s_imx)*unm4
					  + (s_qpres(4)-s_qpres(-4))))*dxinv;

		flux_imy  = - ( g->ALP*(s_cons(1, s_imy)*unp1-s_cons(-1, s_imy)*unm1)
					  + g->BET*(s_cons(2, s_imy)*unp2-s_cons(-2, s_imy)*unm2)
					  + g->GAM*(s_cons(3, s_imy)*unp3-s_cons(-3, s_imy)*unm3)
					  + g->DEL*(s_cons(4, s_imy)*unp4-s_cons(-4, s_imy)*unm4))*dxinv;

		flux_imz  = - ( g->ALP*(s_cons(1, s_imz)*unp1-s_cons(-1, s_imz)*unm1)
					  + g->BET*(s_cons(2, s_imz)*unp2-s_cons(-2, s_imz)*unm2)
					  + g->GAM*(s_cons(3, s_imz)*unp3-s_cons(-3, s_imz)*unm3)
					  + g->DEL*(s_cons(4, s_imz)*unp4-s_cons(-4, s_imz)*unm4))*dxinv;

		flux_iene = - ( g->ALP*(s_cons(1, s_iene)*unp1-s_cons(-1, s_iene)*unm1
					  + (s_qpres(1)*unp1-s_qpres(-1)*unm1))
					  + g->BET*(s_cons(2, s_iene)*unp2-s_cons(-2, s_iene)*unm2
					  + (s_qpres(2)*unp2-s_qpres(-2)*unm2))
					  + g->GAM*(s_cons(3, s_iene)*unp3-s_cons(-3, s_iene)*unm3
					  + (s_qpres(3)*unp3-s_qpres(-3)*unm3))
					  + g->DEL*(s_cons(4, s_iene)*unp4-s_cons(-4, s_iene)*unm4
					  + (s_qpres(4)*unp4-s_qpres(-4)*unm4)))*dxinv;

#undef	s_q
#undef 	s_qpres
#undef	s_cons

#define	s_q(i)			s_qv[threadIdx.y+g->ng+(i)][threadIdx.x+g->ng]
#define	s_qpres(i)		s_qpres[threadIdx.y+g->ng+(i)][threadIdx.x+g->ng]
#define	s_cons(i, comp)	s_cons[comp][threadIdx.y+g->ng+(i)][threadIdx.x+g->ng]

		dxinv = 1.0E0/g->dx[1];
		unp1 = s_q(1); 		//q(i,j+1,k,qv);
		unp2 = s_q(2); 		//q(i,j+2,k,qv);
		unp3 = s_q(3); 		//q(i,j+3,k,qv);
		unp4 = s_q(4); 		//q(i,j+4,k,qv);

		unm1 = s_q(-1); 	//q(i,j-1,k,qv);
		unm2 = s_q(-2); 	//q(i,j-2,k,qv);
		unm3 = s_q(-3); 	//q(i,j-3,k,qv);
		unm4 = s_q(-4); 	//q(i,j-4,k,qv);

		flux_irho -=  ( g->ALP*(s_cons(1, s_imy)-s_cons(-1, s_imy))
					  + g->BET*(s_cons(2, s_imy)-s_cons(-2, s_imy))
					  + g->GAM*(s_cons(3, s_imy)-s_cons(-3, s_imy))
					  + g->DEL*(s_cons(4, s_imy)-s_cons(-4, s_imy)))*dxinv;

		flux_imx -=   ( g->ALP*(s_cons(1, s_imx)*unp1-s_cons(-1, s_imx)*unm1)
					  + g->BET*(s_cons(2, s_imx)*unp2-s_cons(-2, s_imx)*unm2)
					  + g->GAM*(s_cons(3, s_imx)*unp3-s_cons(-3, s_imx)*unm3)
					  + g->DEL*(s_cons(4, s_imx)*unp4-s_cons(-4, s_imx)*unm4))*dxinv;

		flux_imy -=   ( g->ALP*(s_cons(1, s_imy)*unp1-s_cons(-1, s_imy)*unm1
					  + (s_qpres(1)-s_qpres(-1)))
					  + g->BET*(s_cons(2, s_imy)*unp2-s_cons(-2, s_imy)*unm2
					  + (s_qpres(2)-s_qpres(-2)))
					  + g->GAM*(s_cons(3, s_imy)*unp3-s_cons(-3, s_imy)*unm3
					  + (s_qpres(3)-s_qpres(-3)))
					  + g->DEL*(s_cons(4, s_imy)*unp4-s_cons(-4, s_imy)*unm4
					  + (s_qpres(4)-s_qpres(-4))))*dxinv;

		flux_imz -=   ( g->ALP*(s_cons(1, s_imz)*unp1-s_cons(-1, s_imz)*unm1)
					  + g->BET*(s_cons(2, s_imz)*unp2-s_cons(-2, s_imz)*unm2)
					  + g->GAM*(s_cons(3, s_imz)*unp3-s_cons(-3, s_imz)*unm3)
					  + g->DEL*(s_cons(4, s_imz)*unp4-s_cons(-4, s_imz)*unm4))*dxinv;

		flux_iene -=  ( g->ALP*(s_cons(1, s_iene)*unp1-s_cons(-1, s_iene)*unm1
					  + (s_qpres(1)*unp1-s_qpres(-1)*unm1))
					  + g->BET*(s_cons(2, s_iene)*unp2-s_cons(-2, s_iene)*unm2
					  + (s_qpres(2)*unp2-s_qpres(-2)*unm2))
					  + g->GAM*(s_cons(3, s_iene)*unp3-s_cons(-3, s_iene)*unm3
					  + (s_qpres(3)*unp3-s_qpres(-3)*unm3))
					  + g->DEL*(s_cons(4, s_iene)*unp4-s_cons(-4, s_iene)*unm4
					  + (s_qpres(4)*unp4-s_qpres(-4)*unm4)))*dxinv;

#undef	s_q
#undef 	s_qpres
#undef	s_cons

		// Update changes
		idx = sk*g->plane_offset_padded + sj*g->pitch[0] + si;

		flux[idx + irho*g->comp_offset_padded] = flux_irho;
		flux[idx + imx *g->comp_offset_padded] = flux_imx;
		flux[idx + imy *g->comp_offset_padded] = flux_imy;
		flux[idx + imz *g->comp_offset_padded] = flux_imz;
		flux[idx + iene*g->comp_offset_padded] = flux_iene;

	}
}

__global__ void gpu_hypterm_y_stencil_kernel2(
	global_const_t *g,	// i:
	double *cons,		// i:
	double *q,			// i:
	double *flux		// o:
){
	int idx,si,sj,sk,tidx,tidy;
	double dxinv, unp1, unp2, unp3, unp4, unm1, unm2, unm3, unm4;
	double flux_irho, flux_imx, flux_imy, flux_imz, flux_iene;

	__shared__ double       s_q[BLOCK_SMALL+NG+NG][BLOCK_SMALL+NG+NG];
	__shared__ double   s_qpres[BLOCK_SMALL+NG+NG][BLOCK_SMALL+NG+NG];
	__shared__ double s_cons[4][BLOCK_SMALL+NG+NG][BLOCK_SMALL+NG+NG];

	// Load to shared mem
	sk = blockIdx.z*blockDim.z+threadIdx.z;
	for(sj=blockIdx.y*blockDim.y+threadIdx.y, tidy=threadIdx.y; tidy < BLOCK_SMALL+NG+NG; sj+=blockDim.y, tidy+=blockDim.y){
		for(si=blockIdx.x*blockDim.x+threadIdx.x, tidx=threadIdx.x; tidx < BLOCK_SMALL+NG+NG; si+=blockDim.x, tidx+=blockDim.x){
			if( si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){

                idx = (sk+g->ng)*g->plane_offset_g_padded + sj*g->pitch_g[0] + (si+g->ng);

                           s_q[tidy][tidx]  =     q[idx + qv*g->comp_offset_g_padded];
                       s_qpres[tidy][tidx]	=     q[idx + qpres*g->comp_offset_g_padded];
                 s_cons[s_imx][tidy][tidx] 	=  cons[idx + imx*g->comp_offset_g_padded];
                 s_cons[s_imy][tidy][tidx] 	=  cons[idx + imy*g->comp_offset_g_padded];
                 s_cons[s_imz][tidy][tidx] 	=  cons[idx + imz*g->comp_offset_g_padded];
                s_cons[s_iene][tidy][tidx] 	=  cons[idx + iene*g->comp_offset_g_padded];

			}
		}
	}
	__syncthreads();

    si = blockIdx.x*blockDim.x+threadIdx.x;
	sj = blockIdx.y*blockDim.y+threadIdx.y;
	if(si < g->dim[0] && sj < g->dim[1] && sk < g->dim[2]){

#define	s_q(i)			s_q[threadIdx.y+g->ng+(i)][threadIdx.x]
#define	s_qpres(i)		s_qpres[threadIdx.y+g->ng+(i)][threadIdx.x]
#define	s_cons(i, comp)	s_cons[comp][threadIdx.y+g->ng+(i)][threadIdx.x]

		dxinv = 1.0E0/g->dx[1];
		unp1 = s_q(1); 		//q(i,j+1,k,qv);
		unp2 = s_q(2); 		//q(i,j+2,k,qv);
		unp3 = s_q(3); 		//q(i,j+3,k,qv);
		unp4 = s_q(4); 		//q(i,j+4,k,qv);

		unm1 = s_q(-1); 	//q(i,j-1,k,qv);
		unm2 = s_q(-2); 	//q(i,j-2,k,qv);
		unm3 = s_q(-3); 	//q(i,j-3,k,qv);
		unm4 = s_q(-4); 	//q(i,j-4,k,qv);

		flux_irho =  ( g->ALP*(s_cons(1, s_imy)-s_cons(-1, s_imy))
					  + g->BET*(s_cons(2, s_imy)-s_cons(-2, s_imy))
					  + g->GAM*(s_cons(3, s_imy)-s_cons(-3, s_imy))
					  + g->DEL*(s_cons(4, s_imy)-s_cons(-4, s_imy)))*dxinv;

		flux_imx =   ( g->ALP*(s_cons(1, s_imx)*unp1-s_cons(-1, s_imx)*unm1)
					  + g->BET*(s_cons(2, s_imx)*unp2-s_cons(-2, s_imx)*unm2)
					  + g->GAM*(s_cons(3, s_imx)*unp3-s_cons(-3, s_imx)*unm3)
					  + g->DEL*(s_cons(4, s_imx)*unp4-s_cons(-4, s_imx)*unm4))*dxinv;

		flux_imy =   ( g->ALP*(s_cons(1, s_imy)*unp1-s_cons(-1, s_imy)*unm1
					  + (s_qpres(1)-s_qpres(-1)))
					  + g->BET*(s_cons(2, s_imy)*unp2-s_cons(-2, s_imy)*unm2
					  + (s_qpres(2)-s_qpres(-2)))
					  + g->GAM*(s_cons(3, s_imy)*unp3-s_cons(-3, s_imy)*unm3
					  + (s_qpres(3)-s_qpres(-3)))
					  + g->DEL*(s_cons(4, s_imy)*unp4-s_cons(-4, s_imy)*unm4
					  + (s_qpres(4)-s_qpres(-4))))*dxinv;

		flux_imz =   ( g->ALP*(s_cons(1, s_imz)*unp1-s_cons(-1, s_imz)*unm1)
					  + g->BET*(s_cons(2, s_imz)*unp2-s_cons(-2, s_imz)*unm2)
					  + g->GAM*(s_cons(3, s_imz)*unp3-s_cons(-3, s_imz)*unm3)
					  + g->DEL*(s_cons(4, s_imz)*unp4-s_cons(-4, s_imz)*unm4))*dxinv;

		flux_iene =  ( g->ALP*(s_cons(1, s_iene)*unp1-s_cons(-1, s_iene)*unm1
					  + (s_qpres(1)*unp1-s_qpres(-1)*unm1))
					  + g->BET*(s_cons(2, s_iene)*unp2-s_cons(-2, s_iene)*unm2
					  + (s_qpres(2)*unp2-s_qpres(-2)*unm2))
					  + g->GAM*(s_cons(3, s_iene)*unp3-s_cons(-3, s_iene)*unm3
					  + (s_qpres(3)*unp3-s_qpres(-3)*unm3))
					  + g->DEL*(s_cons(4, s_iene)*unp4-s_cons(-4, s_iene)*unm4
					  + (s_qpres(4)*unp4-s_qpres(-4)*unm4)))*dxinv;

#undef	s_q
#undef 	s_qpres
#undef	s_cons

		// Update changes
		idx = sk*g->plane_offset_padded + sj*g->pitch[0] + si;

		flux[idx + irho*g->comp_offset_padded] -= flux_irho;
		flux[idx + imx *g->comp_offset_padded] -= flux_imx;
		flux[idx + imy *g->comp_offset_padded] -= flux_imy;
		flux[idx + imz *g->comp_offset_padded] -= flux_imz;
		flux[idx + iene*g->comp_offset_padded] -= flux_iene;
	}
}

void gpu_hypterm2(
	global_const_t h_const, 	// i: Global struct containing application parameters
	global_const_t *d_const,	// i: Device pointer to global struct containing application paramters
	double *d_cons,				// i:
	double *d_q,				// i:
	double *d_flux				// o: flux
){

	/** d_flux must be set to zero beforehand (in diffterm, etc) **/

	dim3 block_dim_xy_stencil(BLOCK_SMALL, BLOCK_SMALL, 1);
	dim3 grid_dim_xy_stencil(CEIL(h_const.dim[0], BLOCK_SMALL), CEIL(h_const.dim[1], BLOCK_SMALL), h_const.dim[2]);
	gpu_hypterm_xy_stencil_kernel<<<grid_dim_xy_stencil, block_dim_xy_stencil>>>(d_const, d_cons, d_q, d_flux);

	dim3 block_dim_z_stencil(BLOCK_LARGE, 1, BLOCK_SMALL);
	dim3 grid_dim_z_stencil(CEIL(h_const.dim[0], BLOCK_LARGE), h_const.dim[1], CEIL(h_const.dim[2], BLOCK_SMALL));
	gpu_hypterm_z_stencil_kernel<<<grid_dim_z_stencil, block_dim_z_stencil>>>(d_const, d_cons, d_q, d_flux);

}

