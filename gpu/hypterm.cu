#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "util.h"

#define BLOCK_DIM		16		// Dimension that doesn't have ghost cells
#define	BLOCK_DIM_G		8		// Dimension that has ghost cells
#define	s_q(i)			s_q[threadIdx.x+g->ng+(i)][threadIdx.z]
#define	s_qpres(i)		s_qpres[threadIdx.x+g->ng+(i)][threadIdx.z]
#define	s_cons(i, comp)	s_cons[comp][threadIdx.x+g->ng+(i)][threadIdx.z]

__global__ void gpu_hypterm_x_stencil_kernel(
	global_const_t *g,	// i:
	double *cons,		// i:
	double *q,			// i:
	double *flux		// o:
){
	int idx,bi,bj,bk;
	int si,sj,sk,tidx,tidz;
	kernel_const_t *kc = g->kc;
	double dxinv, unp1, unp2, unp3, unp4, unm1, unm2, unm3, unm4;
	double flux_irho, flux_imx, flux_imy, flux_imz, flux_iene;

	__shared__ double       s_q[BLOCK_DIM_G+NG+NG][BLOCK_DIM];
	__shared__ double   s_qpres[BLOCK_DIM_G+NG+NG][BLOCK_DIM];
	__shared__ double s_cons[4][BLOCK_DIM_G+NG+NG][BLOCK_DIM];

	// Load to shared mem
	bi = (blockIdx.x % (kc->gridDim_plane_xz)) / kc->gridDim_z;
	bk = (blockIdx.x % (kc->gridDim_plane_xz)) % kc->gridDim_z;
	bj =  blockIdx.x / (kc->gridDim_plane_xz);
	si = bi*blockDim.x+threadIdx.x;
	sj = bj*blockDim.y+threadIdx.y; // = bj
	sk = bk*blockDim.z+threadIdx.z;

	tidx = threadIdx.x;
	tidz = threadIdx.z;
	while( tidx < kc->blockDim_x_g && si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){

		idx = (sk+g->ng)*g->plane_offset_g_padded + (sj+g->ng)*g->dim_g_padded[0] + si;

				   s_q[tidx][tidz]  =     q[idx + qu*g->comp_offset_g_padded];
			   s_qpres[tidx][tidz]	=     q[idx + qpres*g->comp_offset_g_padded];
		 s_cons[s_imx][tidx][tidz] 	=  cons[idx + imx*g->comp_offset_g_padded];
		 s_cons[s_imy][tidx][tidz] 	=  cons[idx + imy*g->comp_offset_g_padded];
		 s_cons[s_imz][tidx][tidz] 	=  cons[idx + imz*g->comp_offset_g_padded];
		s_cons[s_iene][tidx][tidz] 	=  cons[idx + iene*g->comp_offset_g_padded];

		tidx += blockDim.x;
		si   += blockDim.x;
	}
	__syncthreads();

	si = bi*blockDim.x+threadIdx.x;
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
		idx = sk*g->plane_offset_padded + sj*g->dim_padded[0] + si;

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

#define	s_q(i)			s_q[threadIdx.y+g->ng+(i)][threadIdx.z]
#define	s_qpres(i)		s_qpres[threadIdx.y+g->ng+(i)][threadIdx.z]
#define	s_cons(i, comp)	s_cons[comp][threadIdx.y+g->ng+(i)][threadIdx.z]

__global__ void gpu_hypterm_y_stencil_kernel(
	global_const_t *g,	// i:
	double *cons,		// i:
	double *q,			// i:
	double *flux		// o:
){
	int idx,bi,bj,bk;
	int si,sj,sk,tidy,tidz;
	kernel_const_t *kc = g->kc;
	double dxinv, unp1, unp2, unp3, unp4, unm1, unm2, unm3, unm4;
	double flux_irho, flux_imx, flux_imy, flux_imz, flux_iene;

	__shared__ double       s_q[BLOCK_DIM_G+NG+NG][BLOCK_DIM];
	__shared__ double   s_qpres[BLOCK_DIM_G+NG+NG][BLOCK_DIM];
	__shared__ double s_cons[4][BLOCK_DIM_G+NG+NG][BLOCK_DIM];

	// Load to shared mem
	bj = (blockIdx.x % (kc->gridDim_plane_yz)) / kc->gridDim_z;
	bk = (blockIdx.x % (kc->gridDim_plane_yz)) % kc->gridDim_z;
	bi =  blockIdx.x / (kc->gridDim_plane_yz);
	si = bi*blockDim.x+threadIdx.x;
	sj = bj*blockDim.y+threadIdx.y;
	sk = bk*blockDim.z+threadIdx.z;

	tidy = threadIdx.y;
	tidz = threadIdx.z;
	while( tidy < kc->blockDim_y_g && si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){

		idx = (sk+g->ng)*g->plane_offset_g_padded + sj*g->dim_g_padded[0] + (si+g->ng);

				   s_q[tidy][tidz]  =     q[idx + qv*g->comp_offset_g_padded];
			   s_qpres[tidy][tidz]	=     q[idx + qpres*g->comp_offset_g_padded];
		 s_cons[s_imx][tidy][tidz] 	=  cons[idx + imx*g->comp_offset_g_padded];
		 s_cons[s_imy][tidy][tidz] 	=  cons[idx + imy*g->comp_offset_g_padded];
		 s_cons[s_imz][tidy][tidz] 	=  cons[idx + imz*g->comp_offset_g_padded];
		s_cons[s_iene][tidy][tidz] 	=  cons[idx + iene*g->comp_offset_g_padded];

		tidy += blockDim.y;
		sj   += blockDim.y;
	}
	__syncthreads();

	sj = bj*blockDim.y+threadIdx.y;
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
		idx = sk*g->plane_offset_padded + sj*g->dim_padded[2] + si;

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

#define	s_q(i)			s_q[threadIdx.y][threadIdx.z+g->ng+(i)]
#define	s_qpres(i)		s_qpres[threadIdx.y][threadIdx.z+g->ng+(i)]
#define	s_cons(i, comp)	s_cons[comp][threadIdx.y][threadIdx.z+g->ng+(i)]

__global__ void gpu_hypterm_z_stencil_kernel(
	global_const_t *g,	// i:
	double *cons,		// i:
	double *q,			// i:
	double *flux		// o:
){
	int idx,bi,bj,bk;
	int si,sj,sk,tidy,tidz;
	kernel_const_t *kc = g->kc;
	double dxinv, unp1, unp2, unp3, unp4, unm1, unm2, unm3, unm4;
	double flux_irho, flux_imx, flux_imy, flux_imz, flux_iene;

	__shared__ double       s_q[BLOCK_DIM][BLOCK_DIM_G+NG+NG];
	__shared__ double   s_qpres[BLOCK_DIM][BLOCK_DIM_G+NG+NG];
	__shared__ double s_cons[4][BLOCK_DIM][BLOCK_DIM_G+NG+NG];

	// Load to shared mem
	bj = (blockIdx.x % (kc->gridDim_plane_yz)) / kc->gridDim_z;
	bk = (blockIdx.x % (kc->gridDim_plane_yz)) % kc->gridDim_z;
	bi =  blockIdx.x / (kc->gridDim_plane_yz);
	si = bi*blockDim.x+threadIdx.x;
	sj = bj*blockDim.y+threadIdx.y;
	sk = bk*blockDim.z+threadIdx.z;

	tidy = threadIdx.y;
	tidz = threadIdx.z;
	while( tidz < kc->blockDim_z_g && si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){

		idx = sk*g->plane_offset_g_padded + (sj+g->ng)*g->dim_g_padded[0] + (si+g->ng);

				   s_q[tidy][tidz]  =     q[idx + qw*g->comp_offset_g_padded];
			   s_qpres[tidy][tidz]	=     q[idx + qpres*g->comp_offset_g_padded];
		 s_cons[s_imx][tidy][tidz] 	=  cons[idx + imx*g->comp_offset_g_padded];
		 s_cons[s_imy][tidy][tidz] 	=  cons[idx + imy*g->comp_offset_g_padded];
		 s_cons[s_imz][tidy][tidz] 	=  cons[idx + imz*g->comp_offset_g_padded];
		s_cons[s_iene][tidy][tidz] 	=  cons[idx + iene*g->comp_offset_g_padded];

		tidz += blockDim.z;
		sk   += blockDim.z;
	}
	__syncthreads();

	sk = bk*blockDim.z+threadIdx.z;
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
		idx = sk*g->plane_offset_padded + sj*g->dim_padded[0] + si;

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

void gpu_hypterm(
	global_const_t h_const, 	// i: Global struct containing application parameters
	global_const_t *d_const,	// i: Device pointer to global struct containing application paramters
	double *d_cons,				// i:
	double *d_q,				// i:
	double *d_flux				// o: flux
){
	int i, grid_dim;
	kernel_const_t h_kc;

	dim3 block_dim_x_stencil(BLOCK_DIM_G, 1, BLOCK_DIM);
    h_kc.gridDim_x = CEIL(h_const.dim[0], BLOCK_DIM_G);
    h_kc.gridDim_y = h_const.dim[1];
    h_kc.gridDim_z = CEIL(h_const.dim[2], BLOCK_DIM);
    h_kc.gridDim_plane_xz = h_kc.gridDim_x * h_kc.gridDim_z;
    h_kc.blockDim_x_g = BLOCK_DIM_G + h_const.ng + h_const.ng;
    grid_dim = h_kc.gridDim_plane_xz * h_kc.gridDim_y;
    cudaMemcpy(h_const.kc, &h_kc, sizeof(kernel_const_t), cudaMemcpyHostToDevice);

	gpu_hypterm_x_stencil_kernel<<<grid_dim, block_dim_x_stencil>>>(d_const, d_cons, d_q, d_flux);

	dim3 block_dim_y_stencil(1, BLOCK_DIM_G, BLOCK_DIM);
	h_kc.gridDim_x = h_const.dim[0];
	h_kc.gridDim_y = CEIL(h_const.dim[1], BLOCK_DIM_G);
	h_kc.gridDim_z = CEIL(h_const.dim[2], BLOCK_DIM);
	h_kc.gridDim_plane_yz = h_kc.gridDim_y * h_kc.gridDim_z;
	h_kc.blockDim_y_g = BLOCK_DIM_G + h_const.ng + h_const.ng;
	grid_dim = h_kc.gridDim_plane_yz * h_kc.gridDim_x;
	cudaMemcpy(h_const.kc, &h_kc, sizeof(kernel_const_t), cudaMemcpyHostToDevice);

	gpu_hypterm_y_stencil_kernel<<<grid_dim, block_dim_y_stencil>>>(d_const, d_cons, d_q, d_flux);

	dim3 block_dim_z_stencil(1, BLOCK_DIM, BLOCK_DIM_G);
	h_kc.gridDim_x = h_const.dim[0];
	h_kc.gridDim_y = CEIL(h_const.dim[1], BLOCK_DIM);
	h_kc.gridDim_z = CEIL(h_const.dim[2], BLOCK_DIM_G);
	h_kc.gridDim_plane_yz = h_kc.gridDim_y * h_kc.gridDim_z;
	h_kc.blockDim_z_g = BLOCK_DIM_G + h_const.ng + h_const.ng;
	grid_dim = h_kc.gridDim_plane_yz * h_kc.gridDim_x;
	cudaMemcpy(h_const.kc, &h_kc, sizeof(kernel_const_t), cudaMemcpyHostToDevice);

	gpu_hypterm_z_stencil_kernel<<<grid_dim, block_dim_z_stencil>>>(d_const, d_cons, d_q, d_flux);

}
