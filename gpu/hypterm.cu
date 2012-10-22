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

		idx = si*g->plane_offset_g + (sj+g->ng)*g->dim_g[2] + (sk+g->ng);

				   s_q[tidx][tidz]  =     q[idx + qu*g->comp_offset_g];
			   s_qpres[tidx][tidz]	=     q[idx + qpres*g->comp_offset_g];
		 s_cons[s_imx][tidx][tidz] 	=  cons[idx + imx*g->comp_offset_g];
		 s_cons[s_imy][tidx][tidz] 	=  cons[idx + imy*g->comp_offset_g];
		 s_cons[s_imz][tidx][tidz] 	=  cons[idx + imz*g->comp_offset_g];
		s_cons[s_iene][tidx][tidz] 	=  cons[idx + iene*g->comp_offset_g];

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
		idx = si*g->plane_offset + sj*g->dim[2] + sk;

		flux[idx + irho*g->comp_offset] = flux_irho;
		flux[idx + imx *g->comp_offset] = flux_imx;
		flux[idx + imy *g->comp_offset] = flux_imy;
		flux[idx + imz *g->comp_offset] = flux_imz;
		flux[idx + iene*g->comp_offset] = flux_iene;
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

		idx = (si+g->ng)*g->plane_offset_g + sj*g->dim_g[2] + (sk+g->ng);

				   s_q[tidy][tidz]  =     q[idx + qv*g->comp_offset_g];
			   s_qpres[tidy][tidz]	=     q[idx + qpres*g->comp_offset_g];
		 s_cons[s_imx][tidy][tidz] 	=  cons[idx + imx*g->comp_offset_g];
		 s_cons[s_imy][tidy][tidz] 	=  cons[idx + imy*g->comp_offset_g];
		 s_cons[s_imz][tidy][tidz] 	=  cons[idx + imz*g->comp_offset_g];
		s_cons[s_iene][tidy][tidz] 	=  cons[idx + iene*g->comp_offset_g];

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
		idx = si*g->plane_offset + sj*g->dim[2] + sk;

		flux[idx + irho*g->comp_offset] -= flux_irho;
		flux[idx + imx *g->comp_offset] -= flux_imx;
		flux[idx + imy *g->comp_offset] -= flux_imy;
		flux[idx + imz *g->comp_offset] -= flux_imz;
		flux[idx + iene*g->comp_offset] -= flux_iene;
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

		idx = (si+g->ng)*g->plane_offset_g + (sj+g->ng)*g->dim_g[2] + sk;

				   s_q[tidy][tidz]  =     q[idx + qw*g->comp_offset_g];
			   s_qpres[tidy][tidz]	=     q[idx + qpres*g->comp_offset_g];
		 s_cons[s_imx][tidy][tidz] 	=  cons[idx + imx*g->comp_offset_g];
		 s_cons[s_imy][tidy][tidz] 	=  cons[idx + imy*g->comp_offset_g];
		 s_cons[s_imz][tidy][tidz] 	=  cons[idx + imz*g->comp_offset_g];
		s_cons[s_iene][tidy][tidz] 	=  cons[idx + iene*g->comp_offset_g];

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
		idx = si*g->plane_offset + sj*g->dim[2] + sk;

		flux[idx + irho*g->comp_offset] -= flux_irho;
		flux[idx + imx *g->comp_offset] -= flux_imx;
		flux[idx + imy *g->comp_offset] -= flux_imy;
		flux[idx + imz *g->comp_offset] -= flux_imz;
		flux[idx + iene*g->comp_offset] -= flux_iene;
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

void hypterm_test(
	global_const_t h_const, // i: Global struct containing application parameters
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
