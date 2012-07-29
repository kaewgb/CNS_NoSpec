#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "helper_functions.h"

#define BLOCK_DIM		16		// Dimension that doesn't have ghost cells
#define	BLOCK_DIM_G		8		// Dimension that has ghost cells

__global__ void gpu_hypterm_merged_kernel(
	global_const_t *g,	// i:
	double *cons,		// i:
	double *q,			// i:
	double *flux		// o:
){
	int idx,bi,bj,bk;
	int si,sj,sk,tidx,tidy,tidz;
	double dxinv, unp1, unp2, unp3, unp4, unm1, unm2, unm3, unm4;
	double flux_irho, flux_imx, flux_imy, flux_imz, flux_iene;

	__shared__ double       s_q[BLOCK_DIM_G+NG+NG][BLOCK_DIM];
	__shared__ double   s_qpres[BLOCK_DIM_G+NG+NG][BLOCK_DIM];
	__shared__ double s_cons[4][BLOCK_DIM_G+NG+NG][BLOCK_DIM];

	// Load to shared mem
	bi = (blockIdx.x % (g->gridDim_plane_xy)) / g->gridDim_y;
	bj = (blockIdx.x % (g->gridDim_plane_xy)) % g->gridDim_y;
	bk =  blockIdx.x / (g->gridDim_plane_xy);
	si = bi*blockDim.x+threadIdx.x;
	sj = bj*blockDim.y+threadIdx.y;
	sk = bk*blockDim.z+threadIdx.z;

	tidx = threadIdx.x;
	tidy = threadIdx.y;
	while( tidx < g->blockDim_x_g && si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){

		idx = si*g->plane_offset_g + (sk+g->ng)*g->dim_g[2] + (sj+g->ng);

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

#define	s_q(i)			s_q[threadIdx.x+g->ng+i][threadIdx.y]
#define	s_qpres(i)		s_qpres[threadIdx.x+g->ng+i][threadIdx.y]
#define	s_cons(i, comp)	s_cons[comp][threadIdx.x+g->ng+i][threadIdx.y]

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

		flux_irho = - ( ALP*(s_cons(1, s_imx)-s_cons(-1, s_imx))
					  + BET*(s_cons(2, s_imx)-s_cons(-2, s_imx))
					  + GAM*(s_cons(3, s_imx)-s_cons(-3, s_imx))
					  + DEL*(s_cons(4, s_imx)-s_cons(-4, s_imx)))*dxinv;

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
	}

	__syncthreads();
	si = bi*blockDim.x+threadIdx.x;
	tidx = threadIdx.x;
	tidy = threadIdx.y;
	while( tidx < g->blockDim_x_g && si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){

		idx = (sk+g->ng)*g->plane_offset_g + si*g->dim_g[2] + (sj+g->ng);

				   s_q[tidx][tidz]  =     q[idx + qv*g->comp_offset_g];
			   s_qpres[tidx][tidz]	=     q[idx + qpres*g->comp_offset_g];
		 s_cons[s_imx][tidx][tidz] 	=  cons[idx + imx*g->comp_offset_g];
		 s_cons[s_imy][tidx][tidz] 	=  cons[idx + imy*g->comp_offset_g];
		 s_cons[s_imz][tidx][tidz] 	=  cons[idx + imz*g->comp_offset_g];
		s_cons[s_iene][tidx][tidz] 	=  cons[idx + iene*g->comp_offset_g];

		tidx += blockDim.x;
		si   += blockDim.x;
	}
	__syncthreads();

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

		flux_irho -=  ( ALP*(s_cons(1, s_imy)-s_cons(-1, s_imy))
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
	}

	__syncthreads();
	si = bi*blockDim.x+threadIdx.x;
	tidx = threadIdx.x;
	tidy = threadIdx.y;
	while( tidx < g->blockDim_x_g && si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){

		idx = (sk+g->ng)*g->plane_offset_g + sj*g->dim_g[2] + (si+g->ng);

				   s_q[tidx][tidz]  =     q[idx + qw*g->comp_offset_g];
			   s_qpres[tidx][tidz]	=     q[idx + qpres*g->comp_offset_g];
		 s_cons[s_imx][tidx][tidz] 	=  cons[idx + imx*g->comp_offset_g];
		 s_cons[s_imy][tidx][tidz] 	=  cons[idx + imy*g->comp_offset_g];
		 s_cons[s_imz][tidx][tidz] 	=  cons[idx + imz*g->comp_offset_g];
		s_cons[s_iene][tidx][tidz] 	=  cons[idx + iene*g->comp_offset_g];

		tidx += blockDim.x;
		si   += blockDim.x;
	}
	__syncthreads();
}
#undef	s_q
#undef 	s_qpres
#undef	s_cons

void gpu_hypterm_merged(
	global_const_t h_const, 	// i: Global struct containing applicatino parameters
	global_const_t *d_const,	// i: Device pointer to global struct containing application paramters
	double *d_cons,				// i:
	double *d_q,				// i:
	double *d_flux				// o: flux
){
	int i, len, dim[3];
	int grid_dim, grid_dim_x, grid_dim_y, grid_dim_z;

	grid_dim_x = CEIL(h_const.dim[0], BLOCK_DIM_G);
	grid_dim_y = CEIL(h_const.dim[1], BLOCK_DIM);
	grid_dim_z = h_const.dim[2];
	grid_dim = grid_dim_x * grid_dim_y * grid_dim_z;

	dim3 block_dim(BLOCK_DIM_G, BLOCK_DIM, 1);
    h_const.gridDim_x = grid_dim_x;
    h_const.gridDim_y = grid_dim_y;
    h_const.gridDim_z = grid_dim_z;
    h_const.gridDim_plane_xy = grid_dim_x * grid_dim_y;
    h_const.blockDim_x_g = BLOCK_DIM_G + h_const.ng + h_const.ng;
    cudaMemcpy(d_const, &h_const, sizeof(global_const_t), cudaMemcpyHostToDevice);

	gpu_hypterm_merged_kernel<<<grid_dim, block_dim>>>(d_const, d_cons, d_q, d_flux);

}
