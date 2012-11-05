#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "util.h"

#define BLOCK_DIM	16

__global__ void gpu_diffterm_lv1_kernel(
	global_const_t *g,			// i: Global struct containing application parameters
	double *q,					// i:
	double *difflux				// o:
){
	int si, sj, sk;
	int idx, tidx, tidy, tidz;
	kernel_const_t *kc = g->kc;
	__shared__ double s_q[s_qend][BLOCK_DIM+NG+NG][BLOCK_DIM+NG+NG];

	/*** XZ ***/
	si = blockIdx.x*blockDim.x + threadIdx.x;
	sk = blockIdx.y*blockDim.y + threadIdx.y;
	sj = blockIdx.z;

	tidx = threadIdx.x;
	tidz = threadIdx.y;
	while(tidz < blockDim.y+NG+NG && si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){
		idx = sk*g->plane_offset_g_padded + sj*g->pitch_g[0] + si;

		s_q[s_qu][tidz][tidx] = q[idx + qu*g->comp_offset_g_padded];
		s_q[s_qv][tidz][tidx] = q[idx + qv*g->comp_offset_g_padded];
		s_q[s_qw][tidz][tidx] = q[idx + qw*g->comp_offset_g_padded];
		s_q[s_qt][tidz][tidx] = q[idx + qt*g->comp_offset_g_padded];

		tidz += blockDim.y;
		sk	 += blockDim.y;
	}
	__syncthreads();

#define	q(i, comp)	s_q[comp][threadIdx.y+g->ng+(i)][threadIdx.x]

	sk = blockIdx.y*blockDim.y + threadIdx.y;
	idx = (sk+g->ng)*g->plane_offset_g_padded + sj*g->pitch_g[0] + si;
	if(si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim[2]){

		g->temp[UZ][idx] =  ( g->ALP*(q(1,s_qu)-q(-1,s_qu))
							+ g->BET*(q(2,s_qu)-q(-2,s_qu))
							+ g->GAM*(q(3,s_qu)-q(-3,s_qu))
							+ g->DEL*(q(4,s_qu)-q(-4,s_qu)))*g->dxinv[2];

		g->temp[VZ][idx] = 	( g->ALP*(q(1,s_qv)-q(-1,s_qv))
							+ g->BET*(q(2,s_qv)-q(-2,s_qv))
							+ g->GAM*(q(3,s_qv)-q(-3,s_qv))
							+ g->DEL*(q(4,s_qv)-q(-4,s_qv)))*g->dxinv[2];

		g->temp[WZ][idx] =	( g->ALP*(q(1,s_qw)-q(-1,s_qw))
							+ g->BET*(q(2,s_qw)-q(-2,s_qw))
							+ g->GAM*(q(3,s_qw)-q(-3,s_qw))
							+ g->DEL*(q(4,s_qw)-q(-4,s_qw)))*g->dxinv[2];

		g->temp[UZZ][idx] = ( g->CENTER*q(0,s_qu)
							+ g->OFF1*(q(1,s_qu)+q(-1,s_qu))
							+ g->OFF2*(q(2,s_qu)+q(-2,s_qu))
							+ g->OFF3*(q(3,s_qu)+q(-3,s_qu))
							+ g->OFF4*(q(4,s_qu)+q(-4,s_qu)))*SQR(g->dxinv[2]);

		g->temp[VZZ][idx] = ( g->CENTER*q(0,s_qv)
							+ g->OFF1*(q(1,s_qv)+q(-1,s_qv))
							+ g->OFF2*(q(2,s_qv)+q(-2,s_qv))
							+ g->OFF3*(q(3,s_qv)+q(-3,s_qv))
							+ g->OFF4*(q(4,s_qv)+q(-4,s_qv)))*SQR(g->dxinv[2]);

		g->temp[WZZ][idx] = ( g->CENTER*q(0,s_qw)
							+ g->OFF1*(q(1,s_qw)+q(-1,s_qw))
							+ g->OFF2*(q(2,s_qw)+q(-2,s_qw))
							+ g->OFF3*(q(3,s_qw)+q(-3,s_qw))
							+ g->OFF4*(q(4,s_qw)+q(-4,s_qw)))*SQR(g->dxinv[2]);
	}

	idx = sk*g->plane_offset_padded + (sj-g->ng)*g->pitch[0] + (si-g->ng);
	if( g->ng <= si && si < g->dim[0] + g->ng &&
		g->ng <= sj && sj < g->dim[1] + g->ng &&
					   sk < g->dim[2] ){

		g->temp[TZZ][idx] = ( g->CENTER*q(0,s_qt)
							+ g->OFF1*(q(1,s_qt)+q(-1,s_qt))
							+ g->OFF2*(q(2,s_qt)+q(-2,s_qt))
							+ g->OFF3*(q(3,s_qt)+q(-3,s_qt))
							+ g->OFF4*(q(4,s_qt)+q(-4,s_qt)))*SQR(g->dxinv[2]);
	}
#undef 	q

	/*** XY ***/
	si = blockIdx.x*blockDim.x + threadIdx.x;
	sj = blockIdx.y*blockDim.y + threadIdx.y;
	sk = blockIdx.z;

	__syncthreads();
	for(sj = blockIdx.y*blockDim.y+threadIdx.y, tidy=threadIdx.y; tidy<blockDim.y+NG+NG; sj+=blockDim.y, tidy+=blockDim.y){
		for(si = blockIdx.x*blockDim.x+threadIdx.x, tidx=threadIdx.x; tidx<blockDim.x+NG+NG; si+=blockDim.x, tidx+=blockDim.x){
			if(si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){
				idx = sk*g->plane_offset_g_padded + sj*g->pitch_g[0] + si;

				s_q[s_qu][tidy][tidx] = q[idx + qu*g->comp_offset_g_padded];
				s_q[s_qv][tidy][tidx] = q[idx + qv*g->comp_offset_g_padded];
				s_q[s_qw][tidy][tidx] = q[idx + qw*g->comp_offset_g_padded];
				s_q[s_qt][tidy][tidx] = q[idx + qt*g->comp_offset_g_padded];

			}
		}
	}
	__syncthreads();

#define	q(i, comp)	s_q[comp][threadIdx.y+g->ng+(i)][threadIdx.x]
	si = blockIdx.x*blockDim.x + threadIdx.x;
	sj = blockIdx.y*blockDim.y + threadIdx.y;
	idx = sk*g->plane_offset_g_padded + (sj+g->ng)*g->pitch_g[0] + si;
	if(si < g->dim_g[0] && sj < g->dim[1] && sk < g->dim_g[2]){

		g->temp[UY][idx] =  ( g->ALP*(q(1,s_qu)-q(-1,s_qu))
							+ g->BET*(q(2,s_qu)-q(-2,s_qu))
							+ g->GAM*(q(3,s_qu)-q(-3,s_qu))
							+ g->DEL*(q(4,s_qu)-q(-4,s_qu)))*g->dxinv[1];

		g->temp[VY][idx] = 	( g->ALP*(q(1,s_qv)-q(-1,s_qv))
							+ g->BET*(q(2,s_qv)-q(-2,s_qv))
							+ g->GAM*(q(3,s_qv)-q(-3,s_qv))
							+ g->DEL*(q(4,s_qv)-q(-4,s_qv)))*g->dxinv[1];

		g->temp[WY][idx] =	( g->ALP*(q(1,s_qw)-q(-1,s_qw))
							+ g->BET*(q(2,s_qw)-q(-2,s_qw))
							+ g->GAM*(q(3,s_qw)-q(-3,s_qw))
							+ g->DEL*(q(4,s_qw)-q(-4,s_qw)))*g->dxinv[1];

		g->temp[UYY][idx] = ( g->CENTER*q(0,s_qu)
							+ g->OFF1*(q(1,s_qu)+q(-1,s_qu))
							+ g->OFF2*(q(2,s_qu)+q(-2,s_qu))
							+ g->OFF3*(q(3,s_qu)+q(-3,s_qu))
							+ g->OFF4*(q(4,s_qu)+q(-4,s_qu)))*SQR(g->dxinv[1]);

		g->temp[VYY][idx] = ( g->CENTER*q(0,s_qv)
							+ g->OFF1*(q(1,s_qv)+q(-1,s_qv))
							+ g->OFF2*(q(2,s_qv)+q(-2,s_qv))
							+ g->OFF3*(q(3,s_qv)+q(-3,s_qv))
							+ g->OFF4*(q(4,s_qv)+q(-4,s_qv)))*SQR(g->dxinv[1]);

		g->temp[WYY][idx] = ( g->CENTER*q(0,s_qw)
							+ g->OFF1*(q(1,s_qw)+q(-1,s_qw))
							+ g->OFF2*(q(2,s_qw)+q(-2,s_qw))
							+ g->OFF3*(q(3,s_qw)+q(-3,s_qw))
							+ g->OFF4*(q(4,s_qw)+q(-4,s_qw)))*SQR(g->dxinv[1]);
	}

	idx = (sk-g->ng)*g->plane_offset_padded + sj*g->pitch[0] + (si-g->ng);
	if( g->ng <= si && si < g->dim[0] + g->ng &&
					   sj < g->dim[1] 		  &&
		g->ng <= sk && sk < g->dim[2] + g->ng ){

		g->temp[TYY][idx] = ( g->CENTER*q(0,s_qt)
							+ g->OFF1*(q(1,s_qt)+q(-1,s_qt))
							+ g->OFF2*(q(2,s_qt)+q(-2,s_qt))
							+ g->OFF3*(q(3,s_qt)+q(-3,s_qt))
							+ g->OFF4*(q(4,s_qt)+q(-4,s_qt)))*SQR(g->dxinv[1]);
	}
#undef	q
#define	q(i, comp)	s_q[comp][threadIdx.y][threadIdx.x+g->ng+(i)]
	idx = sk*g->plane_offset_g_padded + sj*g->pitch_g[0] + (si+g->ng);
	if(si < g->dim[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){

		g->temp[UX][idx] =  ( g->ALP*(q(1,s_qu)-q(-1,s_qu))
							+ g->BET*(q(2,s_qu)-q(-2,s_qu))
							+ g->GAM*(q(3,s_qu)-q(-3,s_qu))
							+ g->DEL*(q(4,s_qu)-q(-4,s_qu)))*g->dxinv[0];

		g->temp[VX][idx] = 	( g->ALP*(q(1,s_qv)-q(-1,s_qv))
							+ g->BET*(q(2,s_qv)-q(-2,s_qv))
							+ g->GAM*(q(3,s_qv)-q(-3,s_qv))
							+ g->DEL*(q(4,s_qv)-q(-4,s_qv)))*g->dxinv[0];

		g->temp[WX][idx] =	( g->ALP*(q(1,s_qw)-q(-1,s_qw))
							+ g->BET*(q(2,s_qw)-q(-2,s_qw))
							+ g->GAM*(q(3,s_qw)-q(-3,s_qw))
							+ g->DEL*(q(4,s_qw)-q(-4,s_qw)))*g->dxinv[0];

		g->temp[UXX][idx] = ( g->CENTER*q(0,s_qu)
							+ g->OFF1*(q(1,s_qu)+q(-1,s_qu))
							+ g->OFF2*(q(2,s_qu)+q(-2,s_qu))
							+ g->OFF3*(q(3,s_qu)+q(-3,s_qu))
							+ g->OFF4*(q(4,s_qu)+q(-4,s_qu)))*SQR(g->dxinv[0]);

		g->temp[VXX][idx] = ( g->CENTER*q(0,s_qv)
							+ g->OFF1*(q(1,s_qv)+q(-1,s_qv))
							+ g->OFF2*(q(2,s_qv)+q(-2,s_qv))
							+ g->OFF3*(q(3,s_qv)+q(-3,s_qv))
							+ g->OFF4*(q(4,s_qv)+q(-4,s_qv)))*SQR(g->dxinv[0]);

		g->temp[WXX][idx] = ( g->CENTER*q(0,s_qw)
							+ g->OFF1*(q(1,s_qw)+q(-1,s_qw))
							+ g->OFF2*(q(2,s_qw)+q(-2,s_qw))
							+ g->OFF3*(q(3,s_qw)+q(-3,s_qw))
							+ g->OFF4*(q(4,s_qw)+q(-4,s_qw)))*SQR(g->dxinv[0]);
	}

	idx = (sk-g->ng)*g->plane_offset_padded + (sj-g->ng)*g->pitch[0] + si;
	if(                si < g->dim[0]         &&
		g->ng <= sj && sj < g->dim[1] + g->ng &&
		g->ng <= sk && sk < g->dim[2] + g->ng ){

		g->temp[TXX][idx] = ( g->CENTER*q(0,s_qt)
							+ g->OFF1*(q(1,s_qt)+q(-1,s_qt))
							+ g->OFF2*(q(2,s_qt)+q(-2,s_qt))
							+ g->OFF3*(q(3,s_qt)+q(-3,s_qt))
							+ g->OFF4*(q(4,s_qt)+q(-4,s_qt)))*SQR(g->dxinv[0]);
	}
#undef	q
}

__global__ void gpu_diffterm_lv2_kernel(
	global_const_t *g,			// i: Global struct containing application parameters
	double *q,					// i:
	double *difflux				// o:
){
	int si, sj, sk;
	int idx, idx_g, tidx, tidy, tidz;
	double divu, tauxx, tauyy, tauzz, tauxy, tauxz, tauyz, mechwork;

	__shared__ double ux[BLOCK_DIM+NG+NG][BLOCK_DIM+NG+NG];
	__shared__ double wz[BLOCK_DIM+NG+NG][BLOCK_DIM+NG+NG];
	__shared__ double vy[BLOCK_DIM+NG+NG][BLOCK_DIM+NG+NG];


#define	ux(i)	ux[threadIdx.y+g->ng+(i)][threadIdx.x]
#define	vy(i)	vy[threadIdx.y+g->ng+(i)][threadIdx.x]

	/*** XZ ***/
	si = blockIdx.x*blockDim.x + threadIdx.x;
	sk = blockIdx.y*blockDim.y + threadIdx.y;
	sj = blockIdx.z;

	tidx = threadIdx.x;
	tidz = threadIdx.y;
	while(tidz < blockDim.y+NG+NG && si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){
		idx = sk*g->plane_offset_g_padded + (sj+g->ng)*g->pitch_g[0] + si+g->ng;

		ux[tidz][tidx] = g->temp[UX][idx];
		vy[tidz][tidx] = g->temp[VY][idx];

		tidz += blockDim.y;
		sk	 += blockDim.y;
	}
	__syncthreads();

	sk 		= blockIdx.y*blockDim.y + threadIdx.y;
	idx		= sk*g->plane_offset_padded + sj*g->pitch[0] + si;
	idx_g	= (sk+g->ng)*g->plane_offset_g_padded + (sj+g->ng)*g->pitch_g[0] + si+g->ng;
	if(si < g->dim[0] && sj < g->dim[1] && sk < g->dim[2]){

		g->temp[UXZ][idx] = ( g->ALP*(ux(1)-ux(-1))
							+ g->BET*(ux(2)-ux(-2))
							+ g->GAM*(ux(3)-ux(-3))
							+ g->DEL*(ux(4)-ux(-4)))*g->dxinv[2];

		g->temp[VYZ][idx] = ( g->ALP*(vy(1)-vy(-1))
							+ g->BET*(vy(2)-vy(-2))
							+ g->GAM*(vy(3)-vy(-3))
							+ g->DEL*(vy(4)-vy(-4)))*g->dxinv[2];

		difflux[idx + imz*g->comp_offset_padded] = 	g->eta * ( g->temp[WXX][idx_g] +
															   g->temp[WYY][idx_g] +
															   g->FourThirds * g->temp[WZZ][idx_g] +
													g->OneThird*(g->temp[UXZ][idx]+g->temp[VYZ][idx]));
	}
#undef	ux
#undef 	vy

	/*** XY ***/
	si = blockIdx.x*blockDim.x + threadIdx.x;
	sj = blockIdx.y*blockDim.y + threadIdx.y;
	sk = blockIdx.z;

	__syncthreads();
	for(sj = blockIdx.y*blockDim.y+threadIdx.y, tidy=threadIdx.y; tidy<blockDim.y+NG+NG; sj+=blockDim.y, tidy+=blockDim.y){
		for(si = blockIdx.x*blockDim.x+threadIdx.x, tidx=threadIdx.x; tidx<blockDim.x+NG+NG; si+=blockDim.x, tidx+=blockDim.x){
			if(si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim[2]){
				idx = (sk+g->ng)*g->plane_offset_g_padded + sj*g->pitch_g[0] + si;
				ux[tidy][tidx] = g->temp[UX][idx];
				wz[tidy][tidx] = g->temp[WZ][idx];
				vy[tidy][tidx] = g->temp[VY][idx];
			}
		}
	}
	__syncthreads();

	si 		= blockIdx.x*blockDim.x + threadIdx.x;
	sj 		= blockIdx.y*blockDim.y + threadIdx.y;
	idx 	= sk*g->plane_offset_padded + sj*g->pitch[0] + si;
	idx_g	= (sk+g->ng)*g->plane_offset_g_padded + (sj+g->ng)*g->pitch_g[0] + si+g->ng;
	if(si < g->dim[0] && sj < g->dim[1] && sk < g->dim[2]){

#define ux(i)	ux[threadIdx.y+g->ng+(i)][threadIdx.x+g->ng]
#define	wz(i)	wz[threadIdx.y+g->ng+(i)][threadIdx.x+g->ng]

		g->temp[UXY][idx] = ( g->ALP*(ux(1)-ux(-1))
							+ g->BET*(ux(2)-ux(-2))
							+ g->GAM*(ux(3)-ux(-3))
							+ g->DEL*(ux(4)-ux(-4)))*g->dxinv[1];

		g->temp[WZY][idx] = ( g->ALP*(wz(1)-wz(-1))
							+ g->BET*(wz(2)-wz(-2))
							+ g->GAM*(wz(3)-wz(-3))
							+ g->DEL*(wz(4)-wz(-4)))*g->dxinv[1];

		difflux[idx + imy*g->comp_offset_padded] = 	g->eta * ( g->temp[VXX][idx_g] +
													g->FourThirds * g->temp[VYY][idx_g] +
																	g->temp[VZZ][idx_g] +
													g->OneThird*(g->temp[UXY][idx]+g->temp[WZY][idx]));
#undef	wz
#define	vy(i)	vy[threadIdx.y+g->ng][threadIdx.x+g->ng+(i)]
#define	wz(i)	wz[threadIdx.y+g->ng][threadIdx.x+g->ng+(i)]

		g->temp[VYX][idx] = ( g->ALP*(vy(1)-vy(-1))
							+ g->BET*(vy(2)-vy(-2))
							+ g->GAM*(vy(3)-vy(-3))
							+ g->DEL*(vy(4)-vy(-4)))*g->dxinv[0];

		g->temp[WZX][idx] = ( g->ALP*(wz(1)-wz(-1))
							+ g->BET*(wz(2)-wz(-2))
							+ g->GAM*(wz(3)-wz(-3))
							+ g->DEL*(wz(4)-wz(-4)))*g->dxinv[0];

		difflux[idx + imx*g->comp_offset_padded] = 	 g->eta *
												   ( g->FourThirds*g->temp[UXX][idx_g] +
																   g->temp[UYY][idx_g] +
																   g->temp[UZZ][idx_g] +
													 g->OneThird *(g->temp[VYX][idx] + g->temp[WZX][idx]));

		difflux[idx + irho*g->comp_offset_padded] = 0.0;

		divu  = g->TwoThirds*(ux(0)+vy(0)+wz(0));
		tauxx = 2.E0*ux(0) - divu;
		tauyy = 2.E0*vy(0) - divu;
		tauzz = 2.E0*wz(0) - divu;
		tauxy = g->temp[UY][idx_g]+g->temp[VX][idx_g];
		tauxz = g->temp[UZ][idx_g]+g->temp[WX][idx_g];
		tauyz = g->temp[VZ][idx_g]+g->temp[WY][idx_g];

		mechwork = 	tauxx*ux(0) +
					tauyy*vy(0) +
					tauzz*wz(0) + SQR(tauxy)+SQR(tauxz)+SQR(tauyz);

		mechwork = 	g->eta*mechwork
					+ difflux[idx + imx*g->comp_offset_padded]*q[idx_g + qu*g->comp_offset_g_padded]
					+ difflux[idx + imy*g->comp_offset_padded]*q[idx_g + qv*g->comp_offset_g_padded];
//					+ difflux[idx + imz*g->comp_offset_padded]*q[idx_g + qw*g->comp_offset_g_padded];

//		difflux[idx + iene*g->comp_offset] = g->alam*(g->temp[TXX][idx]+g->temp[TYY][idx]+g->temp[TZZ][idx]) + mechwork;
		difflux[idx + iene*g->comp_offset_padded] = mechwork;

#undef	vy
#undef	wz
	}
}

__global__ void gpu_diffterm_lv3_kernel(
	global_const_t *g,			// i: Global struct containing application parameters
	double *q,					// i:
	double *difflux,			// o:
	double *flux				// o: set zeroes for hypterm
){
	int si, sj, sk, idx, idx_g;
	double mechwork;

	si = blockIdx.x*blockDim.x + threadIdx.x;
	sj = blockIdx.y*blockDim.y + threadIdx.y;
	sk = blockIdx.z;

	idx = sk*g->plane_offset_padded + sj*g->pitch[0] + si;
	idx_g	= (sk+g->ng)*g->plane_offset_g_padded + (sj+g->ng)*g->pitch_g[0] + si+g->ng;
	if(si < g->dim[0] && sj < g->dim[1] && sk < g->dim[2]){

		mechwork = 	difflux[idx + iene*g->comp_offset_padded] +
					difflux[idx + imz*g->comp_offset_padded]*q[idx_g + qw*g->comp_offset_g_padded];
		difflux[idx + iene*g->comp_offset_padded] =
					g->alam*(g->temp[TXX][idx]+g->temp[TYY][idx]+g->temp[TZZ][idx]) + mechwork;

//		flux[idx + irho*g->comp_offset_padded] = 0.0;
//		flux[idx + imx*g->comp_offset_padded] = 0.0;
//		flux[idx + imy*g->comp_offset_padded] = 0.0;
//		flux[idx + imz*g->comp_offset_padded] = 0.0;
//		flux[idx + iene*g->comp_offset_padded] = 0.0;
	}
}

void gpu_diffterm2(
	global_const_t h_const, 	// i: Global struct containing application parameters
	global_const_t *d_const,	// i: Device pointer to global struct containing application paramters
	double *d_q,				// i:
	double *d_difflux,			// o:
	double *d_flux				// o: just set zeroes for hypterm
){
	kernel_const_t h_kc;
	dim3 grid_dim(CEIL(h_const.dim_g[0], BLOCK_DIM), CEIL(h_const.dim_g[1], BLOCK_DIM), h_const.dim_g[2]);
	dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
	gpu_diffterm_lv1_kernel<<<grid_dim, block_dim>>>(d_const, d_q, d_difflux);
	gpu_diffterm_lv2_kernel<<<grid_dim, block_dim>>>(d_const, d_q, d_difflux);
	gpu_diffterm_lv3_kernel<<<grid_dim, block_dim>>>(d_const, d_q, d_difflux, d_flux);
}
