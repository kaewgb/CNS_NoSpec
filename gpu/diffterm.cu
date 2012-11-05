#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "util.h"

#define	BLOCK_DIM	16
#define	BLOCK_DIM_G	8
#define	q(i, comp)	s_q[comp][threadIdx.x+g->ng+(i)][threadIdx.z]


__global__ void gpu_diffterm_x_stencil_kernel(
	global_const_t *g,			// i: Global struct containing application parameters
	double *q,					// i:
	double *difflux				// o:
){
	int idx, tidx, tidz;
	int bi, bj, bk, si, sj, sk;
	kernel_const_t *kc = g->kc;
	__shared__ double  s_q[s_qend][BLOCK_DIM_G+NG+NG][BLOCK_DIM];

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

		idx = sk*g->plane_offset_g_padded + sj*g->pitch_g[0] + si;

		s_q[s_qu][tidx][tidz]  =  q[idx + qu*g->comp_offset_g_padded];
		s_q[s_qv][tidx][tidz]  =  q[idx + qv*g->comp_offset_g_padded];
		s_q[s_qw][tidx][tidz]  =  q[idx + qw*g->comp_offset_g_padded];
		s_q[s_qt][tidx][tidz]  =  q[idx + qt*g->comp_offset_g_padded];

		tidx += blockDim.x;
		si   += blockDim.x;
	}
	__syncthreads();

	si = bi*blockDim.x+threadIdx.x;
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
	if(                si < g->dim[0] &&
		g->ng <= sj && sj < g->dim[1] + g->ng &&
		g->ng <= sk && sk < g->dim[2] + g->ng ){

		g->temp[TXX][idx] = ( g->CENTER*q(0,s_qt)
							+ g->OFF1*(q(1,s_qt)+q(-1,s_qt))
							+ g->OFF2*(q(2,s_qt)+q(-2,s_qt))
							+ g->OFF3*(q(3,s_qt)+q(-3,s_qt))
							+ g->OFF4*(q(4,s_qt)+q(-4,s_qt)))*SQR(g->dxinv[0]);
	}
}

__global__ void gpu_diffterm_x_stencil_kernel_lv2(
	global_const_t *g,			// i: Global struct containing application parameters
	double *q,					// i:
	double *difflux			// o:
){
	int idx, idx_g, tidx, tidz;
	int bi, bj, bk, si, sj, sk;
	kernel_const_t *kc = g->kc;
	__shared__ double  vy[BLOCK_DIM_G+NG+NG][BLOCK_DIM];
	__shared__ double  wz[BLOCK_DIM_G+NG+NG][BLOCK_DIM];

	// Load to shared mem
	bi = (blockIdx.x % (kc->gridDim_plane_xz)) / kc->gridDim_z;
	bk = (blockIdx.x % (kc->gridDim_plane_xz)) % kc->gridDim_z;
	bj =  blockIdx.x / (kc->gridDim_plane_xz);
	si = bi*blockDim.x+threadIdx.x;
	sj = bj*blockDim.y+threadIdx.y; // = bj
	sk = bk*blockDim.z+threadIdx.z;

	tidx = threadIdx.x;
	tidz = threadIdx.z;
	while( tidx < kc->blockDim_x_g && si < g->dim_g[0] && sj < g->dim[1] && sk < g->dim[2]){

		idx = (sk+g->ng)*g->plane_offset_g_padded + (sj+g->ng)*g->pitch_g[0] + si;
		vy[tidx][tidz]  =  g->temp[VY][idx];
		tidx += blockDim.x;
		si   += blockDim.x;
	}

	tidx = threadIdx.x;
	si = bi*blockDim.x+threadIdx.x;
	while( tidx < kc->blockDim_x_g && si < g->dim_g[0] && sj < g->dim[1] && sk < g->dim[2]){

		idx = (sk+g->ng)*g->plane_offset_g_padded + (sj+g->ng)*g->pitch_g[0] + si;
		wz[tidx][tidz]  =  g->temp[WZ][idx];

		tidx += blockDim.x;
		si   += blockDim.x;
	}
	__syncthreads();

#define	vy(i)	vy[threadIdx.x+g->ng+(i)][threadIdx.z]
#define	wz(i)	wz[threadIdx.x+g->ng+(i)][threadIdx.z]

	si = bi*blockDim.x+threadIdx.x;
	idx 	= sk*g->plane_offset_padded + sj*g->pitch[0] + si;
	idx_g 	= (sk+g->ng)*g->plane_offset_g_padded + (sj+g->ng)*g->pitch_g[0] + (si+g->ng);
	if(si < g->dim[0] && sj < g->dim[1] && sk < g->dim[2]){

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
	}
#undef	vy
#undef	wz
}
#undef	q

__global__ void gpu_diffterm_yz_stencil_kernel(
	global_const_t *g,			// i: Global struct containing application parameters
	double *q,					// i:
	double *difflux				// o:
){
	int idx, tidy, tidz;
	int bi, bj, bk, si, sj, sk;
	kernel_const_t *kc = g->kc;
	__shared__ double  s_q[s_qend][BLOCK_DIM_G+NG+NG][BLOCK_DIM_G+NG+NG];

	// Load to shared mem
	bj = (blockIdx.x % (kc->gridDim_plane_yz)) / kc->gridDim_z;
	bk = (blockIdx.x % (kc->gridDim_plane_yz)) % kc->gridDim_z;
	bi =  blockIdx.x / (kc->gridDim_plane_yz);
	si = bi;
	sj = bj*BLOCK_DIM_G+threadIdx.y;
	sk = bk*BLOCK_DIM_G+threadIdx.z;

	tidy = threadIdx.y;
	tidz = threadIdx.z;
	idx = sk*g->plane_offset_g_padded + sj*g->pitch_g[0] + si;
	if(si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){
		s_q[s_qu][tidy][tidz]  =  q[idx + qu*g->comp_offset_g_padded];
		s_q[s_qv][tidy][tidz]  =  q[idx + qv*g->comp_offset_g_padded];
		s_q[s_qw][tidy][tidz]  =  q[idx + qw*g->comp_offset_g_padded];
		s_q[s_qt][tidy][tidz]  =  q[idx + qt*g->comp_offset_g_padded];
	}
	__syncthreads();

	sj = bj*BLOCK_DIM_G+threadIdx.y;
	sk = bk*BLOCK_DIM_G+threadIdx.z;

	if(threadIdx.y < BLOCK_DIM_G && threadIdx.z < BLOCK_DIM_G){
#define	q(i, comp)	s_q[comp][threadIdx.y+g->ng+(i)][threadIdx.z]

		idx = sk*g->plane_offset_g_padded + (sj+g->ng)*g->pitch_g[0] + si;
		if(si < g->dim_g[0] && sj < g->dim[1] && sk < g->dim_g[2]){

//			g->temp[UY][idx] =  q(0, s_qu);
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

		idx = (sk-g->ng)*g->plane_offset_padded + sj*g->pitch[0] + si-g->ng;
		if( g->ng <= si && si < g->dim[0] + g->ng &&
			               sj < g->dim[1] &&
			g->ng <= sk && sk < g->dim[2] + g->ng ){

			g->temp[TYY][idx] = ( g->CENTER*q(0,s_qt)
								+ g->OFF1*(q(1,s_qt)+q(-1,s_qt))
								+ g->OFF2*(q(2,s_qt)+q(-2,s_qt))
								+ g->OFF3*(q(3,s_qt)+q(-3,s_qt))
								+ g->OFF4*(q(4,s_qt)+q(-4,s_qt)))*SQR(g->dxinv[1]);
		}

#undef	q
#define	q(i, comp)	s_q[comp][threadIdx.y][threadIdx.z+g->ng+(i)]

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
	}
}

__global__ void gpu_diffterm_yz_stencil_kernel_lv2(
	global_const_t *g,			// i: Global struct containing application parameters
	double *q,					// i:
	double *difflux				// o:
){
	int idx, idx_g, tidy, tidz;
	int bi, bj, bk, si, sj, sk;
	kernel_const_t *kc = g->kc;
	__shared__ double  ux[BLOCK_DIM_G+NG+NG][BLOCK_DIM_G+NG+NG];
	__shared__ double  wz[BLOCK_DIM_G+NG+NG][BLOCK_DIM_G+NG+NG];
	__shared__ double  vy[BLOCK_DIM_G+NG+NG][BLOCK_DIM_G+NG+NG];

	double divu, tauxx, tauyy, tauzz, tauxy, tauxz, tauyz, mechwork;

	// Load to shared mem
	bj = (blockIdx.x % (kc->gridDim_plane_yz)) / kc->gridDim_z;
	bk = (blockIdx.x % (kc->gridDim_plane_yz)) % kc->gridDim_z;
	bi =  blockIdx.x / (kc->gridDim_plane_yz);
	si = bi;
	sj = bj*BLOCK_DIM_G+threadIdx.y;
	sk = bk*BLOCK_DIM_G+threadIdx.z;

	tidy = threadIdx.y;
	tidz = threadIdx.z;
	idx = sk*g->plane_offset_g_padded + sj*g->pitch_g[0] + (si+g->ng);
	if(si < g->dim[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){
		ux[tidy][tidz]  =  g->temp[UX][idx];
		wz[tidy][tidz]  =  g->temp[WZ][idx];
		vy[tidy][tidz]  =  g->temp[VY][idx];
	}
	__syncthreads();

#define	ux(i)	ux[threadIdx.y+g->ng+(i)][threadIdx.z+g->ng]
#define	wz(i)	wz[threadIdx.y+g->ng+(i)][threadIdx.z+g->ng]

	if(threadIdx.y < BLOCK_DIM_G && threadIdx.z < BLOCK_DIM_G){

		idx = sk*g->plane_offset_padded + sj*g->pitch[0] + si;
		if(si < g->dim[0] && sj < g->dim[1] && sk < g->dim[2]){

			g->temp[UXY][idx] = ( g->ALP*(ux(1)-ux(-1))
								+ g->BET*(ux(2)-ux(-2))
								+ g->GAM*(ux(3)-ux(-3))
								+ g->DEL*(ux(4)-ux(-4)))*g->dxinv[1];

			g->temp[WZY][idx] = ( g->ALP*(wz(1)-wz(-1))
								+ g->BET*(wz(2)-wz(-2))
								+ g->GAM*(wz(3)-wz(-3))
								+ g->DEL*(wz(4)-wz(-4)))*g->dxinv[1];

			idx_g = (sk+g->ng)*g->plane_offset_g_padded + (sj+g->ng)*g->pitch_g[0] + (si+g->ng);

			difflux[idx + imy*g->comp_offset_padded] = 	g->eta * ( g->temp[VXX][idx_g] +
														g->FourThirds * g->temp[VYY][idx_g] +
																		g->temp[VZZ][idx_g] +
														g->OneThird*(g->temp[UXY][idx]+g->temp[WZY][idx]));
		}
	}

#undef	ux
#define	ux(i)	ux[threadIdx.y+g->ng][threadIdx.z+g->ng+(i)]
#define vy(i)	vy[threadIdx.y+g->ng][threadIdx.z+g->ng+(i)]

	idx = sk*g->plane_offset_padded + sj*g->pitch[0] + si;
	idx_g = (sk+g->ng)*g->plane_offset_g_padded + (sj+g->ng)*g->pitch_g[0] + (si+g->ng);
	if(threadIdx.y < BLOCK_DIM_G && threadIdx.z < BLOCK_DIM_G){

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

		// Last part
		if(si < g->dim[0] && sj < g->dim[1] && sk < g->dim[2]){
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
						+ difflux[idx + imy*g->comp_offset_padded]*q[idx_g + qv*g->comp_offset_g_padded]
						+ difflux[idx + imz*g->comp_offset_padded]*q[idx_g + qw*g->comp_offset_g_padded];

			difflux[idx + iene*g->comp_offset_padded] = g->alam*(g->temp[TXX][idx]+g->temp[TYY][idx]+g->temp[TZZ][idx]) + mechwork;
//			if(si==31 && sj==31 && sk==15){
//				if(difflux[idx + iene*g->comp_offset_padded] !=  g->alam*(g->temp[TXX][idx]+g->temp[TYY][idx]+g->temp[TZZ][idx]) + mechwork + difflux[idx + imz*g->comp_offset_padded]*q[idx_g + qw*g->comp_offset_g_padded])
//					printf("[%d][%d][%d] %le vs %le\n", si, sj, sk, difflux[idx + iene*g->comp_offset_padded],  g->alam*(g->temp[TXX][idx]+g->temp[TYY][idx]+g->temp[TZZ][idx]) + mechwork + difflux[idx + imz*g->comp_offset_padded]*q[idx_g + qw*g->comp_offset_g_padded]);
//			}
		}
	}

#undef	ux
#undef	vy
#undef 	wz
}

void gpu_diffterm(
	global_const_t h_const, 	// i: Global struct containing application parameters
	global_const_t *d_const,	// i: Device pointer to global struct containing application paramters
	double *d_q,				// i:
	double *d_difflux				// o:
){
	int grid_dim;
	kernel_const_t h_kc;

	dim3 block_dim_x_stencil(BLOCK_DIM_G, 1, BLOCK_DIM);
	h_kc.gridDim_x = CEIL(h_const.dim[0], BLOCK_DIM_G);
	h_kc.gridDim_y = h_const.dim_g[1];
	h_kc.gridDim_z = CEIL(h_const.dim_g[2], BLOCK_DIM);
	h_kc.gridDim_plane_xz = h_kc.gridDim_x * h_kc.gridDim_z;
    h_kc.blockDim_x_g = BLOCK_DIM_G + h_const.ng + h_const.ng;
    grid_dim = h_kc.gridDim_plane_xz * h_kc.gridDim_y;
    cudaMemcpy(h_const.kc, &h_kc, sizeof(kernel_const_t), cudaMemcpyHostToDevice);

	gpu_diffterm_x_stencil_kernel<<<grid_dim, block_dim_x_stencil>>>(d_const, d_q, d_difflux);

	dim3 block_dim_yz_stencil(1, BLOCK_DIM, BLOCK_DIM);
	h_kc.gridDim_x = h_const.dim_g[0];
	h_kc.gridDim_y = CEIL(h_const.dim_g[1], BLOCK_DIM_G);
	h_kc.gridDim_z = CEIL(h_const.dim_g[2], BLOCK_DIM_G);
	h_kc.gridDim_plane_yz = h_kc.gridDim_y * h_kc.gridDim_z;
    h_kc.blockDim_y_g = BLOCK_DIM_G + h_const.ng + h_const.ng;
    h_kc.blockDim_z_g = BLOCK_DIM_G + h_const.ng + h_const.ng;
	grid_dim = h_kc.gridDim_x * h_kc.gridDim_plane_yz;
    cudaMemcpy(h_const.kc, &h_kc, sizeof(kernel_const_t), cudaMemcpyHostToDevice);

	gpu_diffterm_yz_stencil_kernel<<<grid_dim, block_dim_yz_stencil>>>(d_const, d_q, d_difflux);

	h_kc.gridDim_x = CEIL(h_const.dim[0], BLOCK_DIM_G);
	h_kc.gridDim_y = h_const.dim[1];
	h_kc.gridDim_z = CEIL(h_const.dim[2], BLOCK_DIM);
	h_kc.gridDim_plane_xz = h_kc.gridDim_x * h_kc.gridDim_z;
    h_kc.blockDim_x_g = BLOCK_DIM_G + h_const.ng + h_const.ng;
    grid_dim = h_kc.gridDim_plane_xz * h_kc.gridDim_y;
    cudaMemcpy(h_const.kc, &h_kc, sizeof(kernel_const_t), cudaMemcpyHostToDevice);

	gpu_diffterm_x_stencil_kernel_lv2<<<grid_dim, block_dim_x_stencil>>>(d_const, d_q, d_difflux);

	h_kc.gridDim_x = h_const.dim_g[0];
	h_kc.gridDim_y = CEIL(h_const.dim_g[1], BLOCK_DIM_G);
	h_kc.gridDim_z = CEIL(h_const.dim_g[2], BLOCK_DIM_G);
	h_kc.gridDim_plane_yz = h_kc.gridDim_y * h_kc.gridDim_z;
    h_kc.blockDim_y_g = BLOCK_DIM_G + h_const.ng + h_const.ng;
    h_kc.blockDim_z_g = BLOCK_DIM_G + h_const.ng + h_const.ng;
	grid_dim = h_kc.gridDim_x * h_kc.gridDim_plane_yz;
    cudaMemcpy(h_const.kc, &h_kc, sizeof(kernel_const_t), cudaMemcpyHostToDevice);

	gpu_diffterm_yz_stencil_kernel_lv2<<<grid_dim, block_dim_yz_stencil>>>(d_const, d_q, d_difflux);

}
void gpu_diffterm_lv1(
	global_const_t h_const, 	// i: Global struct containing application parameters
	global_const_t *d_const,	// i: Device pointer to global struct containing application paramters
	double *d_q,				// i:
	double *d_difflux				// o:
){
	int grid_dim;
	kernel_const_t h_kc;

	dim3 block_dim_x_stencil(BLOCK_DIM_G, 1, BLOCK_DIM);
	h_kc.gridDim_x = CEIL(h_const.dim[0], BLOCK_DIM_G);
	h_kc.gridDim_y = h_const.dim_g[1];
	h_kc.gridDim_z = CEIL(h_const.dim_g[2], BLOCK_DIM);
	h_kc.gridDim_plane_xz = h_kc.gridDim_x * h_kc.gridDim_z;
    h_kc.blockDim_x_g = BLOCK_DIM_G + h_const.ng + h_const.ng;
    grid_dim = h_kc.gridDim_plane_xz * h_kc.gridDim_y;
    cudaMemcpy(h_const.kc, &h_kc, sizeof(kernel_const_t), cudaMemcpyHostToDevice);

	gpu_diffterm_x_stencil_kernel<<<grid_dim, block_dim_x_stencil>>>(d_const, d_q, d_difflux);

	dim3 block_dim_yz_stencil(1, BLOCK_DIM, BLOCK_DIM);
	h_kc.gridDim_x = h_const.dim_g[0];
	h_kc.gridDim_y = CEIL(h_const.dim_g[1], BLOCK_DIM_G);
	h_kc.gridDim_z = CEIL(h_const.dim_g[2], BLOCK_DIM_G);
	h_kc.gridDim_plane_yz = h_kc.gridDim_y * h_kc.gridDim_z;
    h_kc.blockDim_y_g = BLOCK_DIM_G + h_const.ng + h_const.ng;
    h_kc.blockDim_z_g = BLOCK_DIM_G + h_const.ng + h_const.ng;
	grid_dim = h_kc.gridDim_x * h_kc.gridDim_plane_yz;
    cudaMemcpy(h_const.kc, &h_kc, sizeof(kernel_const_t), cudaMemcpyHostToDevice);

	gpu_diffterm_yz_stencil_kernel<<<grid_dim, block_dim_yz_stencil>>>(d_const, d_q, d_difflux);
}

void gpu_diffterm_lv2(
	global_const_t h_const, 	// i: Global struct containing application parameters
	global_const_t *d_const,	// i: Device pointer to global struct containing application paramters
	double *d_q,				// i:
	double *d_difflux				// o:
){
	int grid_dim;
	kernel_const_t h_kc;

	dim3 block_dim_x_stencil(BLOCK_DIM_G, 1, BLOCK_DIM);
	dim3 block_dim_yz_stencil(1, BLOCK_DIM, BLOCK_DIM);

	h_kc.gridDim_x = CEIL(h_const.dim[0], BLOCK_DIM_G);
	h_kc.gridDim_y = h_const.dim[1];
	h_kc.gridDim_z = CEIL(h_const.dim[2], BLOCK_DIM);
	h_kc.gridDim_plane_xz = h_kc.gridDim_x * h_kc.gridDim_z;
    h_kc.blockDim_x_g = BLOCK_DIM_G + h_const.ng + h_const.ng;
    grid_dim = h_kc.gridDim_plane_xz * h_kc.gridDim_y;
    cudaMemcpy(h_const.kc, &h_kc, sizeof(kernel_const_t), cudaMemcpyHostToDevice);

	gpu_diffterm_x_stencil_kernel_lv2<<<grid_dim, block_dim_x_stencil>>>(d_const, d_q, d_difflux);

	h_kc.gridDim_x = h_const.dim_g[0];
	h_kc.gridDim_y = CEIL(h_const.dim_g[1], BLOCK_DIM_G);
	h_kc.gridDim_z = CEIL(h_const.dim_g[2], BLOCK_DIM_G);
	h_kc.gridDim_plane_yz = h_kc.gridDim_y * h_kc.gridDim_z;
    h_kc.blockDim_y_g = BLOCK_DIM_G + h_const.ng + h_const.ng;
    h_kc.blockDim_z_g = BLOCK_DIM_G + h_const.ng + h_const.ng;
	grid_dim = h_kc.gridDim_x * h_kc.gridDim_plane_yz;
    cudaMemcpy(h_const.kc, &h_kc, sizeof(kernel_const_t), cudaMemcpyHostToDevice);

	gpu_diffterm_yz_stencil_kernel_lv2<<<grid_dim, block_dim_yz_stencil>>>(d_const, d_q, d_difflux);

}
