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
		idx = sk*g->plane_offset_g + sj*g->dim_g[0] + si;

		s_q[s_qu][tidz][tidx] = q[idx + qu*g->comp_offset_g];
		s_q[s_qv][tidz][tidx] = q[idx + qv*g->comp_offset_g];
		s_q[s_qw][tidz][tidx] = q[idx + qw*g->comp_offset_g];
		s_q[s_qt][tidz][tidx] = q[idx + qt*g->comp_offset_g];

		tidz += blockDim.y;
		sk	 += blockDim.y;
	}
	__syncthreads();

#define	q(i, comp)	s_q[comp][threadIdx.y+g->ng+(i)][threadIdx.x]

	sk = blockIdx.y*blockDim.y + threadIdx.y;
	idx = (sk+g->ng)*g->plane_offset_g + sj*g->dim_g[0] + si;
	if(si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim[2]){

//		if(g->temp[UZ][idx] != q(0, s_qu) && sk==0 && si==0 && sj==0){
//			printf("[%d][%d][%d] [%d][%d] %le != %le\n",
//					blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
//					g->temp[UZ][idx], q(0,s_qu));
//			for(int i=0;i<BLOCK_DIM+NG+NG;i++)
//				printf("%le\n", s_q[s_qu][i][0]);
//			printf("comp_offset_g = %d\n", g->comp_offset_g);
//		}
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

	idx = sk*g->plane_offset + (sj-g->ng)*g->dim[2] + (si-g->ng);
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

	for(sj = blockIdx.y*blockDim.y+threadIdx.y, tidy=threadIdx.y; tidy<blockDim.y+NG+NG; sj+=blockDim.y, tidy+=blockDim.y){
		for(si = blockIdx.x*blockDim.x+threadIdx.x, tidx=threadIdx.x; tidx<blockDim.x+NG+NG; si+=blockDim.x, tidx+=blockDim.x){
			if(si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){
				idx = sk*g->plane_offset_g + sj*g->dim_g[0] + si;

				s_q[s_qu][tidy][tidx] = q[idx + qu*g->comp_offset_g];
				s_q[s_qv][tidy][tidx] = q[idx + qv*g->comp_offset_g];
				s_q[s_qw][tidy][tidx] = q[idx + qw*g->comp_offset_g];
				s_q[s_qt][tidy][tidx] = q[idx + qt*g->comp_offset_g];

			}
		}
	}
	__syncthreads();

#define	q(i, comp)	s_q[comp][threadIdx.y+g->ng+(i)][threadIdx.x]
	si = blockIdx.x*blockDim.x + threadIdx.x;
	sj = blockIdx.y*blockDim.y + threadIdx.y;
	idx = sk*g->plane_offset_g + (sj+g->ng)*g->dim_g[0] + si;
	if(si < g->dim_g[0] && sj < g->dim[1] && sk < g->dim_g[2]){

//		if(g->temp[UY][idx] != q(0, s_qu) && sk==0 && si==0 && sj==26){
//		double sth =       ( g->ALP*(q(1,s_qu)-q(-1,s_qu))
//							+ g->BET*(q(2,s_qu)-q(-2,s_qu))
//							+ g->GAM*(q(3,s_qu)-q(-3,s_qu))
//							+ g->DEL*(q(4,s_qu)-q(-4,s_qu)))*g->dxinv[1];
//		if(g->temp[UY][idx] != sth){
//			printf("[%d][%d][%d] [%d][%d] -> [%d][%d][%d] %le != %le\n",
//					blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
//					blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y, blockIdx.z,
//					g->temp[UY][idx], q(0,s_qu));
//			for(int i=0;i<BLOCK_DIM+NG+NG;i++)
//				printf("%le\n", s_q[s_qu][i][0]);
//			printf("comp_offset_g = %d\n", g->comp_offset_g);
//		}
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

	idx = (sk-g->ng)*g->plane_offset + sj*g->dim[2] + (si-g->ng);
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
	idx = sk*g->plane_offset_g + sj*g->dim_g[0] + (si+g->ng);
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

	idx = (sk-g->ng)*g->plane_offset + (sj-g->ng)*g->dim[2] + si;
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

void gpu_diffterm2(
	global_const_t h_const, 	// i: Global struct containing application parameters
	global_const_t *d_const,	// i: Device pointer to global struct containing application paramters
	double *d_q,				// i:
	double *d_difflux				// o:
){
	kernel_const_t h_kc;
	dim3 grid_dim(CEIL(h_const.dim_g[0], BLOCK_DIM), CEIL(h_const.dim_g[1], BLOCK_DIM), h_const.dim_g[2]);
	dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
	gpu_diffterm_lv1_kernel<<<grid_dim, block_dim>>>(d_const, d_q, d_difflux);
//	gpu_diffterm_lv2_kernel<<<grid_dim, block_dim>>>(d_const, d_q, d_difflux);
}
