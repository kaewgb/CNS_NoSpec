#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "util.cuh"
#include "util.h"

#define BLOCK_SMALL		8
#define	BLOCK_LARGE		16
#define	THREAD_Z		8

__global__ void gpu_hypterm_xy_stencil_kernel(
	global_const_t *g,	// i:
	double *cons,		// i:
	double *q,			// i:
	double *flux		// o:
){
	bool compute=false;
	int z,idx,out,si,sj,sk,tidx,tidy;
	double unp1, unp2, unp3, unp4, unm1, unm2, unm3, unm4;
	double flux_irho, flux_imx, flux_imy, flux_imz, flux_iene;

	__shared__ double      s_qu[BLOCK_LARGE+NG+NG][BLOCK_LARGE+NG+NG];
	__shared__ double      s_qv[BLOCK_LARGE+NG+NG][BLOCK_LARGE+NG+NG];
	__shared__ double   s_qpres[BLOCK_LARGE+NG+NG][BLOCK_LARGE+NG+NG];
	__shared__ double    s_cons[BLOCK_LARGE+NG+NG][BLOCK_LARGE+NG+NG];

	// Load to shared mem
	for(z=0;z<THREAD_Z;z++){

		si = blockIdx.x*blockDim.x+threadIdx.x;
		sj = blockIdx.y*blockDim.y+threadIdx.y;
		sk = (blockIdx.z*blockDim.z+threadIdx.z)*THREAD_Z + z;

		out = sk*g->plane_offset_padded + sj*g->pitch[0] + si;
		compute = (si < g->dim[0] && sj < g->dim[1] && sk < g->dim[2]);

		for(sj=blockIdx.y*blockDim.y+threadIdx.y, tidy=threadIdx.y; tidy < BLOCK_LARGE+NG+NG; sj+=blockDim.y, tidy+=blockDim.y){
			for(si=blockIdx.x*blockDim.x+threadIdx.x, tidx=threadIdx.x; tidx < BLOCK_LARGE+NG+NG; si+=blockDim.x, tidx+=blockDim.x){
				if( si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){

					idx = (sk+g->ng)*g->plane_offset_g_padded + sj*g->pitch_g[0] + si;

							  s_qu[tidy][tidx]  =     q[idx + qu*g->comp_offset_g_padded];
							  s_qv[tidy][tidx]  =     q[idx + qv*g->comp_offset_g_padded];
						   s_qpres[tidy][tidx]	=     q[idx + qpres*g->comp_offset_g_padded];
							s_cons[tidy][tidx] 	=  cons[idx + imx*g->comp_offset_g_padded];
				}
			}
		}
		__syncthreads();

		if(compute){

	#define	s_qu(i)			s_qu[threadIdx.y+g->ng][threadIdx.x+g->ng+(i)]
	#define	s_qv(i)			s_qv[threadIdx.y+g->ng+(i)][threadIdx.x+g->ng]
	#define	s_qpres_x(i)	s_qpres[threadIdx.y+g->ng][threadIdx.x+g->ng+(i)]
	#define	s_qpres_y(i)	s_qpres[threadIdx.y+g->ng+(i)][threadIdx.x+g->ng]
	#define	s_imx_x(i)	    s_cons[threadIdx.y+g->ng][threadIdx.x+g->ng+(i)]
	#define	s_imx_y(i)	    s_cons[threadIdx.y+g->ng+(i)][threadIdx.x+g->ng]

			flux_irho = - ( g->ALP*(s_imx_x(1)-s_imx_x(-1))
						  + g->BET*(s_imx_x(2)-s_imx_x(-2))
						  + g->GAM*(s_imx_x(3)-s_imx_x(-3))
						  + g->DEL*(s_imx_x(4)-s_imx_x(-4)))*g->dxinv[0];

			flux_imx  = - ( g->ALP*(s_imx_x(1)*s_qu(1)-s_imx_x(-1)*s_qu(-1)
						  + (s_qpres_x(1)-s_qpres_x(-1)))
						  + g->BET*(s_imx_x(2)*s_qu(2)-s_imx_x(-2)*s_qu(-2)
						  + (s_qpres_x(2)-s_qpres_x(-2)))
						  + g->GAM*(s_imx_x(3)*s_qu(3)-s_imx_x(-3)*s_qu(-3)
						  + (s_qpres_x(3)-s_qpres_x(-3)))
						  + g->DEL*(s_imx_x(4)*s_qu(4)-s_imx_x(-4)*s_qu(-4)
						  + (s_qpres_x(4)-s_qpres_x(-4))))*g->dxinv[0];

			flux_imx -=   ( g->ALP*(s_imx_y(1)*s_qv(1)-s_imx_y(-1)*s_qv(-1))
						  + g->BET*(s_imx_y(2)*s_qv(2)-s_imx_y(-2)*s_qv(-2))
						  + g->GAM*(s_imx_y(3)*s_qv(3)-s_imx_y(-3)*s_qv(-3))
						  + g->DEL*(s_imx_y(4)*s_qv(4)-s_imx_y(-4)*s_qv(-4)))*g->dxinv[1];

		}
	#undef s_imx_x
	#undef s_imx_y
	#define	s_imy_x(i)	    s_cons[threadIdx.y+g->ng][threadIdx.x+g->ng+(i)]
	#define	s_imy_y(i)	    s_cons[threadIdx.y+g->ng+(i)][threadIdx.x+g->ng]

		__syncthreads();
		for(sj=blockIdx.y*blockDim.y+threadIdx.y, tidy=threadIdx.y; tidy < BLOCK_LARGE+NG+NG; sj+=blockDim.y, tidy+=blockDim.y){
			for(si=blockIdx.x*blockDim.x+threadIdx.x, tidx=threadIdx.x; tidx < BLOCK_LARGE+NG+NG; si+=blockDim.x, tidx+=blockDim.x){
				if( si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){

					idx = (sk+g->ng)*g->plane_offset_g_padded + sj*g->pitch_g[0] + si;
					s_cons[tidy][tidx] 	=  cons[idx + imy*g->comp_offset_g_padded];
				}
			}
		}
		__syncthreads();

		if(compute){

			flux_irho -=  ( g->ALP*(s_imy_y(1)-s_imy_y(-1))
						  + g->BET*(s_imy_y(2)-s_imy_y(-2))
						  + g->GAM*(s_imy_y(3)-s_imy_y(-3))
						  + g->DEL*(s_imy_y(4)-s_imy_y(-4)))*g->dxinv[1];

			flux_imy =   -( g->ALP*(s_imy_y(1)*s_qv(1)-s_imy_y(-1)*s_qv(-1)
						  + (s_qpres_y(1)-s_qpres_y(-1)))
						  + g->BET*(s_imy_y(2)*s_qv(2)-s_imy_y(-2)*s_qv(-2)
						  + (s_qpres_y(2)-s_qpres_y(-2)))
						  + g->GAM*(s_imy_y(3)*s_qv(3)-s_imy_y(-3)*s_qv(-3)
						  + (s_qpres_y(3)-s_qpres_y(-3)))
						  + g->DEL*(s_imy_y(4)*s_qv(4)-s_imy_y(-4)*s_qv(-4)
						  + (s_qpres_y(4)-s_qpres_y(-4))))*g->dxinv[1];

			flux_imy  -=  ( g->ALP*(s_imy_x(1)*s_qu(1)-s_imy_x(-1)*s_qu(-1))
						  + g->BET*(s_imy_x(2)*s_qu(2)-s_imy_x(-2)*s_qu(-2))
						  + g->GAM*(s_imy_x(3)*s_qu(3)-s_imy_x(-3)*s_qu(-3))
						  + g->DEL*(s_imy_x(4)*s_qu(4)-s_imy_x(-4)*s_qu(-4)))*g->dxinv[0];

		}
	#undef  s_imy_x
	#undef  s_imy_y
	#define	s_imz_x(i)	    s_cons[threadIdx.y+g->ng][threadIdx.x+g->ng+(i)]
	#define	s_imz_y(i)	    s_cons[threadIdx.y+g->ng+(i)][threadIdx.x+g->ng]

		__syncthreads();
		for(sj=blockIdx.y*blockDim.y+threadIdx.y, tidy=threadIdx.y; tidy < BLOCK_LARGE+NG+NG; sj+=blockDim.y, tidy+=blockDim.y){
			for(si=blockIdx.x*blockDim.x+threadIdx.x, tidx=threadIdx.x; tidx < BLOCK_LARGE+NG+NG; si+=blockDim.x, tidx+=blockDim.x){
				if( si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){

					idx = (sk+g->ng)*g->plane_offset_g_padded + sj*g->pitch_g[0] + si;
					s_cons[tidy][tidx] 	=  cons[idx + imz*g->comp_offset_g_padded];
				}
			}
		}
		__syncthreads();

		if(compute){

			flux_imz  = - ( g->ALP*(s_imz_x(1)*s_qu(1)-s_imz_x(-1)*s_qu(-1))
						  + g->BET*(s_imz_x(2)*s_qu(2)-s_imz_x(-2)*s_qu(-2))
						  + g->GAM*(s_imz_x(3)*s_qu(3)-s_imz_x(-3)*s_qu(-3))
						  + g->DEL*(s_imz_x(4)*s_qu(4)-s_imz_x(-4)*s_qu(-4)))*g->dxinv[0];

			flux_imz -=   ( g->ALP*(s_imz_y(1)*s_qv(1)-s_imz_y(-1)*s_qv(-1))
						  + g->BET*(s_imz_y(2)*s_qv(2)-s_imz_y(-2)*s_qv(-2))
						  + g->GAM*(s_imz_y(3)*s_qv(3)-s_imz_y(-3)*s_qv(-3))
						  + g->DEL*(s_imz_y(4)*s_qv(4)-s_imz_y(-4)*s_qv(-4)))*g->dxinv[1];

		}

	#undef  s_imz_x
	#undef  s_imz_y
	#define	s_iene_x(i)	    s_cons[threadIdx.y+g->ng][threadIdx.x+g->ng+(i)]
	#define	s_iene_y(i)	    s_cons[threadIdx.y+g->ng+(i)][threadIdx.x+g->ng]

		__syncthreads();
		for(sj=blockIdx.y*blockDim.y+threadIdx.y, tidy=threadIdx.y; tidy < BLOCK_LARGE+NG+NG; sj+=blockDim.y, tidy+=blockDim.y){
			for(si=blockIdx.x*blockDim.x+threadIdx.x, tidx=threadIdx.x; tidx < BLOCK_LARGE+NG+NG; si+=blockDim.x, tidx+=blockDim.x){
				if( si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){

					idx = (sk+g->ng)*g->plane_offset_g_padded + sj*g->pitch_g[0] + si;
					s_cons[tidy][tidx] 	=  cons[idx + iene*g->comp_offset_g_padded];
				}
			}
		}
		__syncthreads();

		if(compute){

			flux_iene =  -( g->ALP*(s_iene_x(1)*s_qu(1)-s_iene_x(-1)*s_qu(-1)
						  + (s_qpres_x(1)*s_qu(1)-s_qpres_x(-1)*s_qu(-1)))
						  + g->BET*(s_iene_x(2)*s_qu(2)-s_iene_x(-2)*s_qu(-2)
						  + (s_qpres_x(2)*s_qu(2)-s_qpres_x(-2)*s_qu(-2)))
						  + g->GAM*(s_iene_x(3)*s_qu(3)-s_iene_x(-3)*s_qu(-3)
						  + (s_qpres_x(3)*s_qu(3)-s_qpres_x(-3)*s_qu(-3)))
						  + g->DEL*(s_iene_x(4)*s_qu(4)-s_iene_x(-4)*s_qu(-4)
						  + (s_qpres_x(4)*s_qu(4)-s_qpres_x(-4)*s_qu(-4))))*g->dxinv[0];

			flux_iene -=  ( g->ALP*(s_iene_y(1)*s_qv(1)-s_iene_y(-1)*s_qv(-1)
						  + (s_qpres_y(1)*s_qv(1)-s_qpres_y(-1)*s_qv(-1)))
						  + g->BET*(s_iene_y(2)*s_qv(2)-s_iene_y(-2)*s_qv(-2)
						  + (s_qpres_y(2)*s_qv(2)-s_qpres_y(-2)*s_qv(-2)))
						  + g->GAM*(s_iene_y(3)*s_qv(3)-s_iene_y(-3)*s_qv(-3)
						  + (s_qpres_y(3)*s_qv(3)-s_qpres_y(-3)*s_qv(-3)))
						  + g->DEL*(s_iene_y(4)*s_qv(4)-s_iene_y(-4)*s_qv(-4)
						  + (s_qpres_y(4)*s_qv(4)-s_qpres_y(-4)*s_qv(-4))))*g->dxinv[1];
		}

	#undef  s_iene_x
	#undef  s_iene_y
	#undef	s_qu
	#undef  s_qv
	#undef 	s_qpres_x
	#undef 	s_qpres_y


		/** Z dimension **/
		si = blockIdx.x*blockDim.x+threadIdx.x;
		sj = blockIdx.y*blockDim.y+threadIdx.y;
		if(compute){

			idx = (sk+g->ng)*g->plane_offset_g_padded + (sj+g->ng)*g->pitch_g[0] + si+g->ng;

			double imxp1, imxp2, imxp3, imxp4;
			double imxm1, imxm2, imxm3, imxm4;
			double imyp1, imyp2, imyp3, imyp4;
			double imym1, imym2, imym3, imym4;
			double imzp1, imzp2, imzp3, imzp4;
			double imzm1, imzm2, imzm3, imzm4;
			double ienep1, ienep2, ienep3, ienep4;
			double ienem1, ienem2, ienem3, ienem4;
			double qpresp1, qpresp2, qpresp3, qpresp4;
			double qpresm1, qpresm2, qpresm3, qpresm4;

			unp1 = q[qw*g->comp_offset_g_padded + idx + 1*g->plane_offset_g_padded];
			unp2 = q[qw*g->comp_offset_g_padded + idx + 2*g->plane_offset_g_padded];
			unp3 = q[qw*g->comp_offset_g_padded + idx + 3*g->plane_offset_g_padded];
			unp4 = q[qw*g->comp_offset_g_padded + idx + 4*g->plane_offset_g_padded];
			unm1 = q[qw*g->comp_offset_g_padded + idx - 1*g->plane_offset_g_padded];
			unm2 = q[qw*g->comp_offset_g_padded + idx - 2*g->plane_offset_g_padded];
			unm3 = q[qw*g->comp_offset_g_padded + idx - 3*g->plane_offset_g_padded];
			unm4 = q[qw*g->comp_offset_g_padded + idx - 4*g->plane_offset_g_padded];

			imxp1 = cons[imx*g->comp_offset_g_padded + idx + 1*g->plane_offset_g_padded];
			imxp2 = cons[imx*g->comp_offset_g_padded + idx + 2*g->plane_offset_g_padded];
			imxp3 = cons[imx*g->comp_offset_g_padded + idx + 3*g->plane_offset_g_padded];
			imxp4 = cons[imx*g->comp_offset_g_padded + idx + 4*g->plane_offset_g_padded];
			imxm1 = cons[imx*g->comp_offset_g_padded + idx - 1*g->plane_offset_g_padded];
			imxm2 = cons[imx*g->comp_offset_g_padded + idx - 2*g->plane_offset_g_padded];
			imxm3 = cons[imx*g->comp_offset_g_padded + idx - 3*g->plane_offset_g_padded];
			imxm4 = cons[imx*g->comp_offset_g_padded + idx - 4*g->plane_offset_g_padded];

			imyp1 = cons[imy*g->comp_offset_g_padded + idx + 1*g->plane_offset_g_padded];
			imyp2 = cons[imy*g->comp_offset_g_padded + idx + 2*g->plane_offset_g_padded];
			imyp3 = cons[imy*g->comp_offset_g_padded + idx + 3*g->plane_offset_g_padded];
			imyp4 = cons[imy*g->comp_offset_g_padded + idx + 4*g->plane_offset_g_padded];
			imym1 = cons[imy*g->comp_offset_g_padded + idx - 1*g->plane_offset_g_padded];
			imym2 = cons[imy*g->comp_offset_g_padded + idx - 2*g->plane_offset_g_padded];
			imym3 = cons[imy*g->comp_offset_g_padded + idx - 3*g->plane_offset_g_padded];
			imym4 = cons[imy*g->comp_offset_g_padded + idx - 4*g->plane_offset_g_padded];

			imzp1 = cons[imz*g->comp_offset_g_padded + idx + 1*g->plane_offset_g_padded];
			imzp2 = cons[imz*g->comp_offset_g_padded + idx + 2*g->plane_offset_g_padded];
			imzp3 = cons[imz*g->comp_offset_g_padded + idx + 3*g->plane_offset_g_padded];
			imzp4 = cons[imz*g->comp_offset_g_padded + idx + 4*g->plane_offset_g_padded];
			imzm1 = cons[imz*g->comp_offset_g_padded + idx - 1*g->plane_offset_g_padded];
			imzm2 = cons[imz*g->comp_offset_g_padded + idx - 2*g->plane_offset_g_padded];
			imzm3 = cons[imz*g->comp_offset_g_padded + idx - 3*g->plane_offset_g_padded];
			imzm4 = cons[imz*g->comp_offset_g_padded + idx - 4*g->plane_offset_g_padded];

			ienep1 = cons[iene*g->comp_offset_g_padded + idx + 1*g->plane_offset_g_padded];
			ienep2 = cons[iene*g->comp_offset_g_padded + idx + 2*g->plane_offset_g_padded];
			ienep3 = cons[iene*g->comp_offset_g_padded + idx + 3*g->plane_offset_g_padded];
			ienep4 = cons[iene*g->comp_offset_g_padded + idx + 4*g->plane_offset_g_padded];
			ienem1 = cons[iene*g->comp_offset_g_padded + idx - 1*g->plane_offset_g_padded];
			ienem2 = cons[iene*g->comp_offset_g_padded + idx - 2*g->plane_offset_g_padded];
			ienem3 = cons[iene*g->comp_offset_g_padded + idx - 3*g->plane_offset_g_padded];
			ienem4 = cons[iene*g->comp_offset_g_padded + idx - 4*g->plane_offset_g_padded];

			qpresp1 = q[qpres*g->comp_offset_g_padded + idx + 1*g->plane_offset_g_padded];
			qpresp2 = q[qpres*g->comp_offset_g_padded + idx + 2*g->plane_offset_g_padded];
			qpresp3 = q[qpres*g->comp_offset_g_padded + idx + 3*g->plane_offset_g_padded];
			qpresp4 = q[qpres*g->comp_offset_g_padded + idx + 4*g->plane_offset_g_padded];
			qpresm1 = q[qpres*g->comp_offset_g_padded + idx - 1*g->plane_offset_g_padded];
			qpresm2 = q[qpres*g->comp_offset_g_padded + idx - 2*g->plane_offset_g_padded];
			qpresm3 = q[qpres*g->comp_offset_g_padded + idx - 3*g->plane_offset_g_padded];
			qpresm4 = q[qpres*g->comp_offset_g_padded + idx - 4*g->plane_offset_g_padded];


			flux_irho -=  ( g->ALP*(imzp1-imzm1)
						  + g->BET*(imzp2-imzm2)
						  + g->GAM*(imzp3-imzm3)
						  + g->DEL*(imzp4-imzm4))*g->dxinv[2];

			flux_imx -=   ( g->ALP*(imxp1*unp1-imxm1*unm1)
						  + g->BET*(imxp2*unp2-imxm2*unm2)
						  + g->GAM*(imxp3*unp3-imxm3*unm3)
						  + g->DEL*(imxp4*unp4-imxm4*unm4))*g->dxinv[2];

			flux_imy -=   ( g->ALP*(imyp1*unp1-imym1*unm1)
						  + g->BET*(imyp2*unp2-imym2*unm2)
						  + g->GAM*(imyp3*unp3-imym3*unm3)
						  + g->DEL*(imyp4*unp4-imym4*unm4))*g->dxinv[2];

			flux_imz -=   ( g->ALP*(imzp1*unp1-imzm1*unm1
						  + (qpresp1-qpresm1))
						  + g->BET*(imzp2*unp2-imzm2*unm2
						  + (qpresp2-qpresm2))
						  + g->GAM*(imzp3*unp3-imzm3*unm3
						  + (qpresp3-qpresm3))
						  + g->DEL*(imzp4*unp4-imzm4*unm4
						  + (qpresp4-qpresm4)))*g->dxinv[2];

			flux_iene -=  ( g->ALP*(ienep1*unp1-ienem1*unm1
						  + (qpresp1*unp1-qpresm1*unm1))
						  + g->BET*(ienep2*unp2-ienem2*unm2
						  + (qpresp2*unp2-qpresm2*unm2))
						  + g->GAM*(ienep3*unp3-ienem3*unm3
						  + (qpresp3*unp3-qpresm3*unm3))
						  + g->DEL*(ienep4*unp4-ienem4*unm4
						  + (qpresp4*unp4-qpresm4*unm4)))*g->dxinv[2];

			// Update global memory
			flux[out + irho*g->comp_offset_padded] = flux_irho;
			flux[out + imx *g->comp_offset_padded] = flux_imx;
			flux[out + imy*g->comp_offset_padded] = flux_imy;
			flux[out + imz*g->comp_offset_padded] = flux_imz;
			flux[out + iene*g->comp_offset_padded] = flux_iene;
		}
	}
}

void gpu_hypterm3(
	global_const_t h_const, 	// i: Global struct containing application parameters
	global_const_t *d_const,	// i: Device pointer to global struct containing application paramters
	double *d_cons,				// i:
	double *d_q,				// i:
	double *d_flux				// o: flux
){

	/** d_flux must be set to zero beforehand (in diffterm, etc) **/

	dim3 block_dim_xy_stencil(BLOCK_LARGE, BLOCK_LARGE, 1);
	dim3 grid_dim_xy_stencil(CEIL(h_const.dim[0], BLOCK_LARGE), CEIL(h_const.dim[1], BLOCK_LARGE), CEIL(h_const.dim[2], THREAD_Z));
	gpu_hypterm_xy_stencil_kernel<<<grid_dim_xy_stencil, block_dim_xy_stencil>>>(d_const, d_cons, d_q, d_flux);

}


