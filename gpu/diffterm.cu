#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "helper_functions.h"

#define	BLOCK_DIM	16
#define	BLOCK_DIM_G	8
#define	q(i, comp)	s_q[comp][threadIdx.x+g->ng+i][threadIdx.z]

__device__ double values[9];
__device__ double plane[BLOCK_DIM_G+NG+NG][BLOCK_DIM];

__global__ void gpu_diffterm_x_stencil_kernel(
	global_const_t *g,			// i: Global struct containing application parameters
	double *q,					// i:
	double *d_flux				// o:
){
	int idx, tidx, tidz;
	int bi, bj, bk, si, sj, sk;
	__shared__ double  s_q[s_qend][BLOCK_DIM_G+NG+NG][BLOCK_DIM];

	// Load to shared mem
	bi = (blockIdx.x % (g->gridDim_plane_xz)) / g->gridDim_z;
	bk = (blockIdx.x % (g->gridDim_plane_xz)) % g->gridDim_z;
	bj =  blockIdx.x / (g->gridDim_plane_xz);
	si = bi*blockDim.x+threadIdx.x;
	sj = bj*blockDim.y+threadIdx.y; // = bj
	sk = bk*blockDim.z+threadIdx.z;

	tidx = threadIdx.x;
	tidz = threadIdx.z;
	while( tidx < g->blockDim_x_g && si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){

		idx = si*g->plane_offset_g + sj*g->dim_g[2] + sk;

		s_q[s_qu][tidx][tidz]  =  q[idx + qu*g->comp_offset_g];
		s_q[s_qv][tidx][tidz]  =  q[idx + qv*g->comp_offset_g];
		s_q[s_qw][tidx][tidz]  =  q[idx + qw*g->comp_offset_g];
		s_q[s_qt][tidx][tidz]  =  q[idx + qt*g->comp_offset_g];

		tidx += blockDim.x;
		si   += blockDim.x;
	}
	__syncthreads();

	si = bi*blockDim.x+threadIdx.x;
	idx = (si+g->ng)*g->plane_offset_g + sj*g->dim_g[2] + sk;
	if(si < g->dim[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){

		g->temp[UX][idx] =  ( ALP*(q(1,s_qu)-q(-1,s_qu))
							+ BET*(q(2,s_qu)-q(-2,s_qu))
							+ GAM*(q(3,s_qu)-q(-3,s_qu))
							+ DEL*(q(4,s_qu)-q(-4,s_qu)))*g->dxinv[0];

		g->temp[VX][idx] = 	( ALP*(q(1,s_qv)-q(-1,s_qv))
							+ BET*(q(2,s_qv)-q(-2,s_qv))
							+ GAM*(q(3,s_qv)-q(-3,s_qv))
							+ DEL*(q(4,s_qv)-q(-4,s_qv)))*g->dxinv[0];

		g->temp[WX][idx] =	( ALP*(q(1,s_qw)-q(-1,s_qw))
							+ BET*(q(2,s_qw)-q(-2,s_qw))
							+ GAM*(q(3,s_qw)-q(-3,s_qw))
							+ DEL*(q(4,s_qw)-q(-4,s_qw)))*g->dxinv[0];

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
	double *d_flux				// o:
){
	int idx, idx_g, tidx, tidz;
	int bi, bj, bk, si, sj, sk;
	__shared__ double  vy[BLOCK_DIM_G+NG+NG][BLOCK_DIM];
	__shared__ double  wz[BLOCK_DIM_G+NG+NG][BLOCK_DIM];

	// Load to shared mem
	bi = (blockIdx.x % (g->gridDim_plane_xz)) / g->gridDim_z;
	bk = (blockIdx.x % (g->gridDim_plane_xz)) % g->gridDim_z;
	bj =  blockIdx.x / (g->gridDim_plane_xz);
	si = bi*blockDim.x+threadIdx.x;
	sj = bj*blockDim.y+threadIdx.y; // = bj
	sk = bk*blockDim.z+threadIdx.z;

	tidx = threadIdx.x;
	tidz = threadIdx.z;
	while( tidx < g->blockDim_x_g && si < g->dim_g[0] && sj < g->dim[1] && sk < g->dim[2]){

		idx = si*g->plane_offset_g + (sj+g->ng)*g->dim_g[2] + (sk+g->ng);
		vy[tidx][tidz]  =  g->temp[VY][idx];
		tidx += blockDim.x;
		si   += blockDim.x;
	}

	tidx = threadIdx.x;
	si = bi*blockDim.x+threadIdx.x;
	while( tidx < g->blockDim_x_g && si < g->dim_g[0] && sj < g->dim[1] && sk < g->dim[2]){

		idx = si*g->plane_offset_g + (sj+g->ng)*g->dim_g[2] + (sk+g->ng);
		wz[tidx][tidz]  =  g->temp[WZ][idx];

		tidx += blockDim.x;
		si   += blockDim.x;
	}
	__syncthreads();

#define	vy(i)	vy[threadIdx.x+g->ng+i][threadIdx.z]
#define	wz(i)	wz[threadIdx.x+g->ng+i][threadIdx.z]
	si = bi*blockDim.x+threadIdx.x;
	if(si==0 && sj==0 && sk==0){
		int i, k;
		FOR(i, 0, BLOCK_DIM){
			FOR(k, 0, BLOCK_DIM)
				plane[i][k] = wz[i][k];
		}
	}

	idx 	= si*g->plane_offset + sj*g->dim[2] + sk;
	idx_g 	= (si+g->ng)*g->plane_offset_g + (sj+g->ng)*g->dim_g[2] + (sk+g->ng);
	if(si < g->dim[0] && sj < g->dim[1] && sk < g->dim[2]){

		g->temp[VYX][idx] = ( ALP*(vy(1)-vy(-1))
							+ BET*(vy(2)-vy(-2))
							+ GAM*(vy(3)-vy(-3))
							+ DEL*(vy(4)-vy(-4)))*g->dxinv[0];

		g->temp[WZX][idx] = ( ALP*(wz(1)-wz(-1))
							+ BET*(wz(2)-wz(-2))
							+ GAM*(wz(3)-wz(-3))
							+ DEL*(wz(4)-wz(-4)))*g->dxinv[0];

		d_flux[idx + imx*g->comp_offset] =  	 g->eta *
											   ( g->FourThirds*g->temp[UXX][idx_g] +
															   g->temp[UYY][idx_g] +
															   g->temp[UZZ][idx_g] +
												 g->OneThird *(g->temp[VYX][idx] + g->temp[WZX][idx]));

		if(si==0 && sj==0 && sk==0){
			values[0] = g->temp[WZX][idx];
			values[1] = wz(1);
			values[2] = wz(2);
			values[3] = wz(3);
			values[4] = wz(4);
			values[5] = wz(-1);
			values[6] = wz(-2);
			values[7] = wz(-3);
			values[8] = wz(-4);
		}
	}
#undef	vy
#undef	wz
}
#undef	q

__global__ void gpu_diffterm_yz_stencil_kernel(
	global_const_t *g,			// i: Global struct containing application parameters
	double *q,					// i:
	double *d_flux				// o:
){
	int idx, tidy, tidz;
	int bi, bj, bk, si, sj, sk;
	__shared__ double  s_q[s_qend][BLOCK_DIM_G+NG+NG][BLOCK_DIM_G+NG+NG];

	// Load to shared mem
	bj = (blockIdx.x % (g->gridDim_plane_yz)) / g->gridDim_z;
	bk = (blockIdx.x % (g->gridDim_plane_yz)) % g->gridDim_z;
	bi =  blockIdx.x / (g->gridDim_plane_yz);
	si = bi;
	sj = bj*BLOCK_DIM_G+threadIdx.y;
	sk = bk*BLOCK_DIM_G+threadIdx.z;

	tidy = threadIdx.y;
	tidz = threadIdx.z;
	idx = si*g->plane_offset_g + sj*g->dim_g[2] + sk;
	if(si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim_g[2]){
		s_q[s_qu][tidy][tidz]  =  q[idx + qu*g->comp_offset_g];
		s_q[s_qv][tidy][tidz]  =  q[idx + qv*g->comp_offset_g];
		s_q[s_qw][tidy][tidz]  =  q[idx + qw*g->comp_offset_g];
		s_q[s_qt][tidy][tidz]  =  q[idx + qt*g->comp_offset_g];
	}
	__syncthreads();

	sj = bj*BLOCK_DIM_G+threadIdx.y;
	sk = bk*BLOCK_DIM_G+threadIdx.z;

	if(threadIdx.y < BLOCK_DIM_G && threadIdx.z < BLOCK_DIM_G){
#define	q(i, comp)	s_q[comp][threadIdx.y+g->ng+i][threadIdx.z]

		idx = si*g->plane_offset_g + (sj+g->ng)*g->dim_g[2] + sk;
		if(si < g->dim_g[0] && sj < g->dim[1] && sk < g->dim_g[2]){

			g->temp[UY][idx] =  ( ALP*(q(1,s_qu)-q(-1,s_qu))
								+ BET*(q(2,s_qu)-q(-2,s_qu))
								+ GAM*(q(3,s_qu)-q(-3,s_qu))
								+ DEL*(q(4,s_qu)-q(-4,s_qu)))*g->dxinv[1];

			g->temp[VY][idx] = 	( ALP*(q(1,s_qv)-q(-1,s_qv))
								+ BET*(q(2,s_qv)-q(-2,s_qv))
								+ GAM*(q(3,s_qv)-q(-3,s_qv))
								+ DEL*(q(4,s_qv)-q(-4,s_qv)))*g->dxinv[1];

			g->temp[WY][idx] =	( ALP*(q(1,s_qw)-q(-1,s_qw))
								+ BET*(q(2,s_qw)-q(-2,s_qw))
								+ GAM*(q(3,s_qw)-q(-3,s_qw))
								+ DEL*(q(4,s_qw)-q(-4,s_qw)))*g->dxinv[1];

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

			g->temp[TYY][idx] = ( g->CENTER*q(0,s_qt)
								+ g->OFF1*(q(1,s_qt)+q(-1,s_qt))
								+ g->OFF2*(q(2,s_qt)+q(-2,s_qt))
								+ g->OFF3*(q(3,s_qt)+q(-3,s_qt))
								+ g->OFF4*(q(4,s_qt)+q(-4,s_qt)))*SQR(g->dxinv[1]);
		}

#undef	q
#define	q(i, comp)	s_q[comp][threadIdx.y][threadIdx.z+g->ng+i]

		idx = si*g->plane_offset_g + sj*g->dim_g[2] + (sk+g->ng);
		if(si < g->dim_g[0] && sj < g->dim_g[1] && sk < g->dim[2]){

			g->temp[UZ][idx] =  ( ALP*(q(1,s_qu)-q(-1,s_qu))
								+ BET*(q(2,s_qu)-q(-2,s_qu))
								+ GAM*(q(3,s_qu)-q(-3,s_qu))
								+ DEL*(q(4,s_qu)-q(-4,s_qu)))*g->dxinv[2];

			g->temp[VZ][idx] = 	( ALP*(q(1,s_qv)-q(-1,s_qv))
								+ BET*(q(2,s_qv)-q(-2,s_qv))
								+ GAM*(q(3,s_qv)-q(-3,s_qv))
								+ DEL*(q(4,s_qv)-q(-4,s_qv)))*g->dxinv[2];

			g->temp[WZ][idx] =	( ALP*(q(1,s_qw)-q(-1,s_qw))
								+ BET*(q(2,s_qw)-q(-2,s_qw))
								+ GAM*(q(3,s_qw)-q(-3,s_qw))
								+ DEL*(q(4,s_qw)-q(-4,s_qw)))*g->dxinv[2];

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

			g->temp[TZZ][idx] = ( g->CENTER*q(0,s_qt)
								+ g->OFF1*(q(1,s_qt)+q(-1,s_qt))
								+ g->OFF2*(q(2,s_qt)+q(-2,s_qt))
								+ g->OFF3*(q(3,s_qt)+q(-3,s_qt))
								+ g->OFF4*(q(4,s_qt)+q(-4,s_qt)))*SQR(g->dxinv[2]);
		}

#undef 	q
	}
}

void gpu_diffterm(
	global_const_t h_const, 	// i: Global struct containing applicatino parameters
	global_const_t *d_const,	// i: Device pointer to global struct containing application paramters
	double *d_q,				// i:
	double *d_flux				// o:
){
	int grid_dim;

	dim3 block_dim_x_stencil(BLOCK_DIM_G, 1, BLOCK_DIM);
	h_const.gridDim_x = CEIL(h_const.dim[0], BLOCK_DIM_G);
	h_const.gridDim_y = h_const.dim_g[1];
	h_const.gridDim_z = CEIL(h_const.dim_g[2], BLOCK_DIM);
	h_const.gridDim_plane_xz = h_const.gridDim_x * h_const.gridDim_z;
    h_const.blockDim_x_g = BLOCK_DIM_G + h_const.ng + h_const.ng;
    grid_dim = h_const.gridDim_plane_xz * h_const.gridDim_y;
    cudaMemcpy(d_const, &h_const, sizeof(global_const_t), cudaMemcpyHostToDevice);

	gpu_diffterm_x_stencil_kernel<<<grid_dim, block_dim_x_stencil>>>(d_const, d_q, d_flux);

	dim3 block_dim_yz_stencil(1, BLOCK_DIM, BLOCK_DIM);
	h_const.gridDim_x = h_const.dim_g[0];
	h_const.gridDim_y = CEIL(h_const.dim_g[1], BLOCK_DIM_G);
	h_const.gridDim_z = CEIL(h_const.dim_g[2], BLOCK_DIM_G);
	h_const.gridDim_plane_yz = h_const.gridDim_y * h_const.gridDim_z;
    h_const.blockDim_y_g = BLOCK_DIM_G + h_const.ng + h_const.ng;
    h_const.blockDim_z_g = BLOCK_DIM_G + h_const.ng + h_const.ng;
	grid_dim = h_const.gridDim_x * h_const.gridDim_plane_yz;
    cudaMemcpy(d_const, &h_const, sizeof(global_const_t), cudaMemcpyHostToDevice);

    printf("dim_g = %d %d %d\n", h_const.dim_g[0], h_const.dim_g[1], h_const.dim_g[2]);
	printf("griddim = %d %d %d\n", h_const.gridDim_x, h_const.gridDim_y, h_const.gridDim_z);

	gpu_diffterm_yz_stencil_kernel<<<grid_dim, block_dim_yz_stencil>>>(d_const, d_q, d_flux);

	h_const.gridDim_x = CEIL(h_const.dim[0], BLOCK_DIM_G);
	h_const.gridDim_y = h_const.dim[1];
	h_const.gridDim_z = CEIL(h_const.dim[2], BLOCK_DIM);
	h_const.gridDim_plane_xz = h_const.gridDim_x * h_const.gridDim_z;
    h_const.blockDim_x_g = BLOCK_DIM_G + h_const.ng + h_const.ng;
    grid_dim = h_const.gridDim_plane_xz * h_const.gridDim_y;
    cudaMemcpy(d_const, &h_const, sizeof(global_const_t), cudaMemcpyHostToDevice);

	gpu_diffterm_x_stencil_kernel_lv2<<<grid_dim, block_dim_x_stencil>>>(d_const, d_q, d_flux);

}

void diffterm_test(
	global_const_t h_const, // i: Global struct containing applicatino parameters
	global_const_t *d_const	// i: Device pointer to global struct containing application paramters
){
	int dim_g[3], dim[3];
	int i, l;

	int lo[3], hi[3], ng=4;
	double dx[3], eta, alam;
	double ****q, ****difflux;
	double ***ux, ***vx, ***wx;
	double ***uy, ***vy, ***wy;
	double ***uz, ***vz, ***wz;
	double ***vyx, ***wzx;

	int lo2[3], hi2[3], ng2=4;
	double dx2[3], eta2, alam2;
	double ****q2, ****difflux2;
	double ***ux2, ***vx2, ***wx2;
	double ***uy2, ***vy2, ***wy2;
	double ***uz2, ***vz2, ***wz2;
	double ***vyx2, ***wzx2;

	double *d_q,*d_flux;
	double *d_ux, *d_vx, *d_wx, *d_uy, *d_vy, *d_wy, *d_uz, *d_vz, *d_wz;
	double *d_uxx, *d_uyy, *d_uzz, *d_vyx, *d_wzx;
	double *d_vxx, *d_vyy, *d_vzz, *d_uxy, *d_wzy;
	double *d_wxx, *d_wyy, *d_wzz, *d_uxz, *d_vyz;

	FILE *fin 	= fopen("../testcases/diffterm_input", "r");
	FILE *fout 	= fopen("../testcases/diffterm_output", "r");
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
		dim_g[i] 	= hi[i]-lo[i]+1 + 2*ng;
		dim[i] = hi[i]-lo[i]+1;
	}

	allocate_3D(ux, 	dim_g);		allocate_3D(ux2, 	dim_g);
	allocate_3D(vx, 	dim_g);		allocate_3D(vx2, 	dim_g);
	allocate_3D(wx, 	dim_g);		allocate_3D(wx2, 	dim_g);

	allocate_3D(uy, 	dim_g);		allocate_3D(uy2, 	dim_g);
	allocate_3D(vy, 	dim_g);		allocate_3D(vy2, 	dim_g);
	allocate_3D(wy, 	dim_g);		allocate_3D(wy2, 	dim_g);

	allocate_3D(uz, 	dim_g);		allocate_3D(uz2, 	dim_g);
	allocate_3D(vz, 	dim_g);		allocate_3D(vz2, 	dim_g);
	allocate_3D(wz, 	dim_g);		allocate_3D(wz2, 	dim_g);

	allocate_3D(vyx,	dim);		allocate_3D(vyx2, 	dim);
	allocate_3D(wzx,	dim);		allocate_3D(wzx2, 	dim);

	allocate_4D(q, 		 	dim_g,  6); 	// [40][40][40][6]
	allocate_4D(difflux, 	dim, 5); 	// [32][32][32][5]
	allocate_4D(q2, 	 	dim_g,  6); 	// [40][40][40][6]
	allocate_4D(difflux2, 	dim, 5); 	// [32][32][32][5]

	gpu_allocate_4D(d_q, 	dim_g, 6);
	gpu_allocate_4D(d_flux, dim, 5);

	FOR(i, 0, MAX_TEMP)
		gpu_allocate_3D(h_const.temp[i], dim_g);

	FOR(l, 0, 6)
		read_3D(fin, q, dim_g, l);
	FOR(l, 0, 5)
		read_3D(fin, difflux, dim, l);

	fscanf(fin, "%le %le", &eta, &alam);
	fclose(fin);

	gpu_copy_from_host_4D(d_q, q, dim_g, 6);
	gpu_copy_from_host_4D(d_flux, difflux, dim, 5);

	printf("Applying diffterm()...\n");
	diffterm(lo, hi, ng, dx, q, difflux, eta, alam, ux, vx, wx, uy, vy, wy, uz, vz, wz, vyx, wzx);
	gpu_diffterm(h_const, d_const, d_q, d_flux);

//	gpu_copy_to_host_4D(q, d_q, dim_g, 6);
	gpu_copy_to_host_4D(difflux, d_flux, dim, 5);
	gpu_copy_to_host_3D(ux2, h_const.temp[UX], dim_g);
	gpu_copy_to_host_3D(vx2, h_const.temp[VX], dim_g);
	gpu_copy_to_host_3D(wx2, h_const.temp[WX], dim_g);

	gpu_copy_to_host_3D(uy2, h_const.temp[UY], dim_g);
	gpu_copy_to_host_3D(vy2, h_const.temp[VY], dim_g);
	gpu_copy_to_host_3D(wy2, h_const.temp[WY], dim_g);

	gpu_copy_to_host_3D(uz2, h_const.temp[UZ], dim_g);
	gpu_copy_to_host_3D(vz2, h_const.temp[VZ], dim_g);
	gpu_copy_to_host_3D(wz2, h_const.temp[WZ], dim_g);

	gpu_copy_to_host_3D(vyx2, h_const.temp[VYX], dim);
	gpu_copy_to_host_3D(wzx2, h_const.temp[WZX], dim);


	double vals[9];
	cudaMemcpyFromSymbol(vals, values, 9*sizeof(double));
	printf("--------------\n");
	FOR(i, 0, 9){
		printf("%le\n", vals[i]);
	}
	printf("--------------\n");
	printf("vy[0][0][0] = %le\n", vy2[ng][ng][ng]);
	printf("wz[0][0][0] = %le\n", wz2[ng][ng][ng]);

	double h_plane[BLOCK_DIM_G+NG+NG][BLOCK_DIM];
	cudaMemcpyFromSymbol(h_plane[0], plane, BLOCK_DIM*BLOCK_DIM*sizeof(double));
	int j,k;
//	FOR(i, 0, BLOCK_DIM_G+NG+NG){
//		FOR(k, 0, BLOCK_DIM)
//			printf("%12.4le ", h_plane[i][k]);
//		printf("\n");
//	}
//	printf("=============================\n");
//	FOR(i, 0, h_const.dim_g[0]){
//		FOR(k, ng, h_const.dim[2]+ng)
//			printf("%12.4le ", wz2[i][ng][k]);
//		printf("\n");
//	}

	printf("checking ux, vx, wx...\n");
	FOR(i, ng, dim[0]+ng){
		FOR(j, 0, dim_g[1]){
			FOR(k, 0, dim_g[2]){
				if(!FEQ(ux[i][j][k], ux2[i][j][k])){
					printf("ux2[%d][%d][%d] = %le != %le = ux[%d][%d][%d]\n",
						i,j,k,ux2[i][j][k], ux[i][j][k], i,j,k);
					printf("diff = %le\n", ux2[i][j][k]-ux[i][j][k]);
					exit(1);
				}
				if(!FEQ(vx[i][j][k], vx2[i][j][k])){
					printf("vx2[%d][%d][%d] = %le != %le = vx[%d][%d][%d]\n",
						i,j,k,vx2[i][j][k], vx[i][j][k], i,j,k);
					printf("diff = %le\n", vx2[i][j][k]-vx[i][j][k]);
					exit(1);
				}
				if(!FEQ(wx[i][j][k], wx2[i][j][k])){
					printf("wx2[%d][%d][%d] = %le != %le = wx[%d][%d][%d]\n",
						i,j,k,wx2[i][j][k], wx[i][j][k], i,j,k);
					printf("diff = %le\n", wx2[i][j][k]-wx[i][j][k]);
					exit(1);
				}
			}
		}
	}
	printf("ux, vx, wx is correct!\n");

	printf("checking uy\n");
	FOR(i, 0, dim_g[0]){
		FOR(j, ng, dim[1]+ng){
			FOR(k, 0, dim_g[2]){
				if(!FEQ(uy[i][j][k], uy2[i][j][k])){
					printf("uy2[%d][%d][%d] = %le != %le = uy[%d][%d][%d]\n",
						i,j,k,uy2[i][j][k], uy[i][j][k], i,j,k);
					printf("diff = %le\n", uy2[i][j][k]-uy[i][j][k]);
					exit(1);
				}
				if(!FEQ(vy[i][j][k], vy2[i][j][k])){
					printf("vy2[%d][%d][%d] = %le != %le = vy[%d][%d][%d]\n",
						i,j,k,vy2[i][j][k], vy[i][j][k], i,j,k);
					printf("diff = %le\n", vy2[i][j][k]-vy[i][j][k]);
					exit(1);
				}
				if(!FEQ(wy[i][j][k], wy2[i][j][k])){
					printf("wy2[%d][%d][%d] = %le != %le = wy[%d][%d][%d]\n",
						i,j,k,wy2[i][j][k], wy[i][j][k], i,j,k);
					printf("diff = %le\n", wy2[i][j][k]-wy[i][j][k]);
					exit(1);
				}
			}
		}
	}
	printf("uy, vy, wy is correct!\n");

	printf("checking uz\n");
	FOR(i, 0, dim_g[0]){
		FOR(j, 0, dim_g[1]){
			FOR(k, ng, dim[2]+ng){
				if(!FEQ(uz[i][j][k], uz2[i][j][k])){
					printf("uz2[%d][%d][%d] = %le != %le = uz[%d][%d][%d]\n",
						i,j,k,uz2[i][j][k], uz[i][j][k], i,j,k);
					printf("diff = %le\n", uz2[i][j][k]-uz[i][j][k]);
					exit(1);
				}
				if(!FEQ(vz[i][j][k], vz2[i][j][k])){
					printf("vz2[%d][%d][%d] = %le != %le = vz[%d][%d][%d]\n",
						i,j,k,vz2[i][j][k], vz[i][j][k], i,j,k);
					printf("diff = %le\n", vz2[i][j][k]-vz[i][j][k]);
					exit(1);
				}
				if(!FEQ(wz[i][j][k], wz2[i][j][k])){
					printf("wz2[%d][%d][%d] = %le != %le = wz[%d][%d][%d]\n",
						i,j,k,wz2[i][j][k], wz[i][j][k], i,j,k);
					printf("diff = %le\n", wz2[i][j][k]-wz[i][j][k]);
					exit(1);
				}
			}
		}
	}
	printf("uz, vz, wz is correct!\n");

	printf("checking vyx, wzx\n");
	FOR(i, 0, dim[0]){
		FOR(j, 0, dim[1]){
			FOR(k, 0, dim[2]){
				if(!FEQ(vyx[i][j][k], vyx2[i][j][k])){
					printf("vyx2[%d][%d][%d] = %le != %le = vyx[%d][%d][%d]\n",
						i,j,k,vyx2[i][j][k], vyx[i][j][k], i,j,k);
					printf("diff = %le\n", vyx2[i][j][k]-vyx[i][j][k]);
					exit(1);
				}
				if(!FEQ(wzx[i][j][k], wzx2[i][j][k])){
					printf("wzx2[%d][%d][%d] = %le != %le = wzx[%d][%d][%d]\n",
						i,j,k,wzx2[i][j][k], wzx[i][j][k], i,j,k);
					printf("diff = %le\n", wzx2[i][j][k]-wzx[i][j][k]);
					exit(1);
				}
			}
		}
	}
	printf("vyx, wzx are correct!\n");

	// Scanning output to check
	fscanf(fout, "%d %d %d\n", &lo2[0], &lo2[1], &lo2[2]);
	fscanf(fout, "%d %d %d\n", &hi2[0], &hi2[1], &hi2[2]);
	fscanf(fout, "%d\n", &ng2);
	fscanf(fout, "%le %le %le\n", &dx2[0], &dx2[1], &dx2[2]);

	FOR(l, 0, 6)
		read_3D(fout, q2, dim_g, l);
	FOR(l, 0, 5)
		read_3D(fout, difflux2, dim, l);

	fscanf(fout, "%le %le", &eta2, &alam2);
	fclose(fout);

	// Checking...
//	check_lo_hi_ng_dx(lo, hi, ng, dx, lo2, hi2, ng2, dx2);
//	check_4D_array("q", q, q2, dim_g, 6);
	printf("checking difflux\n");
	FOR(i, 0, dim[0]){
		FOR(j, 0, dim[1]){
			FOR(k, 0, dim[2]){
				if(!FEQ(difflux[imx][i][j][k], difflux2[imx][i][j][k])){
					printf("difflux2[imx][%d][%d][%d] = %le != %le = difflux[imx][%d][%d][%d]\n",
						i,j,k,difflux2[imx][i][j][k], difflux[imx][i][j][k], i,j,k);
					printf("diff = %le\n", difflux2[imx][i][j][k]-difflux[imx][i][j][k]);
					exit(1);
				}
			}
		}
	}
	printf("difflux[imx] correct!\n");
//	check_4D_array("difflux", difflux, difflux2, dim, 5);
//	check_double(eta,  eta2,  "eta");
//	check_double(alam, alam2, "alam");

	FOR(i, 0, MAX_TEMP)
		gpu_free_3D(h_const.temp[i]);

	gpu_free_4D(d_q);
	gpu_free_4D(d_flux);

	free_4D(q,  dim_g, 6);	free_4D(difflux,  dim, 5);
	free_4D(q2, dim_g, 6);	free_4D(difflux2, dim, 5);

	free_3D(ux,  dim_g);	free_3D(ux2, dim_g);
	free_3D(vx,  dim_g);	free_3D(vx2, dim_g);
	free_3D(wx,  dim_g);	free_3D(wx2, dim_g);

	free_3D(uy,  dim_g);	free_3D(uy2, dim_g);
	free_3D(vy,  dim_g);	free_3D(vy2, dim_g);
	free_3D(wy,  dim_g);	free_3D(wy2, dim_g);

	free_3D(uz,  dim_g);	free_3D(uz2, dim_g);
	free_3D(vz,  dim_g);	free_3D(vz2, dim_g);
	free_3D(wz,  dim_g);	free_3D(wz2, dim_g);

	free_3D(vyx, dim);
	free_3D(wzx, dim);

	printf("Correct!\n");
}
