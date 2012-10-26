#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "util.h"


#define	BLOCK_DIM	16
#define	Unew(l,i,j,k)	Unew[(l)*g->comp_offset_g + (i)*g->plane_offset_g + (j)*g->dim_g[2] + (k)]
#define	U(l,i,j,k)		U[(l)*g->comp_offset_g + (i)*g->plane_offset_g + (j)*g->dim_g[2] + (k)]
#define	D(l,i,j,k)		D[(l)*g->comp_offset + (i)*g->plane_offset + (j)*g->dim[2] + (k)]
#define	F(l,i,j,k)		F[(l)*g->comp_offset + (i)*g->plane_offset + (j)*g->dim[2] + (k)]

__global__ void gpu_Unew_1_3_kernel(
	global_const_t *g,	// i: Global Constants
	double *Unew,		// o: New U
	double *U,			// i: Old U
	double *D,			// i: difflux
	double *F,			// i: flux
	double dt			// i: dt
){
	int bi,bj,bk;
	int i,j,k,l;
	kernel_const_t *kc = g->kc;

	bj = (blockIdx.x % (kc->gridDim_plane_yz)) / kc->gridDim_z;
	bk = (blockIdx.x % (kc->gridDim_plane_yz)) % kc->gridDim_z;
	bi =  blockIdx.x / (kc->gridDim_plane_yz);
	i = bi;
	j = bj*BLOCK_DIM+threadIdx.y;
	k = bk*BLOCK_DIM+threadIdx.z;

	if(i < g->dim[0] && j < g->dim[1] && k < g->dim[2]){
		FOR(l, 0, g->nc)
			Unew(l,i+g->ng,j+g->ng,k+g->ng) = U(l,i+g->ng,j+g->ng,k+g->ng) + dt*(D(l,i,j,k) + F(l,i,j,k));
	}
}

__global__ void gpu_Unew_2_3_kernel(
	global_const_t *g,	// i: Global Constants
	double *Unew,		// o: New U
	double *U,			// i: Old U
	double *D,			// i: difflux
	double *F,			// i: flux
	double dt			// i: dt
){
	int bi,bj,bk;
	int i,j,k,l;
	kernel_const_t *kc = g->kc;

	bj = (blockIdx.x % (kc->gridDim_plane_yz)) / kc->gridDim_z;
	bk = (blockIdx.x % (kc->gridDim_plane_yz)) % kc->gridDim_z;
	bi =  blockIdx.x / (kc->gridDim_plane_yz);
	i = bi;
	j = bj*BLOCK_DIM+threadIdx.y;
	k = bk*BLOCK_DIM+threadIdx.z;

	if(i < g->dim[0] && j< g->dim[1] && k < g->dim[2]){
		FOR(l, 0, g->nc){
			Unew(l,i+g->ng,j+g->ng,k+g->ng) =
				g->ThreeQuarters *  U(l,i+g->ng,j+g->ng,k+g->ng) +
				g->OneQuarter	 * (Unew(l,i+g->ng,j+g->ng,k+g->ng) + dt*(D(l,i,j,k) + F(l,i,j,k)));
		}
	}
}

__global__ void gpu_Unew_3_3_kernel(
	global_const_t *g,	// i: Global Constants
	double *Unew,		// o: New U
	double *U,			// i: Old U
	double *D,			// i: difflux
	double *F,			// i: flux
	double dt			// i: dt
){
	int bi,bj,bk;
	int i,j,k,l;
	kernel_const_t *kc = g->kc;

	bj = (blockIdx.x % (kc->gridDim_plane_yz)) / kc->gridDim_z;
	bk = (blockIdx.x % (kc->gridDim_plane_yz)) % kc->gridDim_z;
	bi =  blockIdx.x / (kc->gridDim_plane_yz);
	i = bi;
	j = bj*BLOCK_DIM+threadIdx.y;
	k = bk*BLOCK_DIM+threadIdx.z;

	if(i < g->dim[0] && j < g->dim[1] && k < g->dim[2]){
		FOR(l, 0, g->nc){
			U(l,i+g->ng,j+g->ng,k+g->ng) =
				g->OneThird  *  U(l,i+g->ng,j+g->ng,k+g->ng) +
				g->TwoThirds * (Unew(l,i+g->ng,j+g->ng,k+g->ng) + dt*(D(l,i,j,k) + F(l,i,j,k)));
		}
	}
}

#undef	Unew
#undef	U
#undef	D
#undef	F

void gpu_Unew(
	global_const_t h_const,	// i: Global Constants
	global_const_t *d_const,	// i: Device Pointer to Global Constants
	double *d_Unew,		 		// o: New U
	double *d_U,				// i: Old U
	double *d_D,				// i: difflux
	double *d_F,				// i: flux
	double dt,					// i: dt
	int phase					// i: phase
){
	int grid_dim;
	dim3 block_dim(1, BLOCK_DIM, BLOCK_DIM);
	kernel_const_t h_kc;

	h_kc.gridDim_x = h_const.dim[0];
	h_kc.gridDim_y = CEIL(h_const.dim[1], BLOCK_DIM);
	h_kc.gridDim_z = CEIL(h_const.dim[2], BLOCK_DIM);
	h_kc.gridDim_plane_yz = h_kc.gridDim_y * h_kc.gridDim_z;
	grid_dim = h_kc.gridDim_x * h_kc.gridDim_plane_yz;
	cudaMemcpy(h_const.kc, &h_kc, sizeof(kernel_const_t), cudaMemcpyHostToDevice);

	switch(phase){
		case 1:
			gpu_Unew_1_3_kernel<<<grid_dim, block_dim>>>(d_const, d_Unew, d_U, d_D, d_F, dt);
			break;
		case 2:
			gpu_Unew_2_3_kernel<<<grid_dim, block_dim>>>(d_const, d_Unew, d_U, d_D, d_F, dt);
			break;
		case 3:
			gpu_Unew_3_3_kernel<<<grid_dim, block_dim>>>(d_const, d_Unew, d_U, d_D, d_F, dt);
			break;
	}
}

void gpu_advance(
	global_const_t &h_const,	// i: Global constants
	global_const_t *d_const,	// i: Device pointer to global constants
	double *d_U,				// i/o
	double *d_Unew,
	double *d_Q,
	double *d_D,
	double *d_F,
	double &dt					// o
){
	int i;
	double courno, courno_proc;

	//!
	//! multifab_fill_boundary (U)
	//!
	gpu_fill_boundary(h_const, d_const, d_U);

    //!
    //! Calculate primitive variables based on U.
    //!
    //! Also calculate courno so we can set "dt".
    //!
	courno_proc = 1.0E-50;
	gpu_ctoprim(h_const, d_const, d_U, d_Q, courno_proc);

	courno = courno_proc;
	dt = h_const.cfl/courno;
	printf("dt, courno = %le, %le\n", dt, courno);

    //!
    //! Calculate D at time N.
    //!
//    double ****Q;
//    allocate_4D(Q, h_const.dim_g, h_const.nc+1);
//    number_3D(Q[qu], h_const.dim_g);
//    gpu_copy_from_host_4D(d_Q, Q, h_const.dim_g, h_const.nc+1);
//	gpu_diffterm(h_const, d_const, d_Q, d_D);
//	gpu_diffterm_lv1(h_const, d_const, d_Q, d_D);
	gpu_diffterm2(h_const, d_const, d_Q, d_D);
//	gpu_diffterm_lv2(h_const, d_const, d_Q, d_D);

    //!
    //! Calculate F at time N.
    //!
	gpu_hypterm(h_const, d_const, d_U, d_Q, d_F);

    //!
    //! Calculate U at time N+1/3.
    //!
	gpu_Unew(h_const, d_const, d_Unew, d_U, d_D, d_F, dt, 1);

	//!
    //! Sync U^1/3 prior to calculating D & F. -- multifab_fill_boundary(Unew)
    //!
	gpu_fill_boundary(h_const, d_const, d_Unew);

	//!
    //! Calculate primitive variables based on U^1/3.
    //!
	gpu_ctoprim(h_const, d_const, d_Unew, d_Q);

    //!
    //! Calculate D at time N+1/3.
    //!
//	gpu_diffterm(h_const, d_const, d_Q, d_D);
//	gpu_diffterm_lv1(h_const, d_const, d_Q, d_D);
	gpu_diffterm2(h_const, d_const, d_Q, d_D);
//	gpu_diffterm_lv2(h_const, d_const, d_Q, d_D);

	//!
    //! Calculate F at time N+1/3.
    //!
	gpu_hypterm(h_const, d_const, d_Unew, d_Q, d_F);

	//!
    //! Calculate U at time N+2/3.
    //!
	gpu_Unew(h_const, d_const, d_Unew, d_U, d_D, d_F, dt, 2);

	//!
    //! Sync U^2/3 prior to calculating D & F. -- multifab_fill_boundary(Unew)
    //!
	gpu_fill_boundary(h_const, d_const, d_Unew);

    //!
    //! Calculate primitive variables based on U^2/3.
    //!
	gpu_ctoprim(h_const, d_const, d_Unew, d_Q);

    //!
    //! Calculate D at time N+2/3.
    //!
//    gpu_diffterm(h_const, d_const, d_Q, d_D);
//	gpu_diffterm_lv1(h_const, d_const, d_Q, d_D);
	gpu_diffterm2(h_const, d_const, d_Q, d_D);
//	gpu_diffterm_lv2(h_const, d_const, d_Q, d_D);

    //!
    //! Calculate F at time N+2/3.
    //!
	gpu_hypterm(h_const, d_const, d_Unew, d_Q, d_F);

    //!
    //! Calculate U at time N+1.
    //!
	gpu_Unew(h_const, d_const, d_Unew, d_U, d_D, d_F, dt, 3);


}

void advance_test(
	global_const_t h_const, 	// i: Global struct containing application parameters
	global_const_t *d_const,	// i: Device pointer to global struct containing application paramters
	double ****U,
	double ****Unew,
	double ****Q,
	double ****D,
	double ****F,
	double *d_U,
	double *d_Unew,
	double *d_Q,
	double *d_D,
	double *d_F
){
	int i, l, n;
	int nc, dim_g[3];
	double dt, dt2, dx[DIM], cfl, eta, alam;
	double ****U2;
	FILE *fin, *fout;

	nc = h_const.nc;
	FOR(i, 0, DIM)
		dim_g[i] = h_const.dim_g[i];

	// Allocation
	allocate_4D(U2, dim_g, nc);

	// Initiation
	fin = fopen("../testcases/advance_input", "r");
	FOR(l, 0, nc)
		read_3D(fin, U, dim_g, l);

	fscanf(fin, "%le", &dt);
	FOR(i, 0, 3)
		fscanf(fin, "%le", &dx[i]);
	fscanf(fin, "%le", &cfl);
	fscanf(fin, "%le", &eta);
	fscanf(fin, "%le", &alam);
	fclose(fin);

	gpu_copy_from_host_4D(d_U, U, dim_g, 5);
	gpu_advance(h_const, d_const, d_U, d_Unew, d_Q, d_D, d_F, dt);
	gpu_copy_to_host_4D(U, d_U, dim_g, 5);

	fout=fopen("../testcases/advance_output", "r");
	FOR(l, 0, nc)
		read_3D(fout, U2, dim_g, l);
	check_4D_array("U", U, U2, dim_g, nc);

	fscanf(fout, "%le", &dt2);
	check_double(dt, dt2, "dt");
	fclose(fout);
	printf("Correct!\n");

	// Free memory
	free_4D(U2, dim_g, nc);
}

void advance_multistep_test(
	global_const_t h_const, 	// i: Global struct containing application parameters
	global_const_t *d_const,	// i: Device pointer to global struct containing application paramters
	double ****U,
	double ****Unew,
	double ****Q,
	double ****D,
	double ****F,
	double *d_U,
	double *d_Unew,
	double *d_Q,
	double *d_D,
	double *d_F
){
	int i, l, n;
	int nc, dim_g[3];
	double dt, dt2, dx[DIM], cfl, eta, alam;
	double ****U2;
	FILE *fin, *fout;

	nc = h_const.nc;
	FOR(i, 0, DIM)
		dim_g[i] = h_const.dim_g[i];

	// Allocation
	allocate_4D(U2, dim_g, nc);

	// Initiation
	fin = fopen("../testcases/multistep_input", "r");
	FOR(l, 0, nc)
		read_3D(fin, U, dim_g, l);
	fclose(fin);

	gpu_copy_from_host_4D(d_U, U, dim_g, 5);

	FOR(i, 0, h_const.nsteps)
		gpu_advance(h_const, d_const, d_U, d_Unew, d_Q, d_D, d_F, dt);

	gpu_copy_to_host_4D(U, d_U, dim_g, 5);

	fout=fopen("../testcases/multistep_output", "r");
	FOR(l, 0, nc)
		read_3D(fout, U2, dim_g, l);
	check_4D_array("U", U, U2, dim_g, nc);

	fscanf(fout, "%le", &dt2);
	check_double(dt, dt2, "dt");
	fclose(fout);
	printf("Correct!\n");

	// Free memory
	free_4D(U2, dim_g, nc);
}
