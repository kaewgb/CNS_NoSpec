#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "helper_functions.h"

__device__ kernel_const_t kc;

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

	bj = (blockIdx.x % (kc.gridDim_plane_yz)) / kc.gridDim_z;
	bk = (blockIdx.x % (kc.gridDim_plane_yz)) % kc.gridDim_z;
	bi =  blockIdx.x / (kc.gridDim_plane_yz);
	i = bi;
	j = bj*BLOCK_DIM+threadIdx.y;
	k = bk*BLOCK_DIM+threadIdx.z;

	if(i < g->dim[0] && j < g->dim[1] && k < g->dim[2]){
		FOR(l, 0, NC)
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

	bj = (blockIdx.x % (kc.gridDim_plane_yz)) / kc.gridDim_z;
	bk = (blockIdx.x % (kc.gridDim_plane_yz)) % kc.gridDim_z;
	bi =  blockIdx.x / (kc.gridDim_plane_yz);
	i = bi;
	j = bj*BLOCK_DIM+threadIdx.y;
	k = bk*BLOCK_DIM+threadIdx.z;

	if(i < g->dim[0] && j< g->dim[1] && k < g->dim[2]){
		FOR(l, 0, NC){
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

	bj = (blockIdx.x % (kc.gridDim_plane_yz)) / kc.gridDim_z;
	bk = (blockIdx.x % (kc.gridDim_plane_yz)) % kc.gridDim_z;
	bi =  blockIdx.x / (kc.gridDim_plane_yz);
	i = bi;
	j = bj*BLOCK_DIM+threadIdx.y;
	k = bk*BLOCK_DIM+threadIdx.z;

	if(i < g->dim[0] && j < g->dim[1] && k < g->dim[2]){
		FOR(l, 0, NC){
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
	global_const_t &h_const,	// i: Global Constants
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
	cudaMemcpyToSymbol(kc, &h_kc, sizeof(kernel_const_t));

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
	double &dt					// o
){
	int lo[3], hi[3], i, j, k, l, n, nc, ng;
	double courno, courno_proc;
//	double ****D, ****F, ****Unew, ****Q;
//	double ****Q2, ****D2, ****F2, ****Unew2, ****U2;

	// GPU variables
	double *d_Unew, *d_Q, *d_D, *d_F;

    // Some arithmetic constants.
//    double OneThird      = 1.E0/3.E0;
//    double TwoThirds     = 2.E0/3.E0;
//    double OneQuarter    = 1.E0/4.E0;
//    double ThreeQuarters = 3.E0/4.E0;

	nc = NC; // ncomp(U)
	ng = NG; // nghost(U)

	int dim[3], dim_g[3];
	FOR(i, 0, 3){
		dim[i] = h_const.dim[i];
		dim_g[i] = h_const.dim_g[i];
	}

	lo[0] = lo[1] = lo[2] = NG;
	hi[0] = hi[1] = hi[2] = NCELLS-1+NG;

	// Allocation
//	allocate_4D(D, dim, nc);
//	allocate_4D(D2, dim, nc);
//	allocate_4D(F, dim, nc);
//	allocate_4D(F2, dim, nc);
//	allocate_4D(Q, dim_g, nc+1);
//	allocate_4D(Q2, dim_g, nc+1);
//	allocate_4D(Unew, dim_g, nc);
//	allocate_4D(Unew2, dim_g, nc);
//	allocate_4D(U2, dim_g, nc);

	printf("Allocating...\n");
	gpu_allocate_4D(d_Unew, dim_g, 	5);
	gpu_allocate_4D(d_Q, 	dim_g, 	6);
	gpu_allocate_4D(d_D, 	dim, 	5);
	gpu_allocate_4D(d_F, 	dim, 	5);

	FOR(i, 0, MAX_TEMP)
		gpu_allocate_3D(h_const.temp[i], dim_g);
	printf("%d elements\n", MAX_TEMP*dim_g[0]*dim_g[1]*dim_g[2]);
	printf("%d sub-elements\n", dim_g[0]*dim_g[1]*dim_g[2]);
	//
	// multifab_fill_boundary(U)
	//
//	fill_boundary(U, dim, dim_g);
	printf("fill boundary..\n");
	gpu_fill_boundary(h_const, d_const, d_U);

    //!
    //! Calculate primitive variables based on U.
    //!
    //! Also calculate courno so we can set "dt".
    //!
    printf("ctoprim..\n");
	courno_proc = 1.0E-50;
//	ctoprim(lo, hi, U, Q, dx, ng, courno_proc);
	gpu_ctoprim(h_const, d_const, d_U, d_Q, courno_proc);

	courno = courno_proc;
	dt = h_const.cfl/courno;
	printf("dt, courno = %le, %le\n", dt, courno);

    //!
    //! Calculate D at time N.
    //!
//	diffterm(lo, hi, ng, dx, Q, D, eta, alam);
	gpu_diffterm(h_const, d_const, d_Q, d_D);


    //!
    //! Calculate F at time N.
    //!
//	hypterm(lo, hi, ng, dx, U, Q, F);
	gpu_hypterm(h_const, d_const, d_U, d_Q, d_F);

    //!
    //! Calculate U at time N+1/3.
    //!
//	FOR(i, 0, dim[0]){
//		FOR(j, 0, dim[1]){
//			FOR(k, 0, dim[2]){
//				FOR(l, 0, nc)
//					Unew[i+NG][j+NG][k+NG][l] = U[i+NG][j+NG][k+NG][l] + dt*(D[i][j][k][l] + F[i][j][k][l]);
//			}
//		}
//	}
	gpu_Unew(h_const, d_const, d_Unew, d_U, d_D, d_F, dt, 1);

	//!
    //! Sync U^1/3 prior to calculating D & F. -- multifab_fill_boundary(Unew)
    //!
//	fill_boundary(Unew, dim, dim_g);
	gpu_fill_boundary(h_const, d_const, d_Unew);

	//!
    //! Calculate primitive variables based on U^1/3.
    //!
//	ctoprim(lo, hi, Unew, Q, dx, ng);
	gpu_ctoprim(h_const, d_const, d_Unew, d_Q);

    //!
    //! Calculate D at time N+1/3.
    //!
//	diffterm(lo, hi, ng, dx, Q, D, eta, alam);
	gpu_diffterm(h_const, d_const, d_Q, d_D);

	//!
    //! Calculate F at time N+1/3.
    //!
//	hypterm(lo, hi, ng, dx, Unew, Q, F);
	gpu_hypterm(h_const, d_const, d_Unew, d_Q, d_F);

	//!
    //! Calculate U at time N+2/3.
    //!
//	FOR(i, 0, dim[0]){
//		FOR(j, 0, dim[0]){
//			FOR(k, 0, dim[0]){
//				FOR(l, 0, nc)
//					Unew[i+NG][j+NG][k+NG][l] =
//						ThreeQuarters *  U[i+NG][j+NG][k+NG][l] +
//						OneQuarter    * (Unew[i+NG][j+NG][k+NG][l] + dt*(D[i][j][k][l] + F[i][j][k][l]));
//			}
//		}
//	}
	gpu_Unew(h_const, d_const, d_Unew, d_U, d_D, d_F, dt, 2);

	//!
    //! Sync U^2/3 prior to calculating D & F. -- multifab_fill_boundary(Unew)
    //!
//	fill_boundary(Unew, dim, dim_g);
	gpu_fill_boundary(h_const, d_const, d_Unew);

    //!
    //! Calculate primitive variables based on U^2/3.
    //!
//	ctoprim(lo, hi, Unew, Q, dx, ng);
	gpu_ctoprim(h_const, d_const, d_Unew, d_Q);

    //!
    //! Calculate D at time N+2/3.
    //!
//    diffterm(lo, hi, ng, dx, Q, D, eta, alam);
    gpu_diffterm(h_const, d_const, d_Q, d_D);

    //!
    //! Calculate F at time N+2/3.
    //!
//	hypterm(lo, hi, ng, dx, Unew, Q, F);
	gpu_hypterm(h_const, d_const, d_Unew, d_Q, d_F);

    //!
    //! Calculate U at time N+1.
    //!
//	FOR(i, 0, dim[0]){
//		FOR(j, 0, dim[0]){
//			FOR(k, 0, dim[0]){
//				FOR(l, 0, nc)
//					U[i+NG][j+NG][k+NG][l] =
//						OneThird    *  U[i+NG][j+NG][k+NG][l] +
//						TwoThirds   * (Unew[i+NG][j+NG][k+NG][l] + dt*(D[i][j][k][l] + F[i][j][k][l]));
//			}
//		}
//	}
	gpu_Unew(h_const, d_const, d_Unew, d_U, d_D, d_F, dt, 3);

	// Free memory
//	free_4D(D, dim);
//	free_4D(D2, dim);
//	free_4D(F, dim);
//	free_4D(F2, dim);
//	free_4D(Q, dim_g);
//	free_4D(Q2, dim_g);
//	free_4D(Unew, dim_g);
//	free_4D(Unew2, dim_g);
//	free_4D(U2, dim_g);

//	double dummy;
//	cudaMemcpy(&dummy, d_Unew, sizeof(double), cudaMemcpyDeviceToHost);
	gpu_free_4D(d_Unew);
	gpu_free_4D(d_Q);
	gpu_free_4D(d_D);
	gpu_free_4D(d_F);

	FOR(i, 0, MAX_TEMP)
		gpu_free_3D(h_const.temp[i]);

}

void advance_test(
	global_const_t &h_const, 	// i: Global struct containing application parameters
	global_const_t *d_const		// i: Device pointer to global struct containing application paramters
){
	int i, l, n;
	int nc, dim_g[3];
	double dt, dt2, dx[DIM], cfl, eta, alam;
	double ****U, ****U2;
	double *d_u;
	FILE *fin, *fout;

	nc = NC;
	FOR(i, 0, 3)
		dim_g[i] = h_const.dim_g[i];

	// Allocation
	allocate_4D(U, dim_g, nc);
	allocate_4D(U2, dim_g, nc);
	gpu_allocate_4D(d_u, dim_g, 5);
	printf("d_u = %p\n", d_u);
	printf("size = %x\n", dim_g[0]*dim_g[1]*dim_g[2]*5);

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

	gpu_copy_from_host_4D(d_u, U, dim_g, 5);

//	advance(U, dt, dx, cfl, eta, alam);
	gpu_advance(h_const, d_const, d_u, dt);
	printf("after gpu_advance()\n");

	gpu_copy_to_host_4D(U, d_u, dim_g, 5);

	fout=fopen("../testcases/advance_output", "r");
	FOR(l, 0, nc)
		read_3D(fout, U2, dim_g, l);
	check_4D_array("U", U, U2, dim_g, nc);

	fscanf(fout, "%le", &dt2);
	check_double(dt, dt2, "dt");
	fclose(fout);
	printf("Correct!\n");

	// Free memory
	free_4D(U, 	dim_g, nc);
	free_4D(U2, dim_g, nc);
	gpu_free_4D(d_u);
}
