#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "helper_functions.h"

void advance_hybrid(
	global_const_t h_const,
	global_const_t *d_const,
	double ****U,	// i/o
	double *d_U,	// i/o
	double *d_Unew,	// i/o
	double *d_Q,	// i/o
	double *d_D,	// i/o
	double *d_F,	// i/o
	double &dt,		// o
	double dx[],	// i: dx[U.dim]
	double cfl,		// i
	double eta,		// i
	double alam		// i
){
	int lo[3], hi[3], i, j, k, l, n, nc, ng;
	double courno, courno_proc;
	double ****D, ****F, ****Unew, ****Q;

    // Some arithmetic constants.
    double OneThird      = 1.E0/3.E0;
    double TwoThirds     = 2.E0/3.E0;
    double OneQuarter    = 1.E0/4.E0;
    double ThreeQuarters = 3.E0/4.E0;

	nc = h_const.nc; // ncomp(U)
	ng = h_const.ng; // nghost(U)

	int dim[3], dim_g[3];
	dim[0] 		= dim[1] 	= dim[2] 	= h_const.ncells;
	dim_g[0] 	= dim_g[1]	= dim_g[2]	= h_const.ncells+ng+ng;

	lo[0] = lo[1] = lo[2] = ng;
	hi[0] = hi[1] = hi[2] = h_const.ncells-1+ng;

	// Allocation
	allocate_4D(D, dim, nc);
	allocate_4D(F, dim, nc);
	allocate_4D(Q, dim_g, nc+1);
	allocate_4D(Unew, dim_g, nc);

	//
	// multifab_fill_boundary(U)
	//
	gpu_copy_from_host_4D(d_U, U, dim_g, nc);
	gpu_fill_boundary(h_const, d_const, d_U);
	gpu_copy_to_host_4D(U, d_U, dim_g, nc);
//	fill_boundary(U, dim, dim_g);

    //!
    //! Calculate primitive variables based on U.
    //!
    //! Also calculate courno so we can set "dt".
    //!
	courno_proc = 1.0E-50;
	gpu_ctoprim(h_const, d_const, d_U, d_Q, courno_proc);
//	gpu_copy_to_host_4D(Q, d_Q, dim_g, nc+1);
//	ctoprim(lo, hi, U, Q, dx, ng, courno_proc);

	courno = courno_proc;
	dt = cfl/courno;
	printf("dt, courno = %le, %le\n", dt, courno);
//	return;

    //!
    //! Calculate D at time N.
    //!
//    gpu_copy_from_host_4D(d_Q, Q, dim_g, nc+1);
    gpu_diffterm(h_const, d_const, d_Q, d_D);
//    gpu_copy_to_host_4D(D, d_D, dim, nc);

//	diffterm(lo, hi, ng, dx, Q, D, eta, alam);
//	gpu_copy_from_host_4D(d_D, D, dim, nc);

    //!
    //! Calculate F at time N.
    //!
    gpu_hypterm(h_const, d_const, d_U, d_Q, d_F);
//    gpu_copy_to_host_4D(F, d_F, dim, nc);
//	hypterm(lo, hi, ng, dx, U, Q, F);

    //!
    //! Calculate U at time N+1/3.
    //!
//	FOR(l, 0, nc){
//		FOR(i, 0, dim[0]){
//			FOR(j, 0, dim[1]){
//				FOR(k, 0, dim[2]){
//					Unew[l][i+NG][j+NG][k+NG] = U[l][i+NG][j+NG][k+NG] + dt*(D[l][i][j][k] + F[l][i][j][k]);
//				}
//			}
//		}
//	}

	gpu_Unew(h_const, d_const, d_Unew, d_U, d_D, d_F, dt, 1);
	gpu_copy_to_host_4D(Unew, d_Unew, dim_g, nc);


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
	hypterm(lo, hi, ng, dx, Unew, Q, F);
	gpu_hypterm(h_const, d_const, d_Unew, d_Q, d_F);

	//!
    //! Calculate U at time N+2/3.
    //!
//    FOR(l, 0, nc){
//		FOR(i, 0, dim[0]){
//			FOR(j, 0, dim[0]){
//				FOR(k, 0, dim[0]){
//					Unew[l][i+NG][j+NG][k+NG] =
//						ThreeQuarters *  U[l][i+NG][j+NG][k+NG] +
//						OneQuarter    * (Unew[l][i+NG][j+NG][k+NG] + dt*(D[l][i][j][k] + F[l][i][j][k]));
//				}
//			}
//		}
//    }
//    gpu_copy_from_host_4D(d_Unew, 	Unew, 	dim_g, 	nc);
//	gpu_copy_from_host_4D(d_U, 		U, 		dim_g, 	nc);
//	gpu_copy_from_host_4D(d_D, 		D, 		dim, 	nc);
//	gpu_copy_from_host_4D(d_F, 		F, 		dim, 	nc);
	gpu_Unew(h_const, d_const, d_Unew, d_U, d_D, d_F, dt, 2);
	gpu_copy_to_host_4D(Unew, d_Unew, dim_g, nc);

	//!
    //! Sync U^2/3 prior to calculating D & F. -- multifab_fill_boundary(Unew)
    //!
	fill_boundary(Unew, dim, dim_g);

    //!
    //! Calculate primitive variables based on U^2/3.
    //!
	ctoprim(lo, hi, Unew, Q, dx, ng);

    //!
    //! Calculate D at time N+2/3.
    //!
    diffterm(lo, hi, ng, dx, Q, D, eta, alam);

    //!
    //! Calculate F at time N+2/3.
    //!
	hypterm(lo, hi, ng, dx, Unew, Q, F);

    //!
    //! Calculate U at time N+1.
    //!
//    FOR(l, 0, nc){
//		FOR(i, 0, dim[0]){
//			FOR(j, 0, dim[0]){
//				FOR(k, 0, dim[0]){
//					U[l][i+NG][j+NG][k+NG] =
//						OneThird    *  U[l][i+NG][j+NG][k+NG] +
//						TwoThirds   * (Unew[l][i+NG][j+NG][k+NG] + dt*(D[l][i][j][k] + F[l][i][j][k]));
//				}
//			}
//		}
//    }
    gpu_copy_from_host_4D(d_Unew, 	Unew, 	dim_g, 	nc);
	gpu_copy_from_host_4D(d_U, 		U, 		dim_g, 	nc);
	gpu_copy_from_host_4D(d_D, 		D, 		dim, 	nc);
	gpu_copy_from_host_4D(d_F, 		F, 		dim, 	nc);
	gpu_Unew(h_const, d_const, d_Unew, d_U, d_D, d_F, dt, 3);
	gpu_copy_to_host_4D(U, d_U, dim_g, nc);

	// Free memory
	free_4D(D, 		dim, 	nc);
	free_4D(F, 		dim,	nc);
	free_4D(Q, 		dim_g,	nc+1);
	free_4D(Unew, 	dim_g,	nc);

}

void advance_hybrid_test(
	global_const_t h_const, 	// i: Global struct containing application parameters
	global_const_t *d_const,	// i: Device pointer to global struct containing application paramters
	double ****U,
	double ****Unew,
	double ****Q,
	double ****D,
	double ****F
){
	int i, l, n;
	int nc, dim[3], dim_g[3];
	double dt, dt2, dx[DIM], cfl, eta, alam;
	double ****U2;
	double *d_U, *d_Unew, *d_Q, *d_D, *d_F;
	FILE *fin, *fout;

	nc = h_const.nc;
	FOR(i, 0, 3){
		dim[i] = h_const.dim[i];
		dim_g[i] = h_const.dim_g[i];
	}

	// Allocation
	allocate_4D(U2, dim_g, nc);

	gpu_allocate_4D(d_U, dim_g, 5);
	gpu_allocate_4D(d_Unew, dim_g, 	5);
	gpu_allocate_4D(d_Q, 	dim_g, 	6);
	gpu_allocate_4D(d_D, 	dim, 	5);
	gpu_allocate_4D(d_F, 	dim, 	5);

//	char *dest = (char *)d_const + ((char *)&h_const.temp - (char *)&h_const);
//	FOR(i, 0, MAX_TEMP)
//		gpu_allocate_3D(h_const.temp[i], dim_g);
//	cudaMemcpy((double *) dest, h_const.temp, MAX_TEMP*sizeof(double *), cudaMemcpyHostToDevice);

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

	advance_hybrid(h_const, d_const, U, d_U, d_Unew, d_Q, d_D, d_F, dt, dx, cfl, eta, alam);
	printf("after advance_hybrid()\n");
	gpu_copy_to_host_4D(Q, d_Q, dim_g, nc+1);
	gpu_copy_to_host_4D(D, d_D, dim, nc);

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

	gpu_free_4D(d_U);
	gpu_free_4D(d_Unew);
	gpu_free_4D(d_Q);
	gpu_free_4D(d_D);
	gpu_free_4D(d_F);

//	FOR(i, 0, MAX_TEMP)
//		gpu_free_3D(h_const.temp[i]);
}
