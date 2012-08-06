#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "helper_functions.h"

void gpu_advance(
	double ****U,	// i/o
	double &dt,		// o
	double dx[],	// i: dx[U.dim]
	double cfl,		// i
	double eta,		// i
	double alam		// i
){
	int lo[3], hi[3], i, j, k, l, n, nc, ng;
	double courno, courno_proc;
	double ****D, ****F, ****Unew, ****Q;
	double ****Q2, ****D2, ****F2, ****Unew2, ****U2;

	// GPU variables
	double *d_u, *d_q, *d_flux, *d_cons;

    // Some arithmetic constants.
    double OneThird      = 1.E0/3.E0;
    double TwoThirds     = 2.E0/3.E0;
    double OneQuarter    = 1.E0/4.E0;
    double ThreeQuarters = 3.E0/4.E0;

	nc = NC; // ncomp(U)
	ng = NG; // nghost(U)

	int dim[3], dim_g[3];
	dim[0] 		= dim[1] 	= dim[2] 	= NCELLS;
	dim_g[0] 	= dim_g[1]	= dim_g[2]	= NCELLS+NG+NG;

	lo[0] = lo[1] = lo[2] = NG;
	hi[0] = hi[1] = hi[2] = NCELLS-1+NG;

	// Allocation
	allocate_4D(D, dim, nc);
	allocate_4D(D2, dim, nc);
	allocate_4D(F, dim, nc);
	allocate_4D(F2, dim, nc);
	allocate_4D(Q, dim_g, nc+1);
	allocate_4D(Q2, dim_g, nc+1);
	allocate_4D(Unew, dim_g, nc);
	allocate_4D(Unew2, dim_g, nc);
	allocate_4D(U2, dim_g, nc);

	gpu_allocate_4D(d_u, dim_g, 5);
	gpu_allocate_4D(d_q, dim_g, 6);
	gpu_allocate_4D(d_flux, dim, 5);
	gpu_allocate_4D(d_cons, dim_g, 5);

	FOR(i, 0, MAX_TEMP)
		gpu_allocate_3D(h_const.temp[i], dim_g);

	//
	// multifab_fill_boundary(U)
	//
	fill_boundary(U, dim, dim_g);

	gpu_copy_from_host_4D(d_u, u, dim_g, 5);
	gpu_copy_from_host_4D(d_q, q, dim_g, 6);
	gpu_copy_from_host_4D(d_flux, difflux, dim, 5);
	gpu_copy_from_host_4D(d_cons, cons, dim_g, 5);

    //!
    //! Calculate primitive variables based on U.
    //!
    //! Also calculate courno so we can set "dt".
    //!
	courno_proc = 1.0E-50;
//	ctoprim(lo, hi, U, Q, dx, ng, courno_proc);
	gpu_ctoprim(h_const, d_const, d_u, d_q, courno);

	courno = courno_proc;
	dt = cfl/courno;
	printf("dt, courno = %le, %le\n", dt, courno);

    //!
    //! Calculate D at time N.
    //!
//	diffterm(lo, hi, ng, dx, Q, D, eta, alam);
	gpu_diffterm(h_const, d_const, d_q, d_flux);

    //!
    //! Calculate F at time N.
    //!
//	hypterm(lo, hi, ng, dx, U, Q, F);
	gpu_hypterm(h_const, d_const, d_cons, d_q, d_flux);


	gpu_copy_to_host_4D(cons, d_cons, dim_g, 5);
	gpu_copy_to_host_4D(q   , d_q   , dim_g, 6);
	gpu_copy_to_host_4D(flux, d_flux, dim  , 5);

    //!
    //! Calculate U at time N+1/3.
    //!
	FOR(i, 0, dim[0]){
		FOR(j, 0, dim[1]){
			FOR(k, 0, dim[2]){
				FOR(l, 0, nc)
					Unew[i+NG][j+NG][k+NG][l] = U[i+NG][j+NG][k+NG][l] + dt*(D[i][j][k][l] + F[i][j][k][l]);
			}
		}
	}

	//!
    //! Sync U^1/3 prior to calculating D & F. -- multifab_fill_boundary(Unew)
    //!
	fill_boundary(Unew, dim, dim_g);

	//!
    //! Calculate primitive variables based on U^1/3.
    //!
	ctoprim(lo, hi, Unew, Q, dx, ng);

    //!
    //! Calculate D at time N+1/3.
    //!
	diffterm(lo, hi, ng, dx, Q, D, eta, alam);

	//!
    //! Calculate F at time N+1/3.
    //!
	hypterm(lo, hi, ng, dx, Unew, Q, F);

	//!
    //! Calculate U at time N+2/3.
    //!
	FOR(i, 0, dim[0]){
		FOR(j, 0, dim[0]){
			FOR(k, 0, dim[0]){
				FOR(l, 0, nc)
					Unew[i+NG][j+NG][k+NG][l] =
						ThreeQuarters *  U[i+NG][j+NG][k+NG][l] +
						OneQuarter    * (Unew[i+NG][j+NG][k+NG][l] + dt*(D[i][j][k][l] + F[i][j][k][l]));
			}
		}
	}

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
	FOR(i, 0, dim[0]){
		FOR(j, 0, dim[0]){
			FOR(k, 0, dim[0]){
				FOR(l, 0, nc)
					U[i+NG][j+NG][k+NG][l] =
						OneThird    *  U[i+NG][j+NG][k+NG][l] +
						TwoThirds   * (Unew[i+NG][j+NG][k+NG][l] + dt*(D[i][j][k][l] + F[i][j][k][l]));
			}
		}
	}

	// Free memory
	free_4D(D, dim);
	free_4D(D2, dim);
	free_4D(F, dim);
	free_4D(F2, dim);
	free_4D(Q, dim_g);
	free_4D(Q2, dim_g);
	free_4D(Unew, dim_g);
	free_4D(Unew2, dim_g);
	free_4D(U2, dim_g);

	gpu_free_4D(d_u);
	gpu_free_4D(d_q);

	FOR(i, 0, MAX_TEMP)
		gpu_free_3D(h_const.temp[i]);

	gpu_free_4D(d_q);
	gpu_free_4D(d_flux);
}

void advance_test(){
	int i, l, n;
	int nc, dim_g[3];
	double dt, dt2, dx[DIM], cfl, eta, alam;
	double ****U, ****U2;
	FILE *fin, *fout;

	nc = NC;
	dim_g[0] = dim_g[1] = dim_g[2] = NCELLS+NG+NG;

	// Allocation
	allocate_4D(U, dim_g, nc);
	allocate_4D(U2, dim_g, nc);

	// Initiation
	fin = fopen("../fortran90/advance_input", "r");
	FOR(l, 0, nc)
		read_3D(fin, U, dim_g, l);

	fscanf(fin, "%le", &dt);
	FOR(i, 0, 3)
		fscanf(fin, "%le", &dx[i]);
	fscanf(fin, "%le", &cfl);
	fscanf(fin, "%le", &eta);
	fscanf(fin, "%le", &alam);
	fclose(fin);

	advance(U, dt, dx, cfl, eta, alam);

	fout=fopen("../fortran90/advance_output", "r");
	FOR(l, 0, nc)
		read_3D(fout, U2, dim_g, l);
	check_4D_array("U", U, U2, dim_g, nc);

	fscanf(fout, "%le", &dt2);
	check_double(dt, dt2, "dt");
	fclose(fout);
	printf("Correct!\n");

	// Free memory
	free_4D(U, dim_g);
	free_4D(U2, dim_g);
}
