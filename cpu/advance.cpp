#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "header.h"
#include "helper_functions.h"

extern global_const_t h_const;

void new_advance(
	double ****U,		// i/o
	double ****Unew,
	double ****Q,
	double ****D,
	double ****F,
	double &dt			// o
){
	int lo[3], hi[3], i, j, k, l, n, nc, ng;
	int *dim = h_const.dim, *dim_ng = h_const.dim_g;
	double courno, courno_proc;
	double *dx = h_const.dx;
	double cfl, eta, alam;
	cfl = h_const.cfl;
	eta = h_const.eta;
	alam = h_const.alam;

	nc = h_const.nc; // ncomp(U)
	ng = h_const.ng; // nghost(U)

	lo[0] = lo[1] = lo[2] = h_const.ng;
	hi[0] = hi[1] = hi[2] = h_const.ncells-1+h_const.ng;

	// Some arithmetic constants.
    double OneThird      = h_const.OneThird;
    double TwoThirds     = h_const.TwoThirds;
    double OneQuarter    = h_const.OneQuarter;
    double ThreeQuarters = h_const.ThreeQuarters;

	//
	// multifab_fill_boundary(U)
	//
	fill_boundary(U, dim, dim_ng);

    //!
    //! Calculate primitive variables based on U.
    //!
    //! Also calculate courno so we can set "dt".
    //!
	courno_proc = 1.0E-50;
	ctoprim(lo, hi, U, Q, dx, ng, courno_proc);

	courno = courno_proc;
	dt = cfl/courno;
	printf("dt, courno = %le, %le\n", dt, courno);

    //!
    //! Calculate D at time N.
    //!
	diffterm(lo, hi, ng, dx, Q, D, eta, alam);

    //!
    //! Calculate F at time N.
    //!
	hypterm(lo, hi, ng, dx, U, Q, F);

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
	fill_boundary(Unew, dim, dim_ng);

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
	fill_boundary(Unew, dim, dim_ng);

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
}

void advance(
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

    // Some arithmetic constants.
    double OneThird      = 1.E0/3.E0;
    double TwoThirds     = 2.E0/3.E0;
    double OneQuarter    = 1.E0/4.E0;
    double ThreeQuarters = 3.E0/4.E0;

	nc = NC; // ncomp(U)
	ng = NG; // nghost(U)

	int dim[3], dim_ng[3];
	dim[0] 		= dim[1] 	= dim[2] 	= NCELLS;
	dim_ng[0] 	= dim_ng[1]	= dim_ng[2]	= NCELLS+NG+NG;

	lo[0] = lo[1] = lo[2] = NG;
	hi[0] = hi[1] = hi[2] = NCELLS-1+NG;

	// Allocation
	allocate_4D(D, dim, nc);
	allocate_4D(D2, dim, nc);
	allocate_4D(F, dim, nc);
	allocate_4D(F2, dim, nc);
	allocate_4D(Q, dim_ng, nc+1);
	allocate_4D(Q2, dim_ng, nc+1);
	allocate_4D(Unew, dim_ng, nc);
	allocate_4D(Unew2, dim_ng, nc);
	allocate_4D(U2, dim_ng, nc);

	//
	// multifab_fill_boundary(U)
	//
	fill_boundary(U, dim, dim_ng);

    //!
    //! Calculate primitive variables based on U.
    //!
    //! Also calculate courno so we can set "dt".
    //!
	courno_proc = 1.0E-50;
	ctoprim(lo, hi, U, Q, dx, ng, courno_proc);

	courno = courno_proc;
	dt = cfl/courno;
	printf("dt, courno = %le, %le\n", dt, courno);

    //!
    //! Calculate D at time N.
    //!
	diffterm(lo, hi, ng, dx, Q, D, eta, alam);

    //!
    //! Calculate F at time N.
    //!
	hypterm(lo, hi, ng, dx, U, Q, F);

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
	fill_boundary(Unew, dim, dim_ng);

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
	fill_boundary(Unew, dim, dim_ng);

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
	free_4D(Q, dim_ng);
	free_4D(Q2, dim_ng);
	free_4D(Unew, dim_ng);
	free_4D(Unew2, dim_ng);
	free_4D(U2, dim_ng);
}

double wall_time ()
{
	struct timeval t;
	gettimeofday (&t, NULL);
	return 1.*t.tv_sec + 1.e-6*t.tv_usec;
}

void advance_test(){
	int i, l, n;
	int nc, dim_ng[3];
	double dt, dt2, dx[DIM], cfl, eta, alam;
	double ****U, ****U2;
	FILE *fin, *fout;
	double seconds;

	nc = NC;
	dim_ng[0] = dim_ng[1] = dim_ng[2] = NCELLS+NG+NG;

	// Allocation
	allocate_4D(U, dim_ng, nc);
	allocate_4D(U2, dim_ng, nc);

	// Initiation
	fin = fopen("../testcases/advance_input", "r");
	FOR(l, 0, nc)
		read_3D(fin, U, dim_ng, l);

	fscanf(fin, "%le", &dt);
	FOR(i, 0, 3)
		fscanf(fin, "%le", &dx[i]);
	fscanf(fin, "%le", &cfl);
	fscanf(fin, "%le", &eta);
	fscanf(fin, "%le", &alam);
	fclose(fin);

	seconds = -wall_time();
	FOR(i, 0, 10)
		advance(U, dt, dx, cfl, eta, alam);
	seconds += wall_time();
	printf("time: %lf\n", seconds);

	fout=fopen("../testcases/advance_output", "r");
	FOR(l, 0, nc)
		read_3D(fout, U2, dim_ng, l);
	check_4D_array("U", U, U2, dim_ng, nc);

	fscanf(fout, "%le", &dt2);
	check_double(dt, dt2, "dt");
	fclose(fout);
	printf("Correct!\n");

	// Free memory
	free_4D(U, dim_ng);
	free_4D(U2, dim_ng);
}
