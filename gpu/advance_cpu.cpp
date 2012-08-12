#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "helper_functions.h"

void advance(
	double ****U,		// i/o
	double ****Unew,	// o
	double ****Q,		// o
	double ****D,		// o
	double ****F,		// o
	double &dt,			// o
	double dx[],		// i: dx[U.dim]
	double cfl,			// i
	double eta,			// i
	double alam			// i
){
	int lo[3], hi[3], i, j, k, l, n, nc, ng;
	double courno, courno_proc;

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
	return;

    //!
    //! Calculate F at time N.
    //!
	hypterm(lo, hi, ng, dx, U, Q, F);

    //!
    //! Calculate U at time N+1/3.
    //!
	FOR(l, 0, nc){
		FOR(i, 0, dim[0]){
			FOR(j, 0, dim[1]){
				FOR(k, 0, dim[2]){
					Unew[l][i+NG][j+NG][k+NG] = U[l][i+NG][j+NG][k+NG] + dt*(D[l][i][j][k] + F[l][i][j][k]);
				}
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
    FOR(l, 0, nc){
		FOR(i, 0, dim[0]){
			FOR(j, 0, dim[0]){
				FOR(k, 0, dim[0]){
					Unew[l][i+NG][j+NG][k+NG] =
						ThreeQuarters *  U[l][i+NG][j+NG][k+NG] +
						OneQuarter    * (Unew[l][i+NG][j+NG][k+NG] + dt*(D[l][i][j][k] + F[l][i][j][k]));
				}
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
    FOR(l, 0, nc){
		FOR(i, 0, dim[0]){
			FOR(j, 0, dim[0]){
				FOR(k, 0, dim[0]){
					U[l][i+NG][j+NG][k+NG] =
						OneThird    *  U[l][i+NG][j+NG][k+NG] +
						TwoThirds   * (Unew[l][i+NG][j+NG][k+NG] + dt*(D[l][i][j][k] + F[l][i][j][k]));
				}
			}
		}
    }
}

void advance_test(
	double ****U,
	double ****Unew,
	double ****Q,
	double ****D,
	double ****F
){
	int i, l, n;
	int nc, dim_ng[3];
	double dt, dt2, dx[DIM], cfl, eta, alam;
	FILE *fin, *fout;
	double ****U2;

	nc = NC;
	dim_ng[0] = dim_ng[1] = dim_ng[2] = NCELLS+NG+NG;

	// Allocation
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

	advance(U, Unew, Q, D, F, dt, dx, cfl, eta, alam);
	return;

	fout=fopen("../testcases/advance_output", "r");
	FOR(l, 0, nc)
		read_3D(fout, U2, dim_ng, l);
	check_4D_array("U", U, U2, dim_ng, nc);

	fscanf(fout, "%le", &dt2);
	check_double(dt, dt2, "dt");
	fclose(fout);
	printf("Correct!\n");

	// Free memory
	free_4D(U2, dim_ng, nc);
}
