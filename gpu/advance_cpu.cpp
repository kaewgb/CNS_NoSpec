#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "helper_functions.h"

void advance(
	global_const_t h_const,
	double ****U,		// i/o
	double ****Unew,	// o
	double ****Q,		// o
	double ****D,		// o
	double ****F,		// o
	double &dt			// o
){
	int lo[3], hi[3], i, j, k, l, n, nc, ng;
	double courno, courno_proc;
	double *dx = h_const.dx;
	double cfl, eta, alam;

    // Some arithmetic constants.
    double OneThird      = 1.E0/3.E0;
    double TwoThirds     = 2.E0/3.E0;
    double OneQuarter    = 1.E0/4.E0;
    double ThreeQuarters = 3.E0/4.E0;

	cfl = h_const.cfl;
	eta = h_const.eta;
	alam = h_const.alam;

	nc = h_const.nc; // ncomp(U)
	ng = h_const.ng; // nghost(U)

	int dim[3], dim_g[3];
	dim[0] 		= dim[1] 	= dim[2] 	= h_const.ncells;
	dim_g[0] 	= dim_g[1]	= dim_g[2]	= h_const.ncells+ng+ng;

	lo[0] = lo[1] = lo[2] = ng;
	hi[0] = hi[1] = hi[2] = h_const.ncells-1+ng;


	//
	// multifab_fill_boundary(U)
	//
	fill_boundary(U, dim, dim_g);

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
//    number_3D(Q[5], dim_g);
	diffterm(lo, hi, ng, dx, Q, D, eta, alam);
//	FILE *fd = fopen("dcpu", "w");
//	fprintf(fd, "%d\n%d %d %d\n", nc, dim[0], dim[1], dim[2]);
//	print_4D(fd, D, dim, nc);
//	fclose(fd);
//	return;

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
					Unew[l][i+ng][j+ng][k+ng] = U[l][i+ng][j+ng][k+ng] + dt*(D[l][i][j][k] + F[l][i][j][k]);
				}
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
    FOR(l, 0, nc){
		FOR(i, 0, dim[0]){
			FOR(j, 0, dim[0]){
				FOR(k, 0, dim[0]){
					Unew[l][i+ng][j+ng][k+ng] =
						ThreeQuarters *  U[l][i+ng][j+ng][k+ng] +
						OneQuarter    * (Unew[l][i+ng][j+ng][k+ng] + dt*(D[l][i][j][k] + F[l][i][j][k]));
				}
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
    FOR(l, 0, nc){
		FOR(i, 0, dim[0]){
			FOR(j, 0, dim[0]){
				FOR(k, 0, dim[0]){
					U[l][i+ng][j+ng][k+ng] =
						OneThird    *  U[l][i+ng][j+ng][k+ng] +
						TwoThirds   * (Unew[l][i+ng][j+ng][k+ng] + dt*(D[l][i][j][k] + F[l][i][j][k]));
				}
			}
		}
    }
}

void advance_cpu_test(
	global_const_t h_const,
	double ****U,
	double ****Unew,
	double ****Q,
	double ****D,
	double ****F
){
	int i, l, n;
	int nc, dim_g[3];
	double dt, dt2, dx[DIM], cfl, eta, alam;
	FILE *fin, *fout;
	double ****U2;

	nc = h_const.nc;
	dim_g[0] = dim_g[1] = dim_g[2] = h_const.ncells+h_const.ng+h_const.ng;

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

	advance(h_const, U, Unew, Q, D, F, dt);

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

void advance_cpu_multistep_test(
	global_const_t h_const,
	double ****U,
	double ****Unew,
	double ****Q,
	double ****D,
	double ****F
){
	int i, l, n;
	int nc, dim_g[3];
	double dt, dt2;
	FILE *fin, *fout;
	double ****U2;

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

	dt = h_const.dt;
	printf("before applying advance\n");
	FOR(i, 0, h_const.nsteps)
		advance(h_const, U, Unew, Q, D, F, dt);
	printf("after advance\n");
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
