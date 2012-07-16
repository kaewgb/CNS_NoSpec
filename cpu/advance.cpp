#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "helper_functions.h"

void advance(
	double ****U[],	// i/o
	double &dt,		// o
	double dx[],	// i: dx[U.dim]
	double cfl,		// i
	double eta,		// i
	double alam		// i
){
	int lo[3], hi[3], i, j, k, l, n, nc, ng;
	double courno, courno_proc;
	double ****D[NBOXES], ****F[NBOXES], ****Unew[NBOXES], ****Q[NBOXES];
	double ****Q2[NBOXES], ****D2[NBOXES], ****F2[NBOXES], ****Unew2[NBOXES], ****U2[NBOXES];

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
	FOR(i, 0, NBOXES){
		allocate_4D(D[i], dim, nc);
		allocate_4D(D2[i], dim, nc);
		allocate_4D(F[i], dim, nc);
		allocate_4D(F2[i], dim, nc);
		allocate_4D(Q[i], dim_ng, nc+1);
		allocate_4D(Q2[i], dim_ng, nc+1);
		allocate_4D(Unew[i], dim_ng, nc);
		allocate_4D(Unew2[i], dim_ng, nc);
		allocate_4D(U2[i], dim_ng, nc);
	}

    //!
    //! Calculate primitive variables based on U.
    //!
    //! Also calculate courno so we can set "dt".
    //!
	courno_proc = 1.0E-50;
	FOR(n, 0, NBOXES)
		ctoprim(lo, hi, U[n], Q[n], dx, ng, courno_proc);

	courno = courno_proc;
	dt = cfl/courno;
	printf("dt, courno = %le, %le\n", dt, courno);

    //!
    //! Calculate D at time N.
    //!
	FOR(n, 0, NBOXES)
		diffterm(lo, hi, ng, dx, Q[n], D[n], eta, alam);

    //!
    //! Calculate F at time N.
    //!
	FOR(n, 0, NBOXES)
		hypterm(lo, hi, ng, dx, U[n], Q[n], F[n]);

    //!
    //! Calculate U at time N+1/3.
    //!
    // Read Unew (for borders)
    FILE *fin=fopen("../testcases/advance_unp", "r");
    FOR(n, 0, NBOXES){
    	FOR(l, 0, nc)
			read_3D(fin, Unew[n], dim_ng, l);
    }
    fclose(fin);

	FOR(n, 0, NBOXES){
		FOR(i, 0, dim[0]){
			FOR(j, 0, dim[1]){
				FOR(k, 0, dim[2]){
					FOR(l, 0, nc)
						Unew[n][i+NG][j+NG][k+NG][l] = U[n][i+NG][j+NG][k+NG][l] + dt*(D[n][i][j][k][l] + F[n][i][j][k][l]);
				}
			}
		}
	}

	// Check answer
	FILE *fout=fopen("../testcases/advance_output", "r");
	FOR(n, 0, NBOXES){
		printf("BOX#%d...\n", n);
//		FOR(l, 0, nc+1)
//			read_3D(fout, Q2[n], dim_ng, l);
//		check_4D_array("Q", Q[n], Q2[n], dim_ng, nc+1);
//		FOR(l, 0, nc)
//			read_3D(fout, D2[n], dim, l);
//		check_4D_array("D", D[n], D2[n], dim, nc);
//		FOR(l, 0, nc)
//			read_3D(fout, F2[n], dim, l);
//		check_4D_array("F", F[n], F2[n], dim, nc);
		FOR(l, 0, nc)
			read_3D(fout, U2[n], dim_ng, l);
		check_4D_array("U", U[n], U2[n], dim_ng, nc);
		FOR(l, 0, nc)
			read_3D(fout, Unew2[n], dim_ng, l);
		check_4D_array("Unew", Unew[n], Unew2[n], dim_ng, nc);

	}
	fclose(fout);
	printf("Correct!\n");

	// Free memory
	FOR(i, 0, NBOXES){
		free_4D(D[i], dim);
		free_4D(D2[i], dim);
		free_4D(F[i], dim);
		free_4D(F2[i], dim);
		free_4D(Q[i], dim_ng);
		free_4D(Q2[i], dim_ng);
		free_4D(Unew[i], dim_ng);
		free_4D(Unew2[i], dim_ng);
		free_4D(U2[i], dim_ng);
	}
}

void advance_test(){
	int i, l, n;
	int nc, dim_ng[3];
	double dt, dx[DIM], cfl, eta, alam;
	double ****U[NBOXES];

	nc = NC;
	dim_ng[0] = dim_ng[1] = dim_ng[2] = NCELLS+NG+NG;

	// Allocation
	FOR(i, 0, NBOXES)
		allocate_4D(U[i], dim_ng, nc);

	// Initiation
	FILE *fin = fopen("../testcases/advance_input", "r");
	FOR(n, 0, NBOXES){
		FOR(l, 0, nc)
			read_3D(fin, U[n], dim_ng, l);
	}
	fscanf(fin, "%le", &dt);
	FOR(i, 0, 3)
		fscanf(fin, "%le", &dx[i]);
	fscanf(fin, "%le", &cfl);
	fscanf(fin, "%le", &eta);
	fscanf(fin, "%le", &alam);
	fclose(fin);

	advance(U, dt, dx, cfl, eta, alam);

	// Free memory
	FOR(i, 0, NBOXES)
		free_4D(U[i], dim_ng);
}