#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "util.h"

void new_advance_hybrid(
	global_const_t h_const,
	global_const_t *d_const,
	double ****U,	// i/o
	double ****Unew,
	double ****Q,
	double ****D,
	double ****F,
	double *d_U,	// i/o
	double *d_Unew,	// i/o
	double *d_Q,	// i/o
	double *d_D,	// i/o
	double *d_F,	// i/o
	double &dt		// o
){
	int lo[3], hi[3], i, j, k, l, n, nc, ng, *dim, *dim_g;
	double courno, courno_proc;
	double cfl, eta, alam, *dx;
	cfl = h_const.cfl;
	eta = h_const.eta;
	alam = h_const.alam;
	dx = h_const.dx;
	dim = h_const.dim;
	dim_g = h_const.dim_g;

    // Some arithmetic constants.
    double OneThird      = h_const.OneThird;
    double TwoThirds     = h_const.TwoThirds;
    double OneQuarter    = h_const.OneQuarter;
    double ThreeQuarters = h_const.ThreeQuarters;

	nc = h_const.nc; // ncomp(U)
	ng = h_const.ng; // nghost(U)

	lo[0] = lo[1] = lo[2] = ng;
	hi[0] = hi[1] = hi[2] = h_const.ncells-1+ng;

	//
	// multifab_fill_boundary(U)
	//
//	printf("fill boundary\n");
//	gpu_copy_from_host_4D(d_U, U, dim_g, nc);
//	gpu_fill_boundary(h_const, d_const, d_U);
//	gpu_copy_to_host_4D(U, d_U, dim_g, nc);
	fill_boundary(h_const, U);

    //!
    //! Calculate primitive variables based on U.
    //!
    //! Also calculate courno so we can set "dt".
    //!
//    printf("ctoprim\n");
	courno_proc = 1.0E-50;
	ctoprim(h_const, U, Q, courno_proc);
//	gpu_ctoprim(h_const, d_const, d_U, d_Q, courno_proc);
//	gpu_copy_to_host_4D(Q, d_Q, dim_g, nc+1);

	courno = courno_proc;
	dt = cfl/courno;
	printf("dt, courno = %le, %le\n", dt, courno);

    //!
    //! Calculate D at time N.
    //!
//    printf("diffterm\n");
//    diffterm(h_const, Q, D);
//	double ***tmp;
//    allocate_3D(tmp, dim);
//    set_3D(0.0, tmp, dim);
//    number_3D(Q[5], dim_g);
//    set_3D(55.55, Q[5], dim_g);
//    gpu_copy_from_host_3D(h_const.temp[TXX], tmp, dim);
    gpu_copy_from_host_4D(d_Q, Q, dim_g, nc+1);
    gpu_diffterm(h_const, d_const, d_Q, d_D);
    gpu_copy_to_host_4D(D, d_D, dim, nc);
//	gpu_copy_to_host_4D(Q, d_Q, dim_g, nc+1);
//	gpu_copy_to_host_3D(tmp, h_const.temp[TXX], dim);


//    FILE *fdebug=fopen("txx", "w");
//    fprintf(fdebug, "1\n%d %d %d\n", dim[0], dim[1], dim[2]);
//    print_3D(fdebug, tmp, dim);
//    free_3D(tmp, dim);
//    fclose(fdebug);
//    FILE *fd = fopen("dgpu", "w");
//    fprintf(fd, "%d\n%d %d %d\n", h_const.nc, dim[0], dim[1], dim[2]);
//	print_4D(fd, D, dim, nc);
//    fclose(fd);
//    FILE *fq = fopen("qgpu", "w");
//    fprintf(fq, "%d\n%d %d %d\n", h_const.nc+1, dim_g[0], dim_g[1], dim_g[2]);
//	print_4D(fq, Q, dim_g, nc+1);
//    fclose(fq);
//	return;

    //!
    //! Calculate F at time N.
    //!
//    printf("hypterm\n");
	hypterm(h_const, U, Q, F);
//	gpu_copy_from_host_4D(d_U, U, dim_g, nc);
//	gpu_copy_from_host_4D(d_Q, Q, dim_g, nc+1);
//    gpu_hypterm(h_const, d_const, d_U, d_Q, d_F);
//    gpu_copy_to_host_4D(F, d_F, dim, nc);


    //!
    //! Calculate U at time N+1/3.
    //!
//    printf("Unew\n");
	FOR(l, 0, nc){
		FOR(k, 0, dim[2]){
			FOR(j, 0, dim[1]){
				FOR(i, 0, dim[0]){
					Unew[l][k+NG][j+NG][i+NG] = U[l][k+NG][j+NG][i+NG] + dt*(D[l][k][j][i] + F[l][k][j][i]);
				}
			}
		}
	}

//	gpu_copy_from_host_4D(d_D, D, dim, nc);
//	gpu_Unew(h_const, d_const, d_Unew, d_U, d_D, d_F, dt, 1);
//	gpu_copy_to_host_4D(Unew, d_Unew, dim_g, nc);


	//!
    //! Sync U^1/3 prior to calculating D & F. -- multifab_fill_boundary(Unew)
    //!
//    printf("fill boundary2\n");
	fill_boundary(h_const, Unew);
//	gpu_copy_from_host_4D(d_Unew, Unew, dim_g, nc);
//	gpu_fill_boundary(h_const, d_const, d_Unew);
//	gpu_copy_to_host_4D(Unew, d_Unew, dim_g, nc);

	//!
    //! Calculate primitive variables based on U^1/3.
    //!
//    printf("ctoprim2\n");
	ctoprim(h_const, Unew, Q);
//	gpu_ctoprim(h_const, d_const, d_Unew, d_Q);
//	gpu_copy_to_host_4D(Q, d_Q, dim_g, nc+1);

    //!
    //! Calculate D at time N+1/3.
    //!
//    printf("diffterm2\n");
	diffterm(h_const, Q, D);
//	gpu_copy_from_host_4D(d_Q, Q, dim_g, nc+1);
//    gpu_diffterm(h_const, d_const, d_Q, d_D);
//    gpu_copy_to_host_4D(D, d_D, dim, nc);

	//!
    //! Calculate F at time N+1/3.
    //!
//    printf("hypterm2\n");
	hypterm(h_const, Unew, Q, F);
//	gpu_copy_from_host_4D(d_Unew, Unew, dim_g, nc);
//	gpu_copy_from_host_4D(d_Q, Q, dim_g, nc+1);
//	gpu_hypterm(h_const, d_const, d_Unew, d_Q, d_F);
//	gpu_copy_to_host_4D(F, d_F, dim, nc);

	//!
    //! Calculate U at time N+2/3.
    //!
//    printf("Unew2\n");
    FOR(l, 0, nc){
    	FOR(k, 0, dim[2]){
			FOR(j, 0, dim[1]){
				FOR(i, 0, dim[0]){
					Unew[l][k+NG][j+NG][i+NG] =
						ThreeQuarters *  U[l][k+NG][j+NG][i+NG] +
						OneQuarter    * (Unew[l][k+NG][j+NG][i+NG] + dt*(D[l][k][j][i] + F[l][k][j][i]));
				}
			}
		}
    }
//    gpu_copy_from_host_4D(d_Unew, 	Unew, 	dim_g, 	nc);
//	gpu_copy_from_host_4D(d_U, 		U, 		dim_g, 	nc);
//	gpu_copy_from_host_4D(d_D, 		D, 		dim, 	nc);
//	gpu_copy_from_host_4D(d_F, 		F, 		dim, 	nc);

//	gpu_Unew(h_const, d_const, d_Unew, d_U, d_D, d_F, dt, 2);
//	gpu_copy_to_host_4D(Unew, d_Unew, dim_g, nc);

	//!
    //! Sync U^2/3 prior to calculating D & F. -- multifab_fill_boundary(Unew)
    //!
//    printf("fill boundary3\n");
	fill_boundary(h_const, Unew);
//	gpu_copy_from_host_4D(d_Unew, Unew, dim_g, nc);
//	gpu_fill_boundary(h_const, d_const, d_Unew);
//	gpu_copy_to_host_4D(Unew, d_Unew, dim_g, nc);

    //!
    //! Calculate primitive variables based on U^2/3.
    //!
//    printf("ctoprim3\n");
	ctoprim(h_const, Unew, Q);
//	gpu_ctoprim(h_const, d_const, d_Unew, d_Q);
//	gpu_copy_to_host_4D(Q, d_Q, dim_g, nc+1);

    //!
    //! Calculate D at time N+2/3.
    //!
//    printf("diffterm3\n");
    diffterm(h_const, Q, D);
//    gpu_copy_from_host_4D(d_Q, Q, dim_g, nc+1);
//    gpu_diffterm(h_const, d_const, d_Q, d_D);
//    gpu_copy_to_host_4D(D, d_D, dim, nc);

    //!
    //! Calculate F at time N+2/3.
    //!
//    printf("hypterm3\n");
	hypterm(h_const, Unew, Q, F);
//	gpu_copy_from_host_4D(d_Unew, Unew, dim_g, nc);
//	gpu_copy_from_host_4D(d_Q, Q, dim_g, nc+1);
//	gpu_hypterm(h_const, d_const, d_Unew, d_Q, d_F);
//	gpu_copy_to_host_4D(F, d_F, dim, nc);

    //!
    //! Calculate U at time N+1.
    //!
//    printf("Unew3\n");
    FOR(l, 0, nc){
    	FOR(k, 0, dim[2]){
			FOR(j, 0, dim[1]){
				FOR(i, 0, dim[0]){
					U[l][i+NG][j+NG][k+NG] =
						OneThird    *  U[l][k+NG][j+NG][i+NG] +
						TwoThirds   * (Unew[l][k+NG][j+NG][i+NG] + dt*(D[l][k][j][i] + F[l][k][j][i]));
				}
			}
		}
    }

//    gpu_copy_from_host_4D(d_Unew, 	Unew, 	dim_g, 	nc);
	gpu_copy_from_host_4D(d_U, 		U, 		dim_g, 	nc);
//	gpu_copy_from_host_4D(d_D, 		D, 		dim, 	nc);
//	gpu_copy_from_host_4D(d_F, 		F, 		dim, 	nc);
//	gpu_Unew(h_const, d_const, d_Unew, d_U, d_D, d_F, dt, 3);
//	gpu_copy_to_host_4D(U, d_U, dim_g, nc);
//	printf("exiting\n");
}

