#include "header.h"
#include <math.h>
#define u(i,j,k,l)  u[i][j][k][l-1]
#define q(i,j,k,l)  q[i][j][k][l-1]
#define dx(i)		dx[i-1]

void ctoprim (
    int lo[],       // i: lo[3]
    int hi[],       // i: hi[3]
    double ****u,   // i: u[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][5]
    double ****q, 	// o: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
    double dx[],    // i: dx[3]
    int ng,         // i
    double courno   // i/o
){
    int i, j, k;
    double c, eint, courx, coury, courz, courmx, courmy, courmz, rhoinv;

    const double GAMMA  = 1.4E0;
    const double CV     = 8.3333333333E6;

//    #pragma omp parallel for private(i, j, k, eint, rhoinv)
    DO(i, lo[0]-ng, hi[0]+ng){
        DO(j, lo[1]-ng, hi[1]+ng){
            DO(k, lo[2]-ng, hi[2]+ng){

				rhoinv     = 1.0E0/u(i,j,k,1);
				q(i,j,k,1) = u(i,j,k,1);
				q(i,j,k,2) = u(i,j,k,2)*rhoinv;
				q(i,j,k,3) = u(i,j,k,3)*rhoinv;
				q(i,j,k,4) = u(i,j,k,4)*rhoinv;

				eint = u(i,j,k,5)*rhoinv - 0.5E0*(SQR(q(i,j,k,2)) + SQR(q(i,j,k,3)) + SQR(q(i,j,k,4)));

				q(i,j,k,5) = (GAMMA-1.0E0)*eint*u(i,j,k,1);
				q(i,j,k,6) = eint/CV;

            }
        }
    }

    if(courno != -1.0){
//        #pragma omp parallel for private(i, j, k, c, courx, coury, courz) reduction(max: courmx, courmy, courmz)
        DO(i, lo[0], hi[0]){
            DO(j, lo[1], hi[1]){
                DO(k, lo[2], hi[2]){

					c     = sqrt(GAMMA*q(i,j,k,5)/q(i,j,k,1));
					courx = ( c+fabs(q(i,j,k,2)) ) / dx(1);
					coury = ( c+fabs(q(i,j,k,3)) ) / dx(2);
					courz = ( c+fabs(q(i,j,k,4)) ) / dx(3);

					courmx = MAX( courmx, courx );
					courmy = MAX( courmy, coury );
					courmz = MAX( courmz, courz );

                }
            }
        }
    }

    //
    // Compute running max of Courant number over grids.
    //
    courno = MAX(MAX(courmx, courmy), MAX(courmz, courno));
}

