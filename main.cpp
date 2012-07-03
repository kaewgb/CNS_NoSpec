#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "header.h"

#define FEQ(x, y)	((fabs(x-y)<0.001)? true:false)

static inline void allocate4d(double ****&ptr, int di, int dj, int dk, int dl){

	int i,j,k,l;
	double *temp;

	ptr = (double ****) malloc(di * sizeof(double ***));
	FOR(i, 0, di){
		ptr[i] = (double ***) malloc(dj * sizeof(double **));
		FOR(j, 0, dj)
			ptr[i][j] = (double **) malloc(dk * sizeof(double *));
	}

	temp = (double *) malloc(di*dj*dk*dl * sizeof(double));
	FOR(i, 0, di){
		FOR(j, 0, dj){
			FOR(k, 0, dk){
				ptr[i][j][k] = temp;
				temp += dl;
			}
		}
	}
	printf("done allocate4d\n");

}
void ctoprim_test(){

	int i, j, k, l, dummy;
	int lo[3], hi[3];
	const int ng=4;
//	double u[40][40][40][5];
//	double q[40][40][40][6];
	double ****u, ****q;
	double dx[3], courno;

	int ng2;
	int lo2[3], hi2[3];
//	double u2[40][40][40][5];
//	double q2[40][40][40][6];
	double ****u2, ****q2;
	double dx2[3], courno2;

	allocate4d(u, 40, 40, 40, 5);
	allocate4d(q, 40, 40, 40, 6);
	allocate4d(u2, 40, 40, 40, 5);
	allocate4d(q2, 40, 40, 40, 6);

	FILE *fin = fopen("ctoprim_input", "r");
	FILE *fout = fopen("ctoprim_output", "r");

	fscanf(fin, "%d %d %d\n", &lo[0], &lo[1], &lo[2]);
	fscanf(fin, "%d %d %d\n", &hi[0], &hi[1], &hi[2]);
	printf("done lo-hi\n");

	lo[0] += ng; 	lo[1] += ng; 	lo[2] += ng;
	hi[0] += ng; 	hi[1] += ng; 	hi[2] += ng;
	DO(l, 0, 4){
		DO(k, lo[2]-ng, hi[2]+ng){
			DO(j, lo[1]-ng, hi[1]+ng){
				DO(i, lo[0]-ng, hi[0]+ng){
					fscanf(fin, "%le", &u[i][j][k][l]);
				}
			}
		}
	}
	printf("Done with u\n");
	DO(l, 0, 5){
		DO(k, lo[2]-ng, hi[2]+ng){
			DO(j, lo[1]-ng, hi[1]+ng){
				DO(i, lo[0]-ng, hi[0]+ng){
					fscanf(fin, "%le", &q[i][j][k][l]);
				}
			}
		}
	}
	printf("Done with q\n");
	fscanf(fin, "%le %le %le\n", &dx[0], &dx[1], &dx[2]);
	fscanf(fin, "%d\n", &dummy);
	fscanf(fin, "%le\n", &courno);
	fclose(fin);

	printf("before ctoprim\n");
	ctoprim(lo, hi, u, q, dx, ng, courno);
	printf("after ctoprim\n");

	fscanf(fout, "%d %d %d\n", &lo2[0], &lo2[1], &lo2[2]);
	fscanf(fout, "%d %d %d\n", &hi2[0], &hi2[1], &hi2[2]);
	DO(l, 0, 4){
		DO(k, lo[2]-ng, hi[2]+ng){
			DO(j, lo[1]-ng, hi[1]+ng){
				DO(i, lo[0]-ng, hi[0]+ng){
					fscanf(fout, "%le", &u2[i][j][k][l]);
				}
			}
		}
	}
	DO(l, 0, 5){
		DO(k, lo[2]-ng, hi[2]+ng){
			DO(j, lo[1]-ng, hi[1]+ng){
				DO(i, lo[0]-ng, hi[0]+ng){
					fscanf(fout, "%le", &q2[i][j][k][l]);
				}
			}
		}
	}
	fscanf(fout, "%le %le %le\n", &dx2[0], &dx2[1], &dx2[2]);
	fscanf(fout, "%d\n", &ng2);
	fscanf(fout, "%le\n", &courno2);
	fclose(fout);

	DO(i, 0, 2){

		if(lo[i] != lo2[i]+ng){
			printf("lo[%d] = %d != %d = lo2[%d]\n", i, lo[i], lo2[i], i);
			exit(1);
		}
		if(hi[i] != hi2[i]+ng){
			printf("hi[%d] = %d != %d = hi2[%d]\n", i, hi[i], hi2[i], i);
			exit(1);
		}
		if(!FEQ(dx[i], dx2[i])){
			printf("dx[%d] = %le != %le = dx2[%d]\n", i, dx[i], dx2[i], i);
			exit(1);
		}
	}
	if(ng != ng2){
		printf("ng = %d != %d = ng2\n", ng, ng2);
		exit(1);
	}
	if(!FEQ(courno, courno2)){
		printf("courno = %le != %le = courno2\n", courno, courno2);
//            exit(1);
	}
	DO(i, lo[0]-ng, hi[0]+ng){
		DO(j, lo[1]-ng, hi[1]+ng){
			DO(k, lo[2]-ng, hi[2]+ng){
				DO(l, 0, 4){
					if(!FEQ(u[i][j][k][l], u2[i][j][k][l])){
						printf("u[%d][%d][%d][%d] = %le != %le = u2[%d][%d][%d][%d]\n",
								i, j, k, l, u[i][j][k][l], u2[i][j][k][l], i, j, k, l);
						exit(1);
					}
				}
				DO(l, 0, 5){
					if(!FEQ(q[i][j][k][l], q2[i][j][k][l])){
						printf("q[%d][%d][%d][%d] = %le != %le = q2[%d][%d][%d][%d]\n",
								i, j, k, l, q[i][j][k][l], q2[i][j][k][l], i, j, k, l);
						exit(1);
					}
				}
			}
		}
	}
	printf("Correct!\n");
}

int main(int argc, char *argv[]){

	ctoprim_test();
	return 0;

}
