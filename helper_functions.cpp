#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "helper_functions.h"

void allocate_4D(double ****&ptr, int dim[], int dl){

	int i,j,k,l;
	int di=dim[0], dj=dim[1], dk=dim[2];
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

}

void allocate_3D(double ***&ptr, int dim[]){
	int i,j,k;
	int di=dim[0], dj=dim[1], dk=dim[2];
	double *temp;

	ptr = (double ***) malloc(di * sizeof(double **));
	FOR(i, 0, di){
		ptr[i] = (double **) malloc(dj * sizeof(double *));
	}

	// Allocate memory as a bulk
	temp = (double *) malloc(di*dj*dk * sizeof(double));
	FOR(i, 0, di){
		FOR(j, 0, dj){
			ptr[i][j] = temp;
			temp += dk;
		}
	}
}


void free_4D(double ****ptr, int dim[]){
	int i,j,k;
	int di=dim[0], dj=dim[1], dk=dim[2];

	free(ptr[0][0][0]);
	FOR(i, 0, di){
		FOR(j, 0, dj){
			free(ptr[i][j]);
		}
		free(ptr[i]);
	}
	free(ptr);
}

void free_3D(double ***ptr, int dim[]){
	int i,j;
	free(ptr[0][0]);
	FOR(i, 0, dim[0])
		free(ptr[i]);
	free(ptr);
}

void read_3D(FILE *f, double ****ptr, int dim[], int l){
	int i,j,k;
	FOR(k, 0, dim[2]){
		FOR(j, 0, dim[1]){
			FOR(i, 0, dim[0])
				fscanf(f, "%le", &ptr[i][j][k][l]);
		}
	}
}


void check_double(double a, double b, const char *name){
	if(!FEQ(a, b)){
		printf("%s = %le != %le = %s2\n", name, a, b, name);
		exit(1);
	}
}

void check_lo_hi_ng_dx( int lo[],  int hi[],  int ng,  double dx[],
									  int lo2[], int hi2[], int ng2, double dx2[] ){
	int i;
	FOR(i, 0, 3){

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
}

void check_4D_arrays( int dim[],
							 double ****a, double ****a2, int la, const char *aname,
							 double ****b, double ****b2, int lb, const char *bname ){
	int i,j,k,l;
	FOR(i, 0, dim[0]){
		FOR(j, 0, dim[1]){
			FOR(k, 0, dim[2]){
				FOR(l, 0, la){
					if(!FEQ(a[i][j][k][l], a2[i][j][k][l])){
						printf("%s[%d][%d][%d][%d] = %le != %le = %s2[%d][%d][%d][%d]\n",
								aname, i, j, k, l, a[i][j][k][l], a2[i][j][k][l], aname, i, j, k, l);
						exit(1);
					}
				}
				FOR(l, 0, lb){
					if(!FEQ(b[i][j][k][l], b2[i][j][k][l])){
						printf("%s[%d][%d][%d][%d] = %le != %le = %s2[%d][%d][%d][%d]\n",
								bname, i, j, k, l, b[i][j][k][l], b2[i][j][k][l], bname, i, j, k, l);
						exit(1);
					}
				}
			}
		}
	}
}
