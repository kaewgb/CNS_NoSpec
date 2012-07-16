#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define	CELLS	64
#define	NG		4
#define	NC		5
#define	DIM		(CELLS + 2*NG)

#define FOR(x, y, z)	for(x=y; x<(int)(z); x++)
#define FEQ(x, y)		((fabs((x)-(y)) < 0.000001)? 1:0)


void allocate_4D(double ****&ptr, int dim[], int dl);

int main(int argc, char *argv[]){
	printf("hey\n");
	int i,j,k,l;
	double ****b, ****a;
	int dim[3] = {DIM, DIM, DIM};

	allocate_4D(b, dim, NC);
	allocate_4D(a, dim, NC);

	FILE *fb, *fa, *fout;
	fb = fopen("../fortran90/U_before", "r");
	fa = fopen("../fortran90/U_after", "r");
	fout = fopen("mask", "w");

	FOR(l, 0, NC){
		FOR(k, 0, DIM){
			FOR(j, 0, DIM){
				FOR(i, 0, DIM){
					fscanf(fb, "%le", &b[i][j][k][l]);
					fscanf(fa, "%le", &a[i][j][k][l]);
				}
			}
		}
	}

	FOR(i, 0, DIM){
		FOR(j, 0, DIM){
			FOR(k, 0, DIM){
				fprintf(fout, "%d", FEQ(b[i][j][k][0], a[i][j][k][0]));
			}
			fprintf(fout, "\n");
		}
		fprintf(fout, "\n");
	}

	fclose(fb);
	fclose(fa);
	fclose(fout);
	return 0;
}

void allocate_4D(double ****&ptr, int dim[], int dl){

	int i,j,k;
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
