#include <stdio.h>
#include <math.h>
#include "header.h"

int main(int argc, char *argv[]){
	int i,j,k,l,n;
	int dim1[3], dim2[3];
	int nc1, nc2, exp1, exp2;
	double u1, u2, sig1, sig2;
	double diff, sum=0.0;
	double count=0.0, min=1.0e100, max=0.0;
	if(argc != 3){
		printf("usage: %s <file1> <file2>\n", argv[0]);
		return 0;
	}

	FILE *f1, *f2;
	f1 = fopen(argv[1], "r");
	f2 = fopen(argv[2], "r");

	fscanf(f1, "%d", &nc1);
	fscanf(f2, "%d", &nc2);
	if(nc1 != nc2){
		printf("nc1 = %d != %d = nc2\nAbort...\n", nc1, nc2);
		return 1;
	}

	fscanf(f1, "%d %d %d", &dim1[0], &dim1[1], &dim1[2]);
	fscanf(f2, "%d %d %d", &dim2[0], &dim2[1], &dim2[2]);
	if(dim1[0] != dim2[0] || dim1[1] != dim2[1] || dim1[2] != dim2[2]){
		printf("Dimension mismatched. (%d, %d, %d) != (%d, %d, %d)\nAbort...\n", dim1[0], dim1[1], dim1[2], dim2[0], dim2[1], dim2[2]);
		return 1;
	}

	FOR(l, 0, nc1){
		FOR(k, 0, dim1[2]){
			FOR(j, 0, dim1[1]){
				FOR(i, 0, dim1[0]){

					fscanf(f1, "%le", &u1);
					fscanf(f2, "%le", &u2);

					sig1 = frexp(u1, &exp1);
					sig2 = frexp(u2, &exp2);

					diff = fabs(sig1-sig2) * pow(2, fabs((double) exp1-exp2));
					sum += diff;
					min = MIN(min, diff);
					max = MAX(max, diff);

					if(diff > 10.0 && fabs(u1-u2) > 0.001){
						printf("[%d][%d][%d][%d]: %le vs %le, diff=%lf\n", l,i,j,k, u1, u2, diff);
						return 1;
					}
				}
			}
		}
	}

	count = nc1 * dim1[0] * dim1[1] * dim1[2];
	printf("Average diff:         %lf\n", sum/count);
	printf("Minimum diff:         %lf\n", min);
	printf("Maximum diff:         %lf\n", max);
	fclose(f1);
	fclose(f2);
	return 0;
}
