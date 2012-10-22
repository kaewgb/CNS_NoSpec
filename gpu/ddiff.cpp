#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "header.h"

int main(int argc, char *argv[]){
	int i,j,k,l,n;
	int dim1[3], dim2[3];
	int nc1, nc2, exp1, exp2;
	int layer=-1;
	double u1, u2, sig1, sig2, count=0.0;
	double diff, diff_sig, diff_exp;
	double sum=0.0, min=1.0e100, max=0.0;
	double sum_sig=0.0, min_sig=1.0e100, max_sig=0.0;

	if(argc < 3 || argc > 4){
		printf("usage: %s <file1> <file2> [<l -- check for specific layer>]\n", argv[0]);
		return 0;
	}
	if(argc == 4){
		layer = atoi(argv[3]);
		printf("check for layer=%d...\n", layer);
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

					if(layer < 0 || l == layer){
						sig1 = frexp(u1, &exp1);
						sig2 = frexp(u2, &exp2);

						diff = fabs(u1-u2);
						diff_sig = fabs(sig1-sig2);
						diff_exp = diff_sig * pow(2, fabs((double) exp1-exp2));
						if(diff_exp > 2.0 && fabs(u1-u2) > 0.01){
							printf("[%d][%d][%d][%d]: %le vs %le, diff=%lf\n", l,i,j,k, u1, u2, diff);
							return 1;
						}
						else
							printf("[%d][%d][%d][%d]: %le vs %le, diff=%lf\n", l,i,j,k, u1, u2, diff);

						sum += diff;
						min = MIN(min, diff);
						max = MAX(max, diff);

						sum_sig += diff_sig;
						min_sig = MIN(min_sig, diff_sig);
						max_sig = MAX(max_sig, diff_sig);
					}
				}
			}
		}
	}

	count = nc1 * dim1[0] * dim1[1] * dim1[2];
	printf("Average diff:         %.18lf\n", sum/count);
	printf("Minimum diff:         %.18lf\n", min);
	printf("Maximum diff:         %.18lf\n", max);
	printf("Average diff_sig:     %.18lf\n", sum_sig/count);
	printf("Minimum diff_sig:     %.18lf\n", min_sig);
	printf("Maximum diff_sig:     %.18lf\n", max_sig);
	fclose(f1);
	fclose(f2);
	return 0;
}
