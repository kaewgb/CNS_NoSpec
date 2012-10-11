#include <stdio.h>
#include <math.h>
#include "header.h"

int main(int argc, char *argv[]){
	int i,j,k,l,n;
	int dim1[3], dim2[3];
	int nc1, nc2;
	double u1, u2;
	double diff;
	double dummy;
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

	diff = 0.0;
	FOR(l, 0, nc1){
		FOR(k, 0, dim1[2]){
			FOR(j, 0, dim1[1]){
				FOR(i, 0, dim1[0]){
					fscanf(f1, "%le", &u1);
					fscanf(f2, "%le", &u2);
					diff += fabs(u1-u2);
				}
			}
		}
	}

	printf("Average diff: %lf\n", diff);
	fclose(f1);
	fclose(f2);
	return 0;
}
