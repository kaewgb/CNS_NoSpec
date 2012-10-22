#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "util_cpu.h"

int main(int argc, char *argv[]){

	int i,j,k,l;
	int nc;
	int dim[3];
	FILE *fin;
	double ****U;

	if(argc < 2){
		printf("usage: %s <input> [<i>]\n", argv[0]);
		return 1;
	}

	fin = fopen(argv[1], "r");
	fscanf(fin, "%d", &nc);
	fscanf(fin, "%d %d %d", &dim[0], &dim[1], &dim[2]);

	allocate_4D(U, dim, nc);
	FOR(l, 0, nc){
		FOR(k, 0, dim[2]){
			FOR(j, 0, dim[1]){
				FOR(i, 0, dim[0]){
					fscanf(fin, "%le", &U[l][i][j][k]);
				}
			}
		}
	}
	fclose(fin);

	l = 1;
	i = 5;
	if( argc == 3 )
		i = atoi(argv[2]);
	fprintf(stderr, "i=%d\n", i);

	FOR(j, 0, dim[1]){
		FOR(k, 0, dim[2])
			printf("%12lf ", U[l][i][j][k]);
		printf("\n");
	}

	free_4D(U, dim, nc);
	return 0;
}
