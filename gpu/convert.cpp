#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "util_cpu.h"

int main(int argc, char *argv[]){

	FILE *fin, *fout;
	int l, nc;
	int dim_g[3];
	double ****U;

	if(argc != 2){
		printf("usage: %s <input>\n", argv[0]);
		exit(1);
	}
	fin = fopen(argv[1], "r");
	fscanf(fin, "%d %d %d %d", &nc, &dim_g[0], &dim_g[1], &dim_g[2]);

	allocate_4D(U, dim_g, nc);
	FOR(l, 0, nc)
		read_3D(fin, U, dim_g, l);
	fclose(fin);

	fout = fopen("converted", "w");
	fprintf(fout, "%d\n%d %d %d\n", nc, dim_g[0], dim_g[1], dim_g[2]);
	print_4D(fout, U, dim_g, nc);
	fclose(fout);

	free_4D(U, dim_g, nc);
	return 0;
}
