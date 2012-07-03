#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "helper_functions.h"

void ctoprim_test();
void diffterm_test();

int main(int argc, char *argv[]){

//	ctoprim_test();
	diffterm_test();
	return 0;

}

void ctoprim_test(){

	int i, j, k, l, dummy, dim[3];
	int lo[3], hi[3];
	int ng=4;
	double ****u, ****q;
	double dx[3], courno;

	int ng2;
	int lo2[3], hi2[3];
	double ****u2, ****q2;
	double dx2[3], courno2;

	FILE *fin = fopen("ctoprim_input", "r");
	FILE *fout = fopen("ctoprim_output", "r");
	if(fin == NULL || fout == NULL){
		printf("Invalid input file\n");
		exit(1);
	}

	// Scanning input
	fscanf(fin, "%d %d %d\n", &lo[0], &lo[1], &lo[2]);
	fscanf(fin, "%d %d %d\n", &hi[0], &hi[1], &hi[2]);

	lo[0] += ng; 	lo[1] += ng; 	lo[2] += ng;
	hi[0] += ng; 	hi[1] += ng; 	hi[2] += ng;

	FOR(i, 0, 3)
		dim[i] = hi[i]-lo[i]+1 + 2*ng;

	allocate_4D(u, 	dim, 5); 	// [40][40][40][5]
	allocate_4D(q, 	dim, 6); 	// [40][40][40][6]
	allocate_4D(u2, dim, 5); 	// [40][40][40][5]
	allocate_4D(q2, dim, 6); 	// [40][40][40][6]

	FOR(l, 0, 5)
		read_3D(fin, u, dim, l);
	FOR(l, 0, 6)
		read_3D(fin, q, dim, l);

	fscanf(fin, "%le %le %le\n", &dx[0], &dx[1], &dx[2]);
	fscanf(fin, "%d\n", &dummy);
	fscanf(fin, "%le\n", &courno);
	fclose(fin);

	printf("Applying ctoprim()...\n");
	ctoprim(lo, hi, u, q, dx, ng, courno);

	// Scanning output to check
	fscanf(fout, "%d %d %d\n", &lo2[0], &lo2[1], &lo2[2]);
	fscanf(fout, "%d %d %d\n", &hi2[0], &hi2[1], &hi2[2]);
	FOR(l, 0, 5)
		read_3D(fout, u2, dim, l);
	FOR(l, 0, 6)
		read_3D(fout, q2, dim, l);

	fscanf(fout, "%le %le %le\n", &dx2[0], &dx2[1], &dx2[2]);
	fscanf(fout, "%d\n", &ng2);
	fscanf(fout, "%le\n", &courno2);
	fclose(fout);

	// Checking...
	check_lo_hi_ng_dx(lo, hi, ng, dx, lo2, hi2, ng2, dx2);
	check_double(courno, courno2, "courno");
	check_4D_arrays(dim, u, u2, 5, "u", q, q2, 6, "q");
	printf("Correct!\n");

	free_4D(u,  dim);		free_4D(q,  dim);
	free_4D(u2, dim);		free_4D(q2, dim);
}

void diffterm_test(){
	int dim[3];
	int i,j,k,l;

	int lo[3], hi[3], ng=4;
	double dx[3], eta, alam;
	double ****q, ****difflux;

	int lo2[3], hi2[3], ng2=4;
	double dx2[3], eta2, alam2;
	double ****q2, ****difflux2;

	FILE *fin 	= fopen("diffterm_input", "r");
	FILE *fout 	= fopen("diffterm_output", "r");
	if(fin == NULL || fout == NULL){
		printf("Invalid input!\n");
		exit(1);
	}

	// Scanning input
	fscanf(fin, "%d %d %d\n", &lo[0], &lo[1], &lo[2]);
	fscanf(fin, "%d %d %d\n", &hi[0], &hi[1], &hi[2]);
	fscanf(fin, "%d\n", &ng);
	fscanf(fin, "%le %le %le\n", &dx[0], &dx[1], &dx[2]);

	lo[0] += ng; 	lo[1] += ng; 	lo[2] += ng;
	hi[0] += ng; 	hi[1] += ng; 	hi[2] += ng;

	FOR(i, 0, 3)
		dim[i] = hi[i]-lo[i]+1 + 2*ng;

	allocate_4D(q, 		 	dim, 6); 	// [40][40][40][6]
	allocate_4D(difflux, 	dim, 5); 	// [40][40][40][5]
	allocate_4D(q2, 	 	dim, 6); 	// [40][40][40][6]
	allocate_4D(difflux2, 	dim, 5); 	// [40][40][40][5]

	FOR(l, 0, 6)
		read_3D(fin, q, dim, l);
	FOR(l, 0, 5)
		read_3D(fin, difflux, dim, l);

	fscanf(fin, "%le %le", &eta, &alam);
	fclose(fin);

	printf("Applying diffterm()...\n");
	diffterm(lo, hi, ng, dx, q, difflux, eta, alam);
	printf("After diffterm()\n");

	// Scanning output to check
	fscanf(fout, "%d %d %d\n", &lo2[0], &lo2[1], &lo2[2]);
	fscanf(fout, "%d %d %d\n", &hi2[0], &hi2[1], &hi2[2]);
	fscanf(fout, "%d\n", &ng2);
	fscanf(fout, "%le %le %le\n", &dx2[0], &dx2[1], &dx2[2]);

	FOR(l, 0, 6)
		read_3D(fout, q2, dim, l);
	FOR(l, 0, 5)
		read_3D(fout, difflux2, dim, l);

	fscanf(fout, "%le %le", &eta2, &alam2);
	fclose(fout);

	// Checking...
	check_lo_hi_ng_dx(lo, hi, ng, dx, lo2, hi2, ng2, dx2);
	check_4D_arrays(dim, q, q2, 6, "q", difflux, difflux2, 5, "difflux");
	check_double(eta,  eta2,  "eta");
	check_double(alam, alam2, "alam");

	free_4D(q,  dim);	free_4D(difflux,  dim);
	free_4D(q2, dim);	free_4D(difflux2, dim);

	printf("Correct!\n");
}

