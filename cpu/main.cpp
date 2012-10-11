#include "header.h"
#include "helper_functions.h"

global_const_t h_const;

int main(int argc, char *argv[]){

	FILE *fin, *fout;
	int i, l, *dim, *dim_g;
	double ****U, ****Unew, ****Q, ****D, ****F;
	double dt;

	read_configurations(h_const);
	allocate_variables(U, Unew, Q, D, F);
	dim_g = h_const.dim_g;

	// Initiation
	fin = fopen("../testcases/multistep_input", "r");
	FOR(l, 0, h_const.nc)
		read_3D(fin, U, dim_g, l);
	fclose(fin);

	FOR(i, 0, h_const.nsteps)
		new_advance(U, Unew, Q, D, F, dt);

	fout = fopen("output", "w");
	fprintf(fout, "%d\n", h_const.nc);
	fprintf(fout, "%d %d %d\n", dim_g[0], dim_g[1], dim_g[2]);
	print_4D(fout, U, h_const.dim_g, h_const.nc);
	fclose(fout);

	free_variables(U, Unew, Q, D, F);
	return 0;

}

