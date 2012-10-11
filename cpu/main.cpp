#include "header.h"
#include "helper_functions.h"

global_const_t h_const;

int main(int argc, char *argv[]){

	FILE *fin, *fout;
	int i, l, *dim, *dim_g;
	double ****U, ****Unew, ****Q, ****D, ****F;
	double dt, total_time;

	read_configurations(h_const);
	allocate_variables(U, Unew, Q, D, F);
	dim = h_const.dim;
	dim_g = h_const.dim_g;

	// Initiation
	fin = fopen("../testcases/multistep_input", "r");
	FOR(l, 0, h_const.nc)
		read_3D(fin, U, dim_g, l);
	fclose(fin);

	total_time = -get_time();
	FOR(i, 0, h_const.nsteps)
		new_advance(U, Unew, Q, D, F, dt);
	total_time += get_time();
	printf("Total time: %lf\n", total_time);

	fout = fopen("output", "w");
	fprintf(fout, "%d\n", h_const.nc);
	fprintf(fout, "%d %d %d\n", dim_g[0], dim_g[1], dim_g[2]);
	print_4D(fout, U, h_const.dim_g, h_const.nc);
	fclose(fout);

//	fout = fopen("ctoprim.out", "w");
//	fprintf(fout, "%d\n", h_const.nc+1);
//	fprintf(fout, "%d %d %d\n", dim_g[0], dim_g[1], dim_g[2]);
//	print_4D(fout, Q, h_const.dim_g, h_const.nc+1);
//	fclose(fout);

//	fout = fopen("diffterm.out", "w");
//	fprintf(fout, "%d\n", h_const.nc);
//	fprintf(fout, "%d %d %d\n", dim[0], dim[1], dim[2]);
//	print_4D(fout, D, h_const.dim, h_const.nc);
//	fclose(fout);

	free_variables(U, Unew, Q, D, F);
	return 0;

}

