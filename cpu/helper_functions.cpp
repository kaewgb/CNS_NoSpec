#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "helper_functions.h"

extern global_const_t h_const;

void allocate_variables(double ****&U, double ****&Unew, double ****&Q, double ****&D, double ****&F){
	int i, nc, dim[3], dim_g[3];

	nc = h_const.nc;
	FOR(i, 0, 3){
		dim[i] = h_const.dim[i];
		dim_g[i] = h_const.dim_g[i];
	}
	allocate_4D(U,  	dim_g, 	nc);
	allocate_4D(Unew,  	dim_g, 	nc);
	allocate_4D(Q,  	dim_g, 	nc+1);
	allocate_4D(D,  	dim, 	nc);
	allocate_4D(F, 		dim, 	nc);
}

void free_variables(double ****U, double ****Unew, double ****Q, double ****D, double ****F){
	int i, nc, dim[3], dim_g[3];
	nc = h_const.nc;
	FOR(i, 0, 3){
		dim[i] = h_const.dim[i];
		dim_g[i] = h_const.dim_g[i];
	}
	free_4D(U,  	dim_g);
	free_4D(Unew,  	dim_g);
	free_4D(Q,  	dim_g);
	free_4D(D,  	dim);
	free_4D(F, 		dim);
}

void read_configurations(global_const_t &h_const){
	int i;
	FILE *fin = fopen("../testcases/general_input", "r");

	fscanf(fin, "%d", &h_const.ng);
	fscanf(fin, "%d", &h_const.nc);
	fscanf(fin, "%d", &h_const.ncells);
	fscanf(fin, "%d %d %d", &h_const.lo[0], &h_const.lo[1], &h_const.lo[2]);
	fscanf(fin, "%d %d %d", &h_const.hi[0], &h_const.hi[1], &h_const.hi[2]);
	fscanf(fin, "%le %le %le", &h_const.dx[0], &h_const.dx[1], &h_const.dx[2]);
	fscanf(fin, "%le", &h_const.cfl);
	fscanf(fin, "%le", &h_const.eta);
	fscanf(fin, "%le", &h_const.alam);
	fscanf(fin, "%d", &h_const.nsteps);
	fscanf(fin, "%le", &h_const.dt);
	fclose(fin);

	FOR(i, 0, 3){
		h_const.dim[i] 		= h_const.hi[i] - h_const.lo[i] + 1;
		h_const.dim_g[i] 	= h_const.hi[i] - h_const.lo[i] + 1 + h_const.ng + h_const.ng;
	}
	h_const.comp_offset_g  = h_const.dim_g[0] * h_const.dim_g[1] * h_const.dim_g[2];
	h_const.comp_offset    = h_const.dim[0]   * h_const.dim[1]   * h_const.dim[2];
	h_const.plane_offset_g = h_const.dim_g[1] * h_const.dim_g[2];
	h_const.plane_offset   = h_const.dim[1]   * h_const.dim[2];

	FOR(i, 0, 3)
		h_const.dxinv[i] = 1.0E0/h_const.dx[i];

	h_const.ALP	=  0.8E0;
	h_const.BET	= -0.2E0;
	h_const.GAM	=  4.0E0/105.0E0;
	h_const.DEL	= -1.0E0/280.0E0;

	h_const.OneThird		= 1.0E0/3.0E0;
	h_const.TwoThirds		= 2.0E0/3.0E0;
	h_const.FourThirds		= 4.0E0/3.0E0;
	h_const.OneQuarter    	= 1.E0/4.E0;
    h_const.ThreeQuarters 	= 3.E0/4.E0;

	h_const.CENTER		= -205.0E0/72.0E0;
	h_const.OFF1		=  8.0E0/5.0E0;
	h_const.OFF2 		= -0.2E0;
	h_const.OFF3		=  8.0E0/315.0E0;
	h_const.OFF4		= -1.0E0/560.0E0;
}

void print_4D(FILE *f, double ****ptr, int dim[], int dl){
	int i,j,k,l;
	FOR(l, 0, dl){
		FOR(k, 0, dim[2]){
			FOR(j, 0, dim[1]){
				FOR(i, 0, dim[0])
					fprintf(f, "%.17e\t", ptr[i][j][k][l]);
				fprintf(f, "\n");
			}
			fprintf(f, "\n");
		}
		fprintf(f, "\n");
	}
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

void allocate_3D(double ***&ptr, int dim[]){
	int i,j;
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
	int i,j;
	int di=dim[0], dj=dim[1];

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
	int i;
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

void check_4D_array( const char *name, double ****a, double ****a2, int dim[],  int la){

	int i,j,k,l;
	FOR(i, 0, dim[0]){
		FOR(j, 0, dim[1]){
			FOR(k, 0, dim[2]){
				FOR(l, 0, la){
					if(!FEQ(a[i][j][k][l], a2[i][j][k][l])){
						printf("%s[%d][%d][%d][%d] = %le != %le = %s2[%d][%d][%d][%d]\n",
								name, i, j, k, l, a[i][j][k][l], a2[i][j][k][l], name, i, j, k, l);
						exit(1);
					}
				}
			}
		}
	}
}

void fill_boundary(
	double ****U,	// Array
	int dim[],		// Dimensions (ghost cells excluded)
	int dim_ng[]	// Dimensions (ghost cells included)
){
	int i, j, k, l;
	FOR(i, NG, dim[0]+NG){
		FOR(j, NG, dim[1]+NG){
			FOR(k, 0, NG){
				FOR(l, 0, NC){
					U[i][j][k][l] = U[i][j][k+dim[2]][l];
					U[i][j][k+dim[2]+NG][l] = U[i][j][k+NG][l];
				}
			}
		}
	}
	FOR(i, NG, dim[0]+NG){
		FOR(j, 0, NG){
			FOR(k, 0, dim_ng[2]){
				FOR(l, 0, NC){
					U[i][j][k][l] = U[i][j+dim[1]][k][l];
					U[i][j+dim[1]+NG][k][l] = U[i][j+NG][k][l];
				}
			}
		}
	}
	FOR(i, 0, NG){
		FOR(j, 0, dim_ng[1]){
			FOR(k, 0, dim_ng[2]){
				FOR(l, 0, NC){
					U[i][j][k][l] = U[i+dim[0]][j][k][l];
					U[i+dim[0]+NG][j][k][l] = U[i+NG][j][k][l];
				}
			}
		}
	}
}
