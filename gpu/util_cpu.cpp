#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "header.h"
#include "util_cpu.h"

extern global_const_t h_const;
extern global_const_t *d_const_ptr;

void allocate_4D(double ****&ptr, int dim[], int dl){

	int l,i,j;
	int di=dim[0], dj=dim[1], dk=dim[2];
	double *temp;

	ptr = (double ****) malloc(dl * sizeof(double ***));
	FOR(l, 0, dl){
		ptr[l] = (double ***) malloc(di * sizeof(double **));
		FOR(i, 0, di)
			ptr[l][i] = (double **) malloc(dj * sizeof(double *));
	}

	temp = (double *) malloc(dl*di*dj*dk * sizeof(double));
	FOR(l, 0, dl){
		FOR(i, 0, di){
			FOR(j, 0, dj){
				ptr[l][i][j] = temp;
				temp += dk;
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

void free_4D(double ****ptr, int dim[], int dl){
	int i,l;
	int di=dim[0], dj=dim[1];

	free(ptr[0][0][0]);
	FOR(l, 0, dl){
		FOR(i, 0, di)
			free(ptr[l][i]);
		free(ptr[l]);
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
				fscanf(f, "%le", &ptr[l][i][j][k]);
//				fscanf(f, "%le", &ptr[i][j][k][l]);
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
	int exp, exp2;
	double sig, sig2;
	FOR(l, 0, la){
		FOR(i, 0, dim[0]){
			FOR(j, 0, dim[1]){
				FOR(k, 0, dim[2]){
					if(!FEQ(a[l][i][j][k], a2[l][i][j][k])){
						printf("%s[%d][%d][%d][%d] = %le != %le = %s2[%d][%d][%d][%d]\n",
								name, l, i, j, k, a[l][i][j][k], a2[l][i][j][k], name, l, i, j, k);
						printf("diff = %le\n", a[l][i][j][k] - a2[l][i][j][k]);
						sig = frexp(a[l][i][j][k], &exp);
						sig2 = frexp(a2[l][i][j][k], &exp2);
						if(exp!=exp2){
							printf("exp = %d != %d = exp2\n", exp, exp2);
							printf("sig1 = %le, sig2 = %le\n", sig, sig2);
							exit(1);
						}
						if(!FEQ(sig, sig2)){
							printf("sig = %le != %le = sig2\n", sig, sig2);
							printf("diff = %le\n", sig - sig2);
							exit(1);
						}
					}
				}
			}
		}
	}
}

void fill_boundary(
	global_const_t h,	// Application parameters
	double ****U		// Array
){
	int i, j, k, l;
	static int *dim = NULL, *dim_g;
	if(dim == NULL){
		dim = h.dim;	dim_g = h.dim_g;
	}

	FOR(l, 0, NC){
		FOR(i, NG, dim[0]+NG){
			FOR(j, NG, dim[1]+NG){
				FOR(k, 0, NG){
					U[l][i][j][k] = U[l][i][j][k+dim[2]];
					U[l][i][j][k+dim[2]+NG] = U[l][i][j][k+NG];
				}
			}
		}
	}

	FOR(l, 0, NC){
		FOR(i, NG, dim[0]+NG){
			FOR(j, 0, NG){
				FOR(k, 0, dim_g[2]){
					U[l][i][j][k] = U[l][i][j+dim[1]][k];
					U[l][i][j+dim[1]+NG][k] = U[l][i][j+NG][k];
				}
			}
		}
	}

	FOR(l, 0, NC){
		FOR(i, 0, NG){
			FOR(j, 0, dim_g[1]){
				FOR(k, 0, dim_g[2]){
					U[l][i][j][k] = U[l][i+dim[0]][j][k];
					U[l][i+dim[0]+NG][j][k] = U[l][i+NG][j][k];
				}
			}
		}
	}
}

void print_4D(FILE *f, double ****ptr, int dim[], int dl){
	int i,j,k,l;
	FOR(l, 0, dl){
		FOR(k, 0, dim[2]){
			FOR(j, 0, dim[1]){
				FOR(i, 0, dim[0])
					fprintf(f, "%.17e\t", ptr[l][i][j][k]);
				fprintf(f, "\n");
			}
			fprintf(f, "\n");
		}
		fprintf(f, "\n");
	}
}

void print_3D(FILE *f, double ***ptr, int dim[]){
	int i,j,k;
	FOR(k, 0, dim[2]){
		FOR(j, 0, dim[1]){
			FOR(i, 0, dim[0])
				fprintf(f, "%.17e\t", ptr[i][j][k]);
			fprintf(f, "\n");
		}
		fprintf(f, "\n");
	}
}

void set_3D(double val, double ***ptr, int dim[]){
	int i,j,k;
	FOR(i, 0, dim[0]){
		FOR(j, 0, dim[1]){
			FOR(k, 0, dim[2])
				ptr[i][j][k] = val;
		}
	}
}

void number_3D(double ***ptr, int dim[]){
	int i,j,k,count=0;
	FOR(i, 0, dim[0]){
		FOR(j, 0, dim[1]){
			FOR(k, 0, dim[2])
				ptr[i][j][k] = count++;
		}
	}
}

double get_time()
{
    struct timeval t;
    gettimeofday( &t, NULL );
    return t.tv_sec + 1.0e-6 * t.tv_usec;
}

#define lo(i)			h.lo[i-1]
#define hi(i)			h.hi[i-1]
#define	dx(i)			h.dx[i-1]
#define	plo(i)			plo[i-1]
#define phi(i)			phi[i-1]
#define	scale(i)		scale[i-1]
#define	cons(i,j,k,l)	cons[l][i][j][k]
#define ROUND17(x)		(floor((x)*1.0E17)/1.0E17)

void init_data(global_const_t h, double ****cons){

	int i,j,k;
	double scale[3];
	double plo[3], phi[3]; //Problem's low and high (continuous space)
    double xloc,yloc,zloc,rholoc,eloc,uvel,vvel,wvel;
	double twopi = 2.0E0 * 3.141592653589793238462643383279502884197E0;

    DO(i, 1, 3){
		plo(i) = -0.1E0;
		phi(i) =  0.1E0;
		scale(i) = (phi(i)-plo(i))/twopi;
    }

    //#pragma omp PARALLEL DO PRIVATE(i,j,k,zloc,yloc,xloc,uvel,vvel,wvel,rholoc,eloc)
    DO(k, lo(3), hi(3)){
       zloc = (double)(k-h.ng)*dx(3)/scale(3);
       DO(j, lo(2), hi(2)){
          yloc = (double)(j-h.ng)*dx(2)/scale(2);
          DO(i, lo(1), hi(1)){
             xloc = (double)(i-h.ng)*dx(1)/scale(1);

             uvel   = 1.1E4*sin(1*xloc)*sin(2*yloc)*sin(3*zloc);
             vvel   = 1.0E4*sin(2*xloc)*sin(4*yloc)*sin(1*zloc);
             wvel   = 1.2E4*sin(3*xloc)*cos(2*yloc)*sin(2*zloc);
             rholoc = 1.0E-3 + 1.0E-5*sin(1*xloc)*cos(2*yloc)*cos(3*zloc);
             eloc   = 2.5E9  + 1.0E-3*sin(2*xloc)*cos(2*yloc)*sin(2*zloc);

             cons(i,j,k,irho) = rholoc;
             cons(i,j,k,imx)  = sin(2*yloc);//uvel;//rholoc*uvel;
             cons(i,j,k,imy)  = rholoc*vvel;
             cons(i,j,k,imz)  = rholoc*wvel;
             cons(i,j,k,iene) = rholoc*(eloc + (uvel*uvel+vvel*vvel+wvel*wvel)/2.0);

             ROUND17(cons(i,j,k,irho));
             ROUND17(cons(i,j,k,imx));
             ROUND17(cons(i,j,k,imy));
             ROUND17(cons(i,j,k,imz));
             ROUND17(cons(i,j,k,iene));

          }
       }
    }

}
#undef lo
#undef hi
#undef dx
#undef plo
#undef phi
#undef scale
#undef cons

void read_configurations(global_const_t &h_const, int argc, char *argv[]){
	int i;
	char *dest, *config_file_name;
	FILE *fin;

	if(argc == 2){
		config_file_name = (char *) malloc(200*sizeof(char));
		sprintf(config_file_name, "../testcases/%s_general_input", argv[1]);

		h_const.input_file_name = (char *) malloc(200*sizeof(char));
		h_const.output_file_name = (char *) malloc(200*sizeof(char));

		sprintf(h_const.input_file_name, "../testcases/%s_multistep_input", argv[1]);
		sprintf(h_const.output_file_name, "%s_multistep_output", argv[1]);
	}
	else{
		config_file_name = (char *) "../testcases/general_input";
		h_const.input_file_name = (char *) "../testcases/multistep_input";
		h_const.output_file_name = (char *) "multistep_output";
	}

	fin = fopen(config_file_name, "r");
	if(fin == NULL){
		printf("usage: %s [<config_file_name>]\n", argv[0]);
		exit(1);
	}

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
		h_const.lo[i] += h_const.ng;
		h_const.hi[i] += h_const.ng;
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
