#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "header.h"
#include "util_cpu.h"

extern global_const_t h_const;
extern global_const_t *d_const_ptr;

void allocate_4D(double ****&ptr, int dim[], int dl){

	int j,k,l;
	int di=dim[0], dj=dim[1], dk=dim[2];
	double *temp;

	ptr = (double ****) malloc(dl * sizeof(double ***));
	FOR(l, 0, dl){
		ptr[l] = (double ***) malloc(dk * sizeof(double **));
		FOR(k, 0, dk)
			ptr[l][k] = (double **) malloc(dj * sizeof(double *));
	}

	temp = (double *) malloc(dl*dk*dj*di * sizeof(double));
	FOR(l, 0, dl){
		FOR(k, 0, dk){
			FOR(j, 0, dj){
				ptr[l][k][j] = temp;
				temp += di;
			}
		}
	}

}

void allocate_3D(double ***&ptr, int dim[]){
	int j,k;
	int di=dim[0], dj=dim[1], dk=dim[2];
	double *temp;

	ptr = (double ***) malloc(di * sizeof(double **));
	FOR(k, 0, dk){
		ptr[k] = (double **) malloc(dj * sizeof(double *));
	}

	// Allocate memory as a bulk
	temp = (double *) malloc(di*dj*dk * sizeof(double));
	FOR(k, 0, dk){
		FOR(j, 0, dj){
			ptr[k][j] = temp;
			temp += di;
		}
	}
}

void free_4D(double ****ptr, int dim[], int dl){
	int l,k;
	int dj=dim[1], dk=dim[2];

	free(ptr[0][0][0]);
	FOR(l, 0, dl){
		FOR(k, 0, dk)
			free(ptr[l][k]);
		free(ptr[l]);
	}
	free(ptr);
}

void free_3D(double ***ptr, int dim[]){
	int k;
	free(ptr[0][0]);
	FOR(k, 0, dim[2])
		free(ptr[k]);
	free(ptr);
}

void read_3D(FILE *f, double ****ptr, int dim[], int l){
	int i,j,k;
	FOR(k, 0, dim[2]){
		FOR(j, 0, dim[1]){
			FOR(i, 0, dim[0])
				fscanf(f, "%le", &ptr[l][k][j][i]);
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

void check_4D_array( const char *name, double ****a, double ****a2, int dim[],  int la){

	int i,j,k,l;
	int exp, exp2;
	double sig, sig2;
	FOR(l, 0, la){
		FOR(k, 0, dim[2]){
			FOR(j, 0, dim[1]){
				FOR(i, 0, dim[0]){
					if(!FEQ(a[l][k][j][i], a2[l][k][j][i])){
						printf("%s[%d][%d][%d][%d] = %le != %le = %s2[%d][%d][%d][%d]\n",
								name, l, i, j, k, a[l][k][j][i], a2[l][k][j][i], name, l, i, j, k);
						printf("diff = %le\n", a[l][k][j][i] - a2[l][k][j][i]);
						sig = frexp(a[l][k][j][i], &exp);
						sig2 = frexp(a2[l][k][j][i], &exp2);
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
		FOR(k, 0, NG){
			FOR(j, NG, dim[1]+NG){
				FOR(i, NG, dim[0]+NG){
					U[l][k][j][i] = U[l][k+dim[2]][j][i];
					U[l][k+dim[2]+NG][j][i] = U[l][k+NG][j][i];
				}
			}
		}
	}

	FOR(l, 0, NC){
		FOR(k, 0, dim_g[2]){
			FOR(j, 0, NG){
				FOR(i, NG, dim[0]+NG){
					U[l][k][j][i] = U[l][k][j+dim[1]][i];
					U[l][k][j+dim[1]+NG][i] = U[l][k][j+NG][i];
				}
			}
		}
	}

	FOR(l, 0, NC){
		FOR(k, 0, dim_g[2]){
			FOR(j, 0, dim_g[1]){
				FOR(i, 0, NG){
					U[l][k][j][i] = U[l][k][j][i+dim[0]];
					U[l][k][j][i+dim[0]+NG] = U[l][k][j][i+NG];
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
					fprintf(f, "%.17e\t", ptr[l][k][j][i]);
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
				fprintf(f, "%.17e\t", ptr[k][j][i]);
			fprintf(f, "\n");
		}
		fprintf(f, "\n");
	}
}

void set_3D(double val, double ***ptr, int dim[]){
	int i,j,k;
	FOR(k, 0, dim[2]){
		FOR(j, 0, dim[1]){
			FOR(i, 0, dim[0])
				ptr[k][j][i] = val;
		}
	}
}

void number_3D(double ***ptr, int dim[]){
	int i,j,k,count=0;
	FOR(k, 0, dim[2]){
		FOR(j, 0, dim[1]){
			FOR(i, 0, dim[0])
				ptr[k][j][i] = count++;
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
#define	cons(i,j,k,l)	cons[l][k][j][i]
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
		h_const.pitch[i]   = h_const.dim[i];
		h_const.pitch_g[i] = h_const.dim_g[i];
	}
	h_const.pitch[0]   = PAD(h_const.dim[0]);      // No need to pad Y and Z, padding X alone will make them aligned
	h_const.pitch_g[0] = PAD(h_const.dim_g[0]);    // No need to pad Y and Z, padding X alone will make them aligned
	printf("pitch:   %d %d %d\n", h_const.pitch[0], h_const.pitch[1], h_const.pitch[2]);
	printf("pitch_g: %d %d %d\n", h_const.pitch_g[0], h_const.pitch_g[1], h_const.pitch_g[2]);

	h_const.comp_offset_g_padded    = h_const.pitch_g[0] * h_const.pitch_g[1] * h_const.pitch_g[2];
	h_const.plane_offset_g_padded   = h_const.pitch_g[0] * h_const.pitch_g[1];
	h_const.comp_offset_padded      = h_const.pitch[0] * h_const.pitch[1] * h_const.pitch[2];
	h_const.plane_offset_padded     = h_const.pitch[0] * h_const.pitch[1];

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
