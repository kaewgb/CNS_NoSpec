#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "helper_functions.h"
#define CUDA_SAFE_CALL( call )                               						\
{                                                              						\
    cudaError_t err = call;                                                       	\
    if( cudaSuccess != err) {                                                    	\
		fprintf(stderr, "Cuda error in call at file '%s' in line %i : %s.\n",     	\
				__FILE__, __LINE__, cudaGetErrorString( err) );             		\
		exit(-1);                                                             		\
	}                                         										\
}

void gpu_allocate_3D(double *&d_ptr, int dim[]){
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_ptr, dim[0]*dim[1]*dim[2] * sizeof(double)));
}
void gpu_copy_to_host_3D(double ***host, double *dev, int dim[]){
	CUDA_SAFE_CALL(cudaMemcpy(host[0][0], dev, dim[0]*dim[1]*dim[2] * sizeof(double), cudaMemcpyDeviceToHost));
}
void gpu_free_3D(double *d_ptr){
	CUDA_SAFE_CALL(cudaFree(d_ptr));
}

void gpu_allocate_4D(double *&d_ptr, int dim[], int dl){
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_ptr, dim[0]*dim[1]*dim[2]*dl * sizeof(double)));
}

void gpu_copy_from_host_4D(double *dev, double ****host, int dim[], int dl){
	CUDA_SAFE_CALL(cudaMemcpy(dev, host[0][0][0], dim[0]*dim[1]*dim[2]*dl * sizeof(double), cudaMemcpyHostToDevice));
}

void gpu_copy_to_host_4D(double ****host, double *dev, int dim[], int dl){
	CUDA_SAFE_CALL(cudaMemcpy(host[0][0][0], dev, dim[0]*dim[1]*dim[2]*dl * sizeof(double), cudaMemcpyDeviceToHost));
}

void gpu_free_4D(double *d_ptr){
	CUDA_SAFE_CALL(cudaFree(d_ptr));
}

#define d_ptr(l,i,j,k)	d_ptr[(l)*g->comp_offset_g + (i)*g->plane_offset_g + (j)*g->dim_g[2] + (k)]

__device__ kernel_const_t k_const;
__global__ void gpu_fill_boundary_z_kernel(
	global_const_t *g, 		// i: Global Constants
	double *d_ptr			// i/o:	Device Pointer
){
	int i,j,k,l;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i < g->dim[0] && j < g->dim[1]){
		i += g->ng;
		j += g->ng;
		FOR(l, 0, g->nc){
			FOR(k, 0, g->ng){
				d_ptr(l,i,j,k) 					= d_ptr(l,i,j,k+g->dim[2]);
				d_ptr(l,i,j,k+g->dim[2]+g->ng)	= d_ptr(l,i,j,k+g->ng);
			}
		}
	}

}
__global__ void gpu_fill_boundary_y_kernel(
	global_const_t *g, 		// i: Global Constants
	double *d_ptr			// i/o:	Device Pointer
){
	int i,j,k,l;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	k = blockIdx.y * blockDim.y + threadIdx.y;

	if(i < g->dim[0] && k < g->dim_g[2]){
		i += g->ng;
		FOR(l, 0, g->nc){
			FOR(j, 0, g->ng){
				d_ptr(l,i,j,k) 					= d_ptr(l,i,j+g->dim[1],k);
				d_ptr(l,i,j+g->dim[1]+g->ng,k)	= d_ptr(l,i,j+g->ng,k);
			}
		}
	}

}
__global__ void gpu_fill_boundary_x_kernel(
	global_const_t *g, 		// i: Global Constants
	double *d_ptr			// i/o:	Device Pointer
){
	int i,j,k,l;
	j = blockIdx.x * blockDim.x + threadIdx.x;
	k = blockIdx.y * blockDim.y + threadIdx.y;

	if(j < g->dim_g[1] && k < g->dim_g[2]){
		FOR(l, 0, g->nc){
			FOR(i, 0, g->ng){
				d_ptr(l,i,j,k) 					= d_ptr(l,i+g->dim[0],j,k);
				d_ptr(l,i+g->dim[0]+g->ng,j,k)	= d_ptr(l,i+g->ng,j,k);
			}
		}
	}
}
#undef	d_ptr

void gpu_fill_boundary(
	global_const_t &h_const,	// i:	Global Constants
	global_const_t *d_const,	// i:	Device Pointer to Global Constants
	double *d_ptr		 		// i/o: Device Pointer
){

	dim3 block_dim(16, 16);
	dim3 grid_dim(CEIL(h_const.dim[0], 16), CEIL(h_const.dim[1], 16));

	gpu_fill_boundary_z_kernel<<<grid_dim, block_dim>>>(d_const, d_ptr);

	grid_dim.y = CEIL(h_const.dim_g[2], 16);
	gpu_fill_boundary_y_kernel<<<grid_dim, block_dim>>>(d_const, d_ptr);

	grid_dim.x = CEIL(h_const.dim_g[1], 16);
	gpu_fill_boundary_x_kernel<<<grid_dim, block_dim>>>(d_const, d_ptr);
}

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
	double ****U,	// Array
	int dim[],		// Dimensions (ghost cells excluded)
	int dim_ng[]	// Dimensions (ghost cells included)
){
	int i, j, k, l;
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
				FOR(k, 0, dim_ng[2]){
					U[l][i][j][k] = U[l][i][j+dim[1]][k];
					U[l][i][j+dim[1]+NG][k] = U[l][i][j+NG][k];
				}
			}
		}
	}

	FOR(l, 0, NC){
		FOR(i, 0, NG){
			FOR(j, 0, dim_ng[1]){
				FOR(k, 0, dim_ng[2]){
					U[l][i][j][k] = U[l][i+dim[0]][j][k];
					U[l][i+dim[0]+NG][j][k] = U[l][i+NG][j][k];
				}
			}
		}
	}
}

void fill_boundary_test(
	global_const_t h_const, // i: Global struct containing application parameters
	global_const_t *d_const	// i: Device pointer to global struct containing application paramters
){
	int i, l, n;
	int nc, dim[3], dim_g[3];
	double dt, dt2, dx[DIM], cfl, eta, alam;
	double ****U, ****U2;
	double *d_u;
	FILE *fin, *fout;

	nc = h_const.nc;
	dim[0] = dim[1] = dim[2] = h_const.ncells;
	dim_g[0] = dim_g[1] = dim_g[2] = h_const.ncells+h_const.ng+h_const.ng;

	// Allocation
	allocate_4D(U, dim_g, nc);
	allocate_4D(U2, dim_g, nc);
	gpu_allocate_4D(d_u, dim_g, 5);

	// Initiation
	fin = fopen("../testcases/advance_input", "r");
	FOR(l, 0, nc)
		read_3D(fin, U, dim_g, l);

	fscanf(fin, "%le", &dt);
	FOR(i, 0, 3)
		fscanf(fin, "%le", &dx[i]);
	fscanf(fin, "%le", &cfl);
	fscanf(fin, "%le", &eta);
	fscanf(fin, "%le", &alam);
	fclose(fin);

	gpu_copy_from_host_4D(d_u, U, dim_g, 5);

	printf("Applying fill_boundary()...\n");
//	fill_boundary(U, dim, dim_g);
	gpu_fill_boundary(h_const, d_const, d_u);

	gpu_copy_to_host_4D(U, d_u, dim_g, 5);
	fout=fopen("../testcases/fill_boundary_output", "r");
	FOR(l, 0, nc)
		read_3D(fout, U2, dim_g, l);
	check_4D_array("U", U, U2, dim_g, nc);

	fclose(fout);
	printf("Correct!\n");

	// Free memory
	free_4D(U, dim_g, nc);
	free_4D(U2, dim_g, nc);
	gpu_free_4D(d_u);
}
