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
	FOR(l, 0, la){
		FOR(i, 0, dim[0]){
			FOR(j, 0, dim[1]){
				FOR(k, 0, dim[2]){
					if(!FEQ(a[l][i][j][k], a2[l][i][j][k])){
						printf("%s[%d][%d][%d][%d] = %le != %le = %s2[%d][%d][%d][%d]\n",
								name, l, i, j, k, a[l][i][j][k], a2[l][i][j][k], name, l, i, j, k);
						printf("diff = %le\n", a[l][i][j][k] - a2[l][i][j][k]);
						exit(1);
					}
				}
			}
		}
	}
}

//void check_4D_array( const char *name, double ****a, double ****a2, int dim[],  int la){
//
//	int i,j,k,l;
//	FOR(i, 0, dim[0]){
//		FOR(j, 0, dim[1]){
//			FOR(k, 0, dim[2]){
//				FOR(l, 0, la){
//					if(!FEQ(a[i][j][k][l], a2[i][j][k][l])){
//						printf("%s[%d][%d][%d][%d] = %le != %le = %s2[%d][%d][%d][%d]\n",
//								name, i, j, k, l, a[i][j][k][l], a2[i][j][k][l], name, i, j, k, l);
//						exit(1);
//					}
//				}
//			}
//		}
//	}
//}

