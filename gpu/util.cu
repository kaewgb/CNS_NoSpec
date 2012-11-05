#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "header.h"
#include "util.h"
#define	ROUND16(x)	((((int)(x+15))>>4)<<4)
#define CUDA_SAFE_CALL( call )                               						\
{                                                              						\
    cudaError_t err = call;                                                       	\
    if( cudaSuccess != err) {                                                    	\
		fprintf(stderr, "Cuda error in call at file '%s' in line %i : %s.\n",     	\
				__FILE__, __LINE__, cudaGetErrorString( err) );             		\
		exit(-1);                                                             		\
	}                                         										\
}

extern global_const_t h_const;
extern global_const_t *d_const_ptr;


void gpu_allocate_3D(double *&d_ptr, int dim[]){
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_ptr, dim[0]*dim[1]*dim[2] * sizeof(double)));
}
void gpu_copy_from_host_3D(double *dev, double ***host, int dim[]){
	CUDA_SAFE_CALL(cudaMemcpy(dev, host[0][0], dim[0]*dim[1]*dim[2] * sizeof(double), cudaMemcpyHostToDevice));
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

#define d_ptr(l,k,j,i)	d_ptr[(l)*g->comp_offset_g_padded + (k)*g->plane_offset_g_padded + (j)*g->pitch_g[0] + (i)]

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
				d_ptr(l,k,j,i) 					= d_ptr(l,k+g->dim[2],j,i);
				d_ptr(l,k+g->dim[2]+g->ng,j,i)	= d_ptr(l,k+g->ng,j,i);
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
				d_ptr(l,k,j,i) 					= d_ptr(l,k,j+g->dim[1],i);
				d_ptr(l,k,j+g->dim[1]+g->ng,i)	= d_ptr(l,k,j+g->ng,i);
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
				d_ptr(l,k,j,i) 					= d_ptr(l,k,j,i+g->dim[0]);
				d_ptr(l,k,j,i+g->dim[0]+g->ng)	= d_ptr(l,k,j,i+g->ng);
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

void copy_configurations(global_const_t h_const, global_const_t *d_const_ptr){

	char *dest;
	cudaMemcpy(d_const_ptr, &h_const, sizeof(global_const_t), cudaMemcpyHostToDevice);

	dest = (char *)d_const_ptr + ((char *)&h_const.lo - (char *)&h_const);
	cudaMemcpy((int *) dest, h_const.lo, 3*sizeof(int), cudaMemcpyHostToDevice);
	dest = (char *)d_const_ptr + ((char *)&h_const.hi - (char *)&h_const);
	cudaMemcpy((int *) dest, h_const.hi, 3*sizeof(int), cudaMemcpyHostToDevice);
	dest = (char *)d_const_ptr + ((char *)&h_const.dim - (char *)&h_const);
	cudaMemcpy((int *) dest, h_const.dim, 3*sizeof(int), cudaMemcpyHostToDevice);
	dest = (char *)d_const_ptr + ((char *)&h_const.dim_g - (char *)&h_const);
	cudaMemcpy((int *) dest, h_const.dim_g, 3*sizeof(int), cudaMemcpyHostToDevice);

	dest = (char *)d_const_ptr + ((char *)&h_const.dx - (char *)&h_const);
	cudaMemcpy((int *) dest, h_const.dx, 3*sizeof(double), cudaMemcpyHostToDevice);
	dest = (char *)d_const_ptr + ((char *)&h_const.dxinv - (char *)&h_const);
	cudaMemcpy((int *) dest, h_const.dxinv, 3*sizeof(double), cudaMemcpyHostToDevice);
}

void allocate_variables(
	double ****&U, double ****&Unew, double ****&Q, double ****&D, double ****&F,
	double *&d_U, double *&d_Unew, double *&d_Q, double *&d_D, double *&d_F,
	bool gpu /* == true by default */, bool cpu
){
	int i, nc, *pitch=h_const.pitch, *pitch_g=h_const.pitch_g;
	char *dest;

	nc = h_const.nc;
	allocate_4D(U,  	pitch_g, 	nc);
	allocate_4D(Unew,  	pitch_g, 	nc);
	allocate_4D(Q,  	pitch_g, 	nc+1);
	allocate_4D(D,  	pitch, 	nc);
	allocate_4D(F, 		pitch, 	nc);

	if(gpu){
		gpu_allocate_4D(d_U, 	pitch_g, 	nc);
		gpu_allocate_4D(d_Unew, pitch_g, 	nc);
		gpu_allocate_4D(d_Q, 	pitch_g, 	nc+1);
		gpu_allocate_4D(d_D, 	pitch, 	nc);
		gpu_allocate_4D(d_F, 	pitch, 	nc);

		dest = (char *)d_const_ptr + ((char *)&h_const.temp - (char *)&h_const);

		FOR(i, 0, MAX_TEMP)
			gpu_allocate_3D(h_const.temp[i], pitch_g);
		cudaMemcpy((double *) dest, h_const.temp, MAX_TEMP*sizeof(double *), cudaMemcpyHostToDevice);
	}
	if(cpu){
		DO(i, 0, WZ)
			allocate_3D(h_const.cpu_temp[i], pitch_g);
	}
}

void free_variables(
	double ****U, double ****Unew, double ****Q, double ****D, double ****F,
	double *d_U, double *d_Unew, double *d_Q, double *d_D, double *d_F,
	bool gpu /* == true by default */, bool cpu
){
	int i, nc, *pitch=h_const.pitch, *pitch_g=h_const.pitch_g;
	nc = h_const.nc;

	free_4D(U,  	pitch_g, 	nc);
	free_4D(Unew,  	pitch_g, 	nc);
	free_4D(Q,  	pitch_g, 	nc+1);
	free_4D(D,  	pitch, 	nc);
	free_4D(F, 		pitch, 	nc);

	if(gpu){
		gpu_free_4D(d_U);
		gpu_free_4D(d_Unew);
		gpu_free_4D(d_Q);
		gpu_free_4D(d_D);
		gpu_free_4D(d_F);

		FOR(i, 0, MAX_TEMP)
			gpu_free_3D(h_const.temp[i]);
	}
	if(cpu){
		DO(i, 0, WZ)
			free_3D(h_const.cpu_temp[i], pitch_g);
	}
}

