#include <stdio.h>
#include <cuda.h>

__global__ void fill(cudaPitchedPtr U){

}
int main(int argc, char *argv[]){

	struct cudaPitchedPtr U;
	struct cudaExtent ext;

	ext.width = 40;
	ext.height = 40;
	ext.depth = 40;
	printf("%15s pitch xsize ysize\n", "extent");

	for(ext.depth=500; ext.depth < 700; ext.depth+=100){
		for(ext.height=500; ext.height < 700; ext.height+=100){
			for(ext.width=500; ext.width < 700; ext.width+=100){
				cudaMalloc3D(&U, ext);
				printf("[%3d][%3d][%3d] %5d %5d %5d\n", ext.depth, ext.height, ext.width, U.pitch, U.xsize, U.ysize);
				cudaFree(U.ptr);
			}
		}
	}
	return 0;
}
