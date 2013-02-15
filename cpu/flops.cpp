#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "header.h"
#include "helper_functions.h"

#define	N			2048
#define N_TILES		8

global_const_t h_const;

int main(int argc, char *argv[]){
	time_t t;
	int i,j,k,bi,bj,iter,tile;
	int tile_list[N_TILES] = {128, 256, 512, 768, 1024, 1536, 2048, 4096};

	double timer;
	double *a = (double *) malloc(N*N*sizeof(double));
	double *b = (double *) malloc(N*N*sizeof(double));
	srand((unsigned) time(&t));
	for(i=0;i<N*N;i++){
		a[i] = (double)(rand()%1000000)/1000.0;
		b[i] = (double)(rand()%1000000)/1000.0;
	}

	for(k=0;k<N_TILES;k++){
		tile=tile_list[k];

		timer = -get_time();
		for(iter=0;iter<100;iter++){
			for(bi=0;bi<N;bi+=tile){
				for(bj=0;bj<N;bj+=tile){
					for(i=bi;i<MIN(N, bi+tile);i++){
						for(j=bj;j<MIN(N, bj+tile);j++){
							a[i*N+j] *= b[i*N+j];
						}
					}
				}
			}
		}
		timer += get_time();
		printf("tile=%4d: %lf s, %lf GFLOPS\n", tile,  timer, (double)(N*N*100)/timer/1.0E9);
	}

	return 0;
}
