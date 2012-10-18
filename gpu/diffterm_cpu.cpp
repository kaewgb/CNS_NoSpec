#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include "helper_functions.h"

#define	lo(i)		lo[i]
#define	hi(i)		hi[i]
#define dxinv(i)	dxinv[i-1]
#define	q(i,j,k,l)	q[l][i][j][k]
#define	ux(i,j,k)	ux[i][j][k]
#define	vx(i,j,k)	vx[i][j][k]
#define	wx(i,j,k)	wx[i][j][k]
#define	uy(i,j,k)	uy[i][j][k]
#define	vy(i,j,k)	vy[i][j][k]
#define	wy(i,j,k)	wy[i][j][k]
#define	uz(i,j,k)	uz[i][j][k]
#define	vz(i,j,k)	vz[i][j][k]
#define	wz(i,j,k)	wz[i][j][k]

#define difflux(i,j,k,l)	difflux[l][i][j][k]

static const double ALP	=  0.8E0;
static const double BET	= -0.2E0;
static const double GAM	=  4.0E0/105.0E0;
static const double DEL	= -1.0E0/280.0E0;

void diffterm (
	int lo[],			// i: lo[3]
	int hi[],			// i: hi[3]
	int ng,				// i
	double dx[],		// i: dx[3]
	double ****q,		// i: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
	double ****difflux,	// i/o: difflux[hi[0]-lo[0]][hi[1]-lo[1]][hi[2]-lo[2]][5]
	double eta,			// i
	double alam		// i
){
	double ***ux, ***uy, ***uz;
	double ***vx, ***vy, ***vz;
	double ***wx, ***wy, ***wz;
//	double ***txx_, ***tyy_, ***tzz_;

	double dxinv[3];
	double tauxx, tauyy, tauzz, tauxy, tauxz, tauyz;
	double divu, uxx, uyy, uzz, vxx, vyy, vzz, wxx, wyy, wzz, txx, tyy, tzz;
	double mechwork, uxy, uxz, vyz, wzx, wzy, vyx;

	int i,j,k;
	int dim[3];

	const double OneThird	= 1.0E0/3.0E0;
	const double TwoThirds	= 2.0E0/3.0E0;
	const double FourThirds	= 4.0E0/3.0E0;

	const double CENTER		= -205.0E0/72.0E0;
	const double OFF1		=  8.0E0/5.0E0;
	const double OFF2 		= -0.2E0;
	const double OFF3		=  8.0E0/315.0E0;
	const double OFF4		= -1.0E0/560.0E0;

	FOR(i, 0, 3){
		dim[i] = hi[i]-lo[i]+1 + 2*ng;
		dxinv[i] = 1.0E0/dx[i];
	}

	allocate_3D(ux, dim);	allocate_3D(uy, dim);	allocate_3D(uz, dim);
	allocate_3D(vx, dim);	allocate_3D(vy, dim);	allocate_3D(vz, dim);
	allocate_3D(wx, dim);	allocate_3D(wy, dim);	allocate_3D(wz, dim);

//	int dim_txx[3];
//	FOR(i, 0, 3)
//		dim_txx[i] = hi[i]-lo[i]+1;
//	allocate_3D(txx_, dim_txx); allocate_3D(tyy_, dim_txx); allocate_3D(tzz_, dim_txx);

//	set_3D(0.0, txx_, dim_txx);
//	set_3D(0.0, tyy_, dim_txx);
//	set_3D(0.0, tzz_, dim_txx);

	DO(i, lo[0]-ng, hi[0]-ng){
		DO(j, lo[1]-ng, hi[1]-ng){
			DO(k, lo[2]-ng, hi[2]-ng)
				difflux[irho][i][j][k] = 0.0E0;
		}
	}

//	#pragma omp parallel private(i,j,k)
	{
//		#pragma omp for nowait
		DO(i, lo[0], hi[0]){
			DO(j, lo[1]-ng, hi[1]+ng){
				DO(k, lo[2]-ng, hi[2]+ng){

					ux(i,j,k)=
						   (ALP*(q(i+1,j,k,qu)-q(i-1,j,k,qu))
						  + BET*(q(i+2,j,k,qu)-q(i-2,j,k,qu))
						  + GAM*(q(i+3,j,k,qu)-q(i-3,j,k,qu))
						  + DEL*(q(i+4,j,k,qu)-q(i-4,j,k,qu)))*dxinv(1);

					vx(i,j,k)=
						   (ALP*(q(i+1,j,k,qv)-q(i-1,j,k,qv))
						  + BET*(q(i+2,j,k,qv)-q(i-2,j,k,qv))
						  + GAM*(q(i+3,j,k,qv)-q(i-3,j,k,qv))
						  + DEL*(q(i+4,j,k,qv)-q(i-4,j,k,qv)))*dxinv(1);

					wx(i,j,k)=
						   (ALP*(q(i+1,j,k,qw)-q(i-1,j,k,qw))
						  + BET*(q(i+2,j,k,qw)-q(i-2,j,k,qw))
						  + GAM*(q(i+3,j,k,qw)-q(i-3,j,k,qw))
						  + DEL*(q(i+4,j,k,qw)-q(i-4,j,k,qw)))*dxinv(1);

				}
			}
		}

		#pragma omp for nowait
		DO(i, lo[0]-ng, hi[0]+ng){
			DO(j, lo[1], hi[1]){
				DO(k, lo[2]-ng, hi[2]+ng){

					uy(i,j,k)=
						   (ALP*(q(i,j+1,k,qu)-q(i,j-1,k,qu))
						  + BET*(q(i,j+2,k,qu)-q(i,j-2,k,qu))
						  + GAM*(q(i,j+3,k,qu)-q(i,j-3,k,qu))
						  + DEL*(q(i,j+4,k,qu)-q(i,j-4,k,qu)))*dxinv(2);

					vy(i,j,k)=
						   (ALP*(q(i,j+1,k,qv)-q(i,j-1,k,qv))
						  + BET*(q(i,j+2,k,qv)-q(i,j-2,k,qv))
						  + GAM*(q(i,j+3,k,qv)-q(i,j-3,k,qv))
						  + DEL*(q(i,j+4,k,qv)-q(i,j-4,k,qv)))*dxinv(2);

					wy(i,j,k)=
						   (ALP*(q(i,j+1,k,qw)-q(i,j-1,k,qw))
						  + BET*(q(i,j+2,k,qw)-q(i,j-2,k,qw))
						  + GAM*(q(i,j+3,k,qw)-q(i,j-3,k,qw))
						  + DEL*(q(i,j+4,k,qw)-q(i,j-4,k,qw)))*dxinv(2);

				}
			}
		}

//		#pragma omp for
		DO(i, lo[0]-ng, hi[0]+ng){
			DO(j, lo[1]-ng, hi[1]+ng){
				DO(k, lo[2], hi[2]){

					uz(i,j,k)=
						   (ALP*(q(i,j,k+1,qu)-q(i,j,k-1,qu))
						  + BET*(q(i,j,k+2,qu)-q(i,j,k-2,qu))
						  + GAM*(q(i,j,k+3,qu)-q(i,j,k-3,qu))
						  + DEL*(q(i,j,k+4,qu)-q(i,j,k-4,qu)))*dxinv(3);

					vz(i,j,k)=
						   (ALP*(q(i,j,k+1,qv)-q(i,j,k-1,qv))
						  + BET*(q(i,j,k+2,qv)-q(i,j,k-2,qv))
						  + GAM*(q(i,j,k+3,qv)-q(i,j,k-3,qv))
						  + DEL*(q(i,j,k+4,qv)-q(i,j,k-4,qv)))*dxinv(3);

					wz(i,j,k)=
						   (ALP*(q(i,j,k+1,qw)-q(i,j,k-1,qw))
						  + BET*(q(i,j,k+2,qw)-q(i,j,k-2,qw))
						  + GAM*(q(i,j,k+3,qw)-q(i,j,k-3,qw))
						  + DEL*(q(i,j,k+4,qw)-q(i,j,k-4,qw)))*dxinv(3);

				}
			}
		}
	}

//	#pragma omp parallel for private(i,j,k,uxx,uyy,uzz,vyx,wzx)
	DO(i, lo[0], hi[0]){
		DO(j, lo[1], hi[1]){
			DO(k, lo[2], hi[2]){

				uxx = (CENTER*q(i,j,k,qu)
					  + OFF1*(q(i+1,j,k,qu)+q(i-1,j,k,qu))
					  + OFF2*(q(i+2,j,k,qu)+q(i-2,j,k,qu))
					  + OFF3*(q(i+3,j,k,qu)+q(i-3,j,k,qu))
					  + OFF4*(q(i+4,j,k,qu)+q(i-4,j,k,qu)))*SQR(dxinv(1));

				uyy = (CENTER*q(i,j,k,qu)
					  + OFF1*(q(i,j+1,k,qu)+q(i,j-1,k,qu))
					  + OFF2*(q(i,j+2,k,qu)+q(i,j-2,k,qu))
					  + OFF3*(q(i,j+3,k,qu)+q(i,j-3,k,qu))
					  + OFF4*(q(i,j+4,k,qu)+q(i,j-4,k,qu)))*SQR(dxinv(2));

				uzz = (CENTER*q(i,j,k,qu)
					  + OFF1*(q(i,j,k+1,qu)+q(i,j,k-1,qu))
					  + OFF2*(q(i,j,k+2,qu)+q(i,j,k-2,qu))
					  + OFF3*(q(i,j,k+3,qu)+q(i,j,k-3,qu))
					  + OFF4*(q(i,j,k+4,qu)+q(i,j,k-4,qu)))*SQR(dxinv(3));

				vyx = (ALP*(vy(i+1,j,k)-vy(i-1,j,k))
					  + BET*(vy(i+2,j,k)-vy(i-2,j,k))
					  + GAM*(vy(i+3,j,k)-vy(i-3,j,k))
					  + DEL*(vy(i+4,j,k)-vy(i-4,j,k)))*dxinv(1);

				wzx = (ALP*(wz(i+1,j,k)-wz(i-1,j,k))
					  + BET*(wz(i+2,j,k)-wz(i-2,j,k))
					  + GAM*(wz(i+3,j,k)-wz(i-3,j,k))
					  + DEL*(wz(i+4,j,k)-wz(i-4,j,k)))*dxinv(1);

				difflux(i-ng,j-ng,k-ng,imx) = eta*(FourThirds*uxx + uyy + uzz + OneThird*(vyx+wzx));

			}
		}
	}

//	#pragma omp parallel for private(i,j,k,vxx,vyy,vzz,uxy,wzy)
	DO(i, lo[0], hi[0]){
		DO(j, lo[1], hi[1]){
			DO(k, lo[2], hi[2]){

				vxx = (CENTER*q(i,j,k,qv)
					  + OFF1*(q(i+1,j,k,qv)+q(i-1,j,k,qv))
					  + OFF2*(q(i+2,j,k,qv)+q(i-2,j,k,qv))
					  + OFF3*(q(i+3,j,k,qv)+q(i-3,j,k,qv))
					  + OFF4*(q(i+4,j,k,qv)+q(i-4,j,k,qv)))*SQR(dxinv(1));

				vyy = (CENTER*q(i,j,k,qv)
					  + OFF1*(q(i,j+1,k,qv)+q(i,j-1,k,qv))
					  + OFF2*(q(i,j+2,k,qv)+q(i,j-2,k,qv))
					  + OFF3*(q(i,j+3,k,qv)+q(i,j-3,k,qv))
					  + OFF4*(q(i,j+4,k,qv)+q(i,j-4,k,qv)))*SQR(dxinv(2));

				vzz = (CENTER*q(i,j,k,qv)
					  + OFF1*(q(i,j,k+1,qv)+q(i,j,k-1,qv))
					  + OFF2*(q(i,j,k+2,qv)+q(i,j,k-2,qv))
					  + OFF3*(q(i,j,k+3,qv)+q(i,j,k-3,qv))
					  + OFF4*(q(i,j,k+4,qv)+q(i,j,k-4,qv)))*SQR(dxinv(3));

				uxy = (ALP*(ux(i,j+1,k)-ux(i,j-1,k))
					  + BET*(ux(i,j+2,k)-ux(i,j-2,k))
					  + GAM*(ux(i,j+3,k)-ux(i,j-3,k))
					  + DEL*(ux(i,j+4,k)-ux(i,j-4,k)))*dxinv(2);

				wzy = (ALP*(wz(i,j+1,k)-wz(i,j-1,k))
					  + BET*(wz(i,j+2,k)-wz(i,j-2,k))
					  + GAM*(wz(i,j+3,k)-wz(i,j-3,k))
					  + DEL*(wz(i,j+4,k)-wz(i,j-4,k)))*dxinv(2);

				difflux(i-ng,j-ng,k-ng,imy) = eta*(vxx + FourThirds*vyy + vzz + OneThird*(uxy+wzy));

			}
		}
	}

//	#pragma omp parallel for private(i,j,k,wxx,wyy,wzz,uxz,vyz)
	DO(i, lo[0], hi[0]){
		DO(j, lo[1], hi[1]){
			DO(k, lo[2], hi[2]){

				wxx = (CENTER*q(i,j,k,qw)
					  + OFF1*(q(i+1,j,k,qw)+q(i-1,j,k,qw))
					  + OFF2*(q(i+2,j,k,qw)+q(i-2,j,k,qw))
					  + OFF3*(q(i+3,j,k,qw)+q(i-3,j,k,qw))
					  + OFF4*(q(i+4,j,k,qw)+q(i-4,j,k,qw)))*SQR(dxinv(1));

				wyy = (CENTER*q(i,j,k,qw)
					  + OFF1*(q(i,j+1,k,qw)+q(i,j-1,k,qw))
					  + OFF2*(q(i,j+2,k,qw)+q(i,j-2,k,qw))
					  + OFF3*(q(i,j+3,k,qw)+q(i,j-3,k,qw))
					  + OFF4*(q(i,j+4,k,qw)+q(i,j-4,k,qw)))*SQR(dxinv(2));

				wzz = (CENTER*q(i,j,k,qw)
					  + OFF1*(q(i,j,k+1,qw)+q(i,j,k-1,qw))
					  + OFF2*(q(i,j,k+2,qw)+q(i,j,k-2,qw))
					  + OFF3*(q(i,j,k+3,qw)+q(i,j,k-3,qw))
					  + OFF4*(q(i,j,k+4,qw)+q(i,j,k-4,qw)))*SQR(dxinv(3));

				uxz = (ALP*(ux(i,j,k+1)-ux(i,j,k-1))
					  + BET*(ux(i,j,k+2)-ux(i,j,k-2))
					  + GAM*(ux(i,j,k+3)-ux(i,j,k-3))
					  + DEL*(ux(i,j,k+4)-ux(i,j,k-4)))*dxinv(3);

				vyz = (ALP*(vy(i,j,k+1)-vy(i,j,k-1))
					  + BET*(vy(i,j,k+2)-vy(i,j,k-2))
					  + GAM*(vy(i,j,k+3)-vy(i,j,k-3))
					  + DEL*(vy(i,j,k+4)-vy(i,j,k-4)))*dxinv(3);

				difflux(i-ng,j-ng,k-ng,imz) = eta*(wxx + wyy + FourThirds*wzz + OneThird*(uxz+vyz));

			}
		}
	}

//	#pragma omp parallel for private(i,j,k,txx,tyy,tzz,divu,tauxx,tauyy,tauzz,tauxy,tauxz,tauyz,mechwork)
	DO(i, lo[0], hi[0]){
		DO(j, lo[1], hi[1]){
			DO(k, lo[2], hi[2]){

				txx = (CENTER*q(i,j,k,qt)
					  + OFF1*(q(i+1,j,k,qt)+q(i-1,j,k,qt))
					  + OFF2*(q(i+2,j,k,qt)+q(i-2,j,k,qt))
					  + OFF3*(q(i+3,j,k,qt)+q(i-3,j,k,qt))
					  + OFF4*(q(i+4,j,k,qt)+q(i-4,j,k,qt)))*SQR(dxinv(1));

				tyy = (CENTER*q(i,j,k,qt)
					  + OFF1*(q(i,j+1,k,qt)+q(i,j-1,k,qt))
					  + OFF2*(q(i,j+2,k,qt)+q(i,j-2,k,qt))
					  + OFF3*(q(i,j+3,k,qt)+q(i,j-3,k,qt))
					  + OFF4*(q(i,j+4,k,qt)+q(i,j-4,k,qt)))*SQR(dxinv(2));

				tzz = (CENTER*q(i,j,k,qt)
					  + OFF1*(q(i,j,k+1,qt)+q(i,j,k-1,qt))
					  + OFF2*(q(i,j,k+2,qt)+q(i,j,k-2,qt))
					  + OFF3*(q(i,j,k+3,qt)+q(i,j,k-3,qt))
					  + OFF4*(q(i,j,k+4,qt)+q(i,j,k-4,qt)))*SQR(dxinv(3));

//				txx_[i-ng][j-ng][k-ng] = txx;
//				tyy_[i-ng][j-ng][k-ng] = tyy;
//				tzz_[i-ng][j-ng][k-ng] = tzz;

				divu  = TwoThirds*(ux(i,j,k)+vy(i,j,k)+wz(i,j,k));
				tauxx = 2.E0*ux(i,j,k) - divu;
				tauyy = 2.E0*vy(i,j,k) - divu;
				tauzz = 2.E0*wz(i,j,k) - divu;
				tauxy = uy(i,j,k)+vx(i,j,k);
				tauxz = uz(i,j,k)+wx(i,j,k);
				tauyz = vz(i,j,k)+wy(i,j,k);

				mechwork = tauxx*ux(i,j,k) +
							tauyy*vy(i,j,k) +
							tauzz*wz(i,j,k) + SQR(tauxy)+SQR(tauxz)+SQR(tauyz);

				mechwork = eta*mechwork
					  + difflux(i-ng,j-ng,k-ng,imx)*q(i,j,k,qu)
					  + difflux(i-ng,j-ng,k-ng,imy)*q(i,j,k,qv)
					  + difflux(i-ng,j-ng,k-ng,imz)*q(i,j,k,qw);

				difflux(i-ng,j-ng,k-ng,iene) = alam*(txx+tyy+tzz) + mechwork;

			}
		}
	}

//	FILE *fdebug=fopen("txx_cpu", "w");
//	fprintf(fdebug, "1\n%d %d %d\n", dim_txx[0], dim_txx[1], dim_txx[2]);
//	print_3D(fdebug, txx_, dim_txx);
//	fclose(fdebug);

//	free_3D(txx_, dim_txx);
//	free_3D(tyy_, dim_txx);
//	free_3D(tzz_, dim_txx);

	free_3D(ux, dim);	free_3D(uy, dim);	free_3D(uz, dim);
	free_3D(vx, dim);	free_3D(vy, dim);	free_3D(vz, dim);
	free_3D(wx, dim);	free_3D(wy, dim);	free_3D(wz, dim);

}

void diffterm_debug (
	int lo[],			// i: lo[3]
	int hi[],			// i: hi[3]
	int ng,				// i
	double dx[],		// i: dx[3]
	double ****q,		// i: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
	double ****difflux,	// i/o: difflux[hi[0]-lo[0]][hi[1]-lo[1]][hi[2]-lo[2]][5]
	double eta,			// i
	double alam,		// i
	double ***ux, double ***vx, double ***wx,
	double ***uy, double ***vy, double ***wy,
	double ***uz, double ***vz, double ***wz,
	double ***vyx_, double ***wzx_,
	double ***uxy_, double ***wzy_,
	double ***uxz_, double ***vyz_,
	double ***txx_, double ***tyy_, double ***tzz_
){
//	double ***ux, ***uy, ***uz;
//	double ***vx, ***vy, ***vz;
//	double ***wx, ***wy, ***wz;

	double dxinv[3];
	double tauxx, tauyy, tauzz, tauxy, tauxz, tauyz;
	double divu, uxx, uyy, uzz, vxx, vyy, vzz, wxx, wyy, wzz, txx, tyy, tzz;
	double mechwork, uxy, uxz, vyz, wzx, wzy, vyx;

	int i,j,k;
	int dim[3];

	const double OneThird	= 1.0E0/3.0E0;
	const double TwoThirds	= 2.0E0/3.0E0;
	const double FourThirds	= 4.0E0/3.0E0;

	const double CENTER		= -205.0E0/72.0E0;
	const double OFF1		=  8.0E0/5.0E0;
	const double OFF2 		= -0.2E0;
	const double OFF3		=  8.0E0/315.0E0;
	const double OFF4		= -1.0E0/560.0E0;

	FOR(i, 0, 3){
		dim[i] = hi[i]-lo[i]+1 + 2*ng;
		dxinv[i] = 1.0E0/dx[i];
	}

	allocate_3D(ux, dim);	allocate_3D(uy, dim);	allocate_3D(uz, dim);
//	allocate_3D(vx, dim);	allocate_3D(vy, dim);	allocate_3D(vz, dim);
//	allocate_3D(wx, dim);	allocate_3D(wy, dim);	allocate_3D(wz, dim);

	DO(i, lo[0]-ng, hi[0]-ng){
		DO(j, lo[1]-ng, hi[1]-ng){
			DO(k, lo[2]-ng, hi[2]-ng)
				difflux[irho][i][j][k] = 0.0E0;
		}
	}

//	#pragma omp parallel private(i,j,k)
	{
//		#pragma omp for nowait
		DO(i, lo[0], hi[0]){
			DO(j, lo[1]-ng, hi[1]+ng){
				DO(k, lo[2]-ng, hi[2]+ng){

					ux(i,j,k)=
						   (ALP*(q(i+1,j,k,qu)-q(i-1,j,k,qu))
						  + BET*(q(i+2,j,k,qu)-q(i-2,j,k,qu))
						  + GAM*(q(i+3,j,k,qu)-q(i-3,j,k,qu))
						  + DEL*(q(i+4,j,k,qu)-q(i-4,j,k,qu)))*dxinv(1);

					vx(i,j,k)=
						   (ALP*(q(i+1,j,k,qv)-q(i-1,j,k,qv))
						  + BET*(q(i+2,j,k,qv)-q(i-2,j,k,qv))
						  + GAM*(q(i+3,j,k,qv)-q(i-3,j,k,qv))
						  + DEL*(q(i+4,j,k,qv)-q(i-4,j,k,qv)))*dxinv(1);

					wx(i,j,k)=
						   (ALP*(q(i+1,j,k,qw)-q(i-1,j,k,qw))
						  + BET*(q(i+2,j,k,qw)-q(i-2,j,k,qw))
						  + GAM*(q(i+3,j,k,qw)-q(i-3,j,k,qw))
						  + DEL*(q(i+4,j,k,qw)-q(i-4,j,k,qw)))*dxinv(1);

				}
			}
		}

		#pragma omp for nowait
		DO(i, lo[0]-ng, hi[0]+ng){
			DO(j, lo[1], hi[1]){
				DO(k, lo[2]-ng, hi[2]+ng){

					uy(i,j,k)=
						   (ALP*(q(i,j+1,k,qu)-q(i,j-1,k,qu))
						  + BET*(q(i,j+2,k,qu)-q(i,j-2,k,qu))
						  + GAM*(q(i,j+3,k,qu)-q(i,j-3,k,qu))
						  + DEL*(q(i,j+4,k,qu)-q(i,j-4,k,qu)))*dxinv(2);

					vy(i,j,k)=
						   (ALP*(q(i,j+1,k,qv)-q(i,j-1,k,qv))
						  + BET*(q(i,j+2,k,qv)-q(i,j-2,k,qv))
						  + GAM*(q(i,j+3,k,qv)-q(i,j-3,k,qv))
						  + DEL*(q(i,j+4,k,qv)-q(i,j-4,k,qv)))*dxinv(2);

					wy(i,j,k)=
						   (ALP*(q(i,j+1,k,qw)-q(i,j-1,k,qw))
						  + BET*(q(i,j+2,k,qw)-q(i,j-2,k,qw))
						  + GAM*(q(i,j+3,k,qw)-q(i,j-3,k,qw))
						  + DEL*(q(i,j+4,k,qw)-q(i,j-4,k,qw)))*dxinv(2);

				}
			}
		}

//		#pragma omp for
		DO(i, lo[0]-ng, hi[0]+ng){
			DO(j, lo[1]-ng, hi[1]+ng){
				DO(k, lo[2], hi[2]){

					uz(i,j,k)=
						   (ALP*(q(i,j,k+1,qu)-q(i,j,k-1,qu))
						  + BET*(q(i,j,k+2,qu)-q(i,j,k-2,qu))
						  + GAM*(q(i,j,k+3,qu)-q(i,j,k-3,qu))
						  + DEL*(q(i,j,k+4,qu)-q(i,j,k-4,qu)))*dxinv(3);

					vz(i,j,k)=
						   (ALP*(q(i,j,k+1,qv)-q(i,j,k-1,qv))
						  + BET*(q(i,j,k+2,qv)-q(i,j,k-2,qv))
						  + GAM*(q(i,j,k+3,qv)-q(i,j,k-3,qv))
						  + DEL*(q(i,j,k+4,qv)-q(i,j,k-4,qv)))*dxinv(3);

					wz(i,j,k)=
						   (ALP*(q(i,j,k+1,qw)-q(i,j,k-1,qw))
						  + BET*(q(i,j,k+2,qw)-q(i,j,k-2,qw))
						  + GAM*(q(i,j,k+3,qw)-q(i,j,k-3,qw))
						  + DEL*(q(i,j,k+4,qw)-q(i,j,k-4,qw)))*dxinv(3);

				}
			}
		}
	}

//	#pragma omp parallel for private(i,j,k,uxx,uyy,uzz,vyx,wzx)
	DO(i, lo[0], hi[0]){
		DO(j, lo[1], hi[1]){
			DO(k, lo[2], hi[2]){

				uxx = (CENTER*q(i,j,k,qu)
					  + OFF1*(q(i+1,j,k,qu)+q(i-1,j,k,qu))
					  + OFF2*(q(i+2,j,k,qu)+q(i-2,j,k,qu))
					  + OFF3*(q(i+3,j,k,qu)+q(i-3,j,k,qu))
					  + OFF4*(q(i+4,j,k,qu)+q(i-4,j,k,qu)))*SQR(dxinv(1));

				uyy = (CENTER*q(i,j,k,qu)
					  + OFF1*(q(i,j+1,k,qu)+q(i,j-1,k,qu))
					  + OFF2*(q(i,j+2,k,qu)+q(i,j-2,k,qu))
					  + OFF3*(q(i,j+3,k,qu)+q(i,j-3,k,qu))
					  + OFF4*(q(i,j+4,k,qu)+q(i,j-4,k,qu)))*SQR(dxinv(2));

				uzz = (CENTER*q(i,j,k,qu)
					  + OFF1*(q(i,j,k+1,qu)+q(i,j,k-1,qu))
					  + OFF2*(q(i,j,k+2,qu)+q(i,j,k-2,qu))
					  + OFF3*(q(i,j,k+3,qu)+q(i,j,k-3,qu))
					  + OFF4*(q(i,j,k+4,qu)+q(i,j,k-4,qu)))*SQR(dxinv(3));

				vyx = (ALP*(vy(i+1,j,k)-vy(i-1,j,k))
					  + BET*(vy(i+2,j,k)-vy(i-2,j,k))
					  + GAM*(vy(i+3,j,k)-vy(i-3,j,k))
					  + DEL*(vy(i+4,j,k)-vy(i-4,j,k)))*dxinv(1);

				wzx = (ALP*(wz(i+1,j,k)-wz(i-1,j,k))
					  + BET*(wz(i+2,j,k)-wz(i-2,j,k))
					  + GAM*(wz(i+3,j,k)-wz(i-3,j,k))
					  + DEL*(wz(i+4,j,k)-wz(i-4,j,k)))*dxinv(1);

				difflux(i-ng,j-ng,k-ng,imx) = eta*(FourThirds*uxx + uyy + uzz + OneThird*(vyx+wzx));

				vyx_[i-ng][j-ng][k-ng] = vyx;
				wzx_[i-ng][j-ng][k-ng] = wzx;

			}
		}
	}

//	#pragma omp parallel for private(i,j,k,vxx,vyy,vzz,uxy,wzy)
	DO(i, lo[0], hi[0]){
		DO(j, lo[1], hi[1]){
			DO(k, lo[2], hi[2]){

				vxx = (CENTER*q(i,j,k,qv)
					  + OFF1*(q(i+1,j,k,qv)+q(i-1,j,k,qv))
					  + OFF2*(q(i+2,j,k,qv)+q(i-2,j,k,qv))
					  + OFF3*(q(i+3,j,k,qv)+q(i-3,j,k,qv))
					  + OFF4*(q(i+4,j,k,qv)+q(i-4,j,k,qv)))*SQR(dxinv(1));

				vyy = (CENTER*q(i,j,k,qv)
					  + OFF1*(q(i,j+1,k,qv)+q(i,j-1,k,qv))
					  + OFF2*(q(i,j+2,k,qv)+q(i,j-2,k,qv))
					  + OFF3*(q(i,j+3,k,qv)+q(i,j-3,k,qv))
					  + OFF4*(q(i,j+4,k,qv)+q(i,j-4,k,qv)))*SQR(dxinv(2));

				vzz = (CENTER*q(i,j,k,qv)
					  + OFF1*(q(i,j,k+1,qv)+q(i,j,k-1,qv))
					  + OFF2*(q(i,j,k+2,qv)+q(i,j,k-2,qv))
					  + OFF3*(q(i,j,k+3,qv)+q(i,j,k-3,qv))
					  + OFF4*(q(i,j,k+4,qv)+q(i,j,k-4,qv)))*SQR(dxinv(3));

				uxy = (ALP*(ux(i,j+1,k)-ux(i,j-1,k))
					  + BET*(ux(i,j+2,k)-ux(i,j-2,k))
					  + GAM*(ux(i,j+3,k)-ux(i,j-3,k))
					  + DEL*(ux(i,j+4,k)-ux(i,j-4,k)))*dxinv(2);

				wzy = (ALP*(wz(i,j+1,k)-wz(i,j-1,k))
					  + BET*(wz(i,j+2,k)-wz(i,j-2,k))
					  + GAM*(wz(i,j+3,k)-wz(i,j-3,k))
					  + DEL*(wz(i,j+4,k)-wz(i,j-4,k)))*dxinv(2);

				uxy_[i-ng][j-ng][k-ng] = uxy;
				wzy_[i-ng][j-ng][k-ng] = wzy;
				difflux(i-ng,j-ng,k-ng,imy) = eta*(vxx + FourThirds*vyy + vzz + OneThird*(uxy+wzy));

			}
		}
	}

//	#pragma omp parallel for private(i,j,k,wxx,wyy,wzz,uxz,vyz)
	DO(i, lo[0], hi[0]){
		DO(j, lo[1], hi[1]){
			DO(k, lo[2], hi[2]){

				wxx = (CENTER*q(i,j,k,qw)
					  + OFF1*(q(i+1,j,k,qw)+q(i-1,j,k,qw))
					  + OFF2*(q(i+2,j,k,qw)+q(i-2,j,k,qw))
					  + OFF3*(q(i+3,j,k,qw)+q(i-3,j,k,qw))
					  + OFF4*(q(i+4,j,k,qw)+q(i-4,j,k,qw)))*SQR(dxinv(1));

				wyy = (CENTER*q(i,j,k,qw)
					  + OFF1*(q(i,j+1,k,qw)+q(i,j-1,k,qw))
					  + OFF2*(q(i,j+2,k,qw)+q(i,j-2,k,qw))
					  + OFF3*(q(i,j+3,k,qw)+q(i,j-3,k,qw))
					  + OFF4*(q(i,j+4,k,qw)+q(i,j-4,k,qw)))*SQR(dxinv(2));

				wzz = (CENTER*q(i,j,k,qw)
					  + OFF1*(q(i,j,k+1,qw)+q(i,j,k-1,qw))
					  + OFF2*(q(i,j,k+2,qw)+q(i,j,k-2,qw))
					  + OFF3*(q(i,j,k+3,qw)+q(i,j,k-3,qw))
					  + OFF4*(q(i,j,k+4,qw)+q(i,j,k-4,qw)))*SQR(dxinv(3));

				uxz = (ALP*(ux(i,j,k+1)-ux(i,j,k-1))
					  + BET*(ux(i,j,k+2)-ux(i,j,k-2))
					  + GAM*(ux(i,j,k+3)-ux(i,j,k-3))
					  + DEL*(ux(i,j,k+4)-ux(i,j,k-4)))*dxinv(3);

				vyz = (ALP*(vy(i,j,k+1)-vy(i,j,k-1))
					  + BET*(vy(i,j,k+2)-vy(i,j,k-2))
					  + GAM*(vy(i,j,k+3)-vy(i,j,k-3))
					  + DEL*(vy(i,j,k+4)-vy(i,j,k-4)))*dxinv(3);

				uxz_[i-ng][j-ng][k-ng] = uxz;
				vyz_[i-ng][j-ng][k-ng] = vyz;
				difflux(i-ng,j-ng,k-ng,imz) = eta*(wxx + wyy + FourThirds*wzz + OneThird*(uxz+vyz));

			}
		}
	}

//	#pragma omp parallel for private(i,j,k,txx,tyy,tzz,divu,tauxx,tauyy,tauzz,tauxy,tauxz,tauyz,mechwork)
	DO(i, lo[0], hi[0]){
		DO(j, lo[1], hi[1]){
			DO(k, lo[2], hi[2]){

				txx = (CENTER*q(i,j,k,qt)
					  + OFF1*(q(i+1,j,k,qt)+q(i-1,j,k,qt))
					  + OFF2*(q(i+2,j,k,qt)+q(i-2,j,k,qt))
					  + OFF3*(q(i+3,j,k,qt)+q(i-3,j,k,qt))
					  + OFF4*(q(i+4,j,k,qt)+q(i-4,j,k,qt)))*SQR(dxinv(1));

				tyy = (CENTER*q(i,j,k,qt)
					  + OFF1*(q(i,j+1,k,qt)+q(i,j-1,k,qt))
					  + OFF2*(q(i,j+2,k,qt)+q(i,j-2,k,qt))
					  + OFF3*(q(i,j+3,k,qt)+q(i,j-3,k,qt))
					  + OFF4*(q(i,j+4,k,qt)+q(i,j-4,k,qt)))*SQR(dxinv(2));

				tzz = (CENTER*q(i,j,k,qt)
					  + OFF1*(q(i,j,k+1,qt)+q(i,j,k-1,qt))
					  + OFF2*(q(i,j,k+2,qt)+q(i,j,k-2,qt))
					  + OFF3*(q(i,j,k+3,qt)+q(i,j,k-3,qt))
					  + OFF4*(q(i,j,k+4,qt)+q(i,j,k-4,qt)))*SQR(dxinv(3));

				txx_[i-ng][j-ng][k-ng] = txx;
				tyy_[i-ng][j-ng][k-ng] = tyy;
				tzz_[i-ng][j-ng][k-ng] = tzz;

				divu  = TwoThirds*(ux(i,j,k)+vy(i,j,k)+wz(i,j,k));
				tauxx = 2.E0*ux(i,j,k) - divu;
				tauyy = 2.E0*vy(i,j,k) - divu;
				tauzz = 2.E0*wz(i,j,k) - divu;
				tauxy = uy(i,j,k)+vx(i,j,k);
				tauxz = uz(i,j,k)+wx(i,j,k);
				tauyz = vz(i,j,k)+wy(i,j,k);

				mechwork = tauxx*ux(i,j,k) +
							tauyy*vy(i,j,k) +
							tauzz*wz(i,j,k) + SQR(tauxy)+SQR(tauxz)+SQR(tauyz);

				mechwork = eta*mechwork
					  + difflux(i-ng,j-ng,k-ng,imx)*q(i,j,k,qu)
					  + difflux(i-ng,j-ng,k-ng,imy)*q(i,j,k,qv)
					  + difflux(i-ng,j-ng,k-ng,imz)*q(i,j,k,qw);

				difflux(i-ng,j-ng,k-ng,iene) = alam*(txx+tyy+tzz) + mechwork;

//				if(i==ng && j==ng+12 && k==ng){
//					printf("difflux[%d][%d][%d] = %le\n", i-ng,j-ng,k-ng, difflux(i-ng,j-ng,k-ng,iene));
//					printf("mechwork: %le\n", mechwork);
//					printf("ux, vy, wz = %le %le %le\n", ux(i,j,k), vy(i,j,k), wz(i,j,k));
//				}

			}
		}
	}

//	free_3D(ux, dim);	free_3D(uy, dim);	free_3D(uz, dim);
//	free_3D(vx, dim);	free_3D(vy, dim);	free_3D(vz, dim);
//	free_3D(wx, dim);	free_3D(wy, dim);	free_3D(wz, dim);

}
#undef	lo
#undef	hi
#undef 	dxinv
#undef	q
#undef	ux
#undef	vx
#undef	wx
#undef	uy
#undef	vy
#undef	wy
#undef	uz
#undef	vz
#undef	wz

#undef difflux
