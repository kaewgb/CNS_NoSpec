#include <stdlib.h>
#include "header.h"
#include "helper_functions.h"

#define	lo(i)		lo[i]
#define	hi(i)		hi[i]
#define dxinv(i)	dxinv[i]
#define	q(i,j,k,l)	q[i][j][k][l]
#define	ux(i,j,k)	ux[i][j][k]
#define	vx(i,j,k)	vx[i][j][k]
#define	wx(i,j,k)	wx[i][j][k]
#define	uy(i,j,k)	uy[i][j][k]
#define	vy(i,j,k)	vy[i][j][k]
#define	wy(i,j,k)	wy[i][j][k]
#define	uz(i,j,k)	uz[i][j][k]
#define	vz(i,j,k)	vz[i][j][k]
#define	wz(i,j,k)	wz[i][j][k]

#define difflux(i,j,k,l)	difflux[i][j][k][l]

void diffterm (
	int lo[],			// i: lo[3]
	int hi[],			// i: hi[3]
	int ng,				// i
	double dx[],		// i: dx[3]
	double ****q,		// i: q[hi[0]-lo[0]+2*ng][hi[1]-lo[1]+2*ng][hi[2]-lo[2]+2*ng][6]
	double ****difflux,	// i/o: difflux[hi[0]-lo[0]][hi[1]-lo[1]][hi[2]-lo[2]][5]
	double eta,			// i
	double alam			// i
){
	double ***ux, ***uy, ***uz;
	double ***vx, ***vy, ***vz;
	double ***wx, ***wy, ***wz;

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

	FOR(i, 0, 3)
		dim[i] = hi[i]-lo[i]+1 + 2*ng;

	allocate_3D(ux, dim);	allocate_3D(uy, dim);	allocate_3D(uz, dim);
	allocate_3D(vx, dim);	allocate_3D(vy, dim);	allocate_3D(vz, dim);
	allocate_3D(wx, dim);	allocate_3D(wy, dim);	allocate_3D(wz, dim);

	DO(i, lo[0], hi[0]){
		DO(j, lo[1], hi[1]){
			DO(k, lo[2], hi[2])
				difflux[i][j][k][irho] = 0.0E0;
		}
	}

	DO(i, 1, 3)
		dxinv[i] = 1.0E0/dx[i];

	#pragma omp parallel private(i,j,k)
	{
		#pragma omp for nowait
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

		#pragma omp for
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

	#pragma omp parallel for private(i,j,k,uxx,uyy,uzz,vyx,wzx)
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

				difflux(i,j,k,imx) = eta*(FourThirds*uxx + uyy + uzz + OneThird*(vyx+wzx));

			}
		}
	}

	#pragma omp parallel for private(i,j,k,vxx,vyy,vzz,uxy,wzy)
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

				difflux(i,j,k,imy) = eta*(vxx + FourThirds*vyy + vzz + OneThird*(uxy+wzy));

			}
		}
	}

	#pragma omp parallel for private(i,j,k,wxx,wyy,wzz,uxz,vyz)
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

				difflux(i,j,k,imz) = eta*(wxx + wyy + FourThirds*wzz + OneThird*(uxz+vyz));

			}
		}
	}

	#pragma omp parallel for private(i,j,k,txx,tyy,tzz,divu,tauxx,tauyy,tauzz,tauxy,tauxz,tauyz,mechwork)
	DO(i, lo[0], hi[0]){
		DO(j, lo[1], hi[1]){
			DO(k, lo[2], hi[2]){

				txx = (CENTER*q(i,j,k,6)
					  + OFF1*(q(i+1,j,k,6)+q(i-1,j,k,6))
					  + OFF2*(q(i+2,j,k,6)+q(i-2,j,k,6))
					  + OFF3*(q(i+3,j,k,6)+q(i-3,j,k,6))
					  + OFF4*(q(i+4,j,k,6)+q(i-4,j,k,6)))*SQR(dxinv(1));

				tyy = (CENTER*q(i,j,k,6)
					  + OFF1*(q(i,j+1,k,6)+q(i,j-1,k,6))
					  + OFF2*(q(i,j+2,k,6)+q(i,j-2,k,6))
					  + OFF3*(q(i,j+3,k,6)+q(i,j-3,k,6))
					  + OFF4*(q(i,j+4,k,6)+q(i,j-4,k,6)))*SQR(dxinv(2));

				tzz = (CENTER*q(i,j,k,6)
					  + OFF1*(q(i,j,k+1,6)+q(i,j,k-1,6))
					  + OFF2*(q(i,j,k+2,6)+q(i,j,k-2,6))
					  + OFF3*(q(i,j,k+3,6)+q(i,j,k-3,6))
					  + OFF4*(q(i,j,k+4,6)+q(i,j,k-4,6)))*SQR(dxinv(3));

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
					  + difflux(i,j,k,imx)*q(i,j,k,qu)
					  + difflux(i,j,k,imy)*q(i,j,k,qv)
					  + difflux(i,j,k,imz)*q(i,j,k,qw);

				difflux(i,j,k,iene) = alam*(txx+tyy+tzz) + mechwork;

			}
		}
	}

	free_3D(ux, dim);	free_3D(uy, dim);	free_3D(uz, dim);
	free_3D(vx, dim);	free_3D(vy, dim);	free_3D(vz, dim);
	free_3D(wx, dim);	free_3D(wy, dim);	free_3D(wz, dim);
}
