/*
	Allen-Cahn Equation - MPI
	
	Written by Joel Eliason

	A program that simulates the Allen-Cahn equation, which models phase separation in alloys

	Inputs:
		int N: the dimension of the square grid
		double v1: first component of the velocity vector
		double v2: second component of the velocity vector
		double b: parameter in Allen-Cahn equation
		double W: parameter in Allen-Cahn equation
		int N_t: number of time steps to take (from T=0 until T=5)
		long int seed: seed for random number generator (optional)

	Outputs:
		Returns "allen.out", a file that contains data at time points t=0.5k for k=0,...,10
		Returns performance measure

	Editing History:
		5/8/2018: Initial draft
	
*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <complex.h>
#include "mpi.h"
#include "fftw3-mpi.h"

int main(int argc, char* argv[])
{	
	//MPI and FFTW initialization
	MPI_Init(&argc,&argv);

	fftw_mpi_init();
	
	int rank;
	FILE *fileid;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	
	//Start the clock
	double precision = MPI_Wtick();
	double starttime = MPI_Wtime();
	
	//parameter initialization
	ptrdiff_t N;
	double v1,v2,b,W;
	int N_t;
	long int seed;
	if(rank==0){
		N = atoi(argv[1]);
		v1 = atof(argv[2]);
		v2 = atof(argv[3]);
		b = atof(argv[4]);
		W = atof(argv[5]);
		N_t = atoi(argv[6]);
		//printf("Nt = %d",N_t);
		seed = 123;
		if(argc==8){
			seed = atoi(argv[7]);
		}
	}
	//Broadcast all parameters to other ranks
	MPI_Bcast(&N,1,MPI_AINT,0,MPI_COMM_WORLD);
	MPI_Bcast(&v1,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&v2,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&b,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&W,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&N_t,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&seed,1,MPI_LONG,0,MPI_COMM_WORLD);
	//printf("Nt=%d",N_t);

	//additional parameter initialization
	double Wsq = W*W;
	double Tf = 5.;
	double delta_t = Tf/(double)N_t;
	double eq_tol=delta_t/100.;
#ifndef M_PI
	const double M_PI = 4.0*atan(1.0);
#endif
	double dx = 2*M_PI/N;
	
	//parameters that flag when to write output to file
	double skip = 0.5;
	int count = 1;

	//initialize local dimensions
	ptrdiff_t localN,local0;
	//ptrdiff_t alloc_local = fftw_mpi_local_size_2d(N,N/2+1,MPI_COMM_WORLD,&localN,&local0);
	ptrdiff_t alloc_local = fftw_mpi_local_size_2d(N,N,MPI_COMM_WORLD,&localN,&local0);

	//Allocate local memory
	//double* datain = fftw_alloc_real(2*alloc_local);
	fftw_complex* datain = fftw_alloc_complex(alloc_local);
	fftw_complex* dataout = fftw_alloc_complex(alloc_local);
	fftw_complex* first_derivx = fftw_alloc_complex(alloc_local);
	fftw_complex* first_derivy = fftw_alloc_complex(alloc_local);
	fftw_complex* second_deriv = fftw_alloc_complex(alloc_local);
	fftw_complex* F1 = fftw_alloc_complex(alloc_local);
	fftw_complex* F2 = fftw_alloc_complex(alloc_local);

	//Set up transform plans
	fftw_plan pf_main, pf1, pf2,pb_deriv1x,pb_deriv1y,pb_deriv2;

	pf_main = fftw_mpi_plan_dft_2d(N,N,datain,dataout,MPI_COMM_WORLD,FFTW_FORWARD,FFTW_ESTIMATE);
	pf1 = fftw_mpi_plan_dft_2d(N,N,F1,dataout,MPI_COMM_WORLD,FFTW_FORWARD,FFTW_ESTIMATE);
	pf2 = fftw_mpi_plan_dft_2d(N,N,F2,dataout,MPI_COMM_WORLD,FFTW_FORWARD,FFTW_ESTIMATE);
	pb_deriv1x = fftw_mpi_plan_dft_2d(N,N,first_derivx,first_derivx,MPI_COMM_WORLD,FFTW_BACKWARD,FFTW_ESTIMATE);
	pb_deriv1y = fftw_mpi_plan_dft_2d(N,N,first_derivy,first_derivy,MPI_COMM_WORLD,FFTW_BACKWARD,FFTW_ESTIMATE);
	pb_deriv2 = fftw_mpi_plan_dft_2d(N,N,second_deriv,second_deriv,MPI_COMM_WORLD,FFTW_BACKWARD,FFTW_ESTIMATE);

	
	//Initialize gathering matrix, rank, file to write, and different random seed for each processor
	fftw_complex* fulldata;
	//printf("Rank = %d,localN = %td\n",rank,localN);
	seed=seed+rank;
	srand48(seed);
	if(rank==0){
		fulldata = (fftw_complex*)fftw_alloc_complex(N*N);
		fileid = fopen("allen.out","w");
	}
	//Initialize data on local allocations
	for (int i=0;i<localN;i++){
		for(int j=0;j<N;j++){
			datain[i*N+j]=2*drand48()-1+0.0*I;
		}
	}
	//printf("Rank %d before gather\n",rank);

	//Gather initial data to rank=0 and write it to file
	MPI_Gather(datain,localN*N,MPI_DOUBLE_COMPLEX,fulldata,localN*N,MPI_DOUBLE_COMPLEX,0,MPI_COMM_WORLD);
	if(rank==0){
                        for(int j=0;j<N;j++){
                                for(int k=0;k<N;k++){
                                        double real = creal(fulldata[j*N+k]);
                                        fwrite(&real,sizeof(double),1,fileid);
                                }
                        }
                }
	//printf("Rank %d after gather\n",rank);

	//index for indexing 2d arrays as 1d arrays
	int idx;

	//BEGIN RK4 LOOP
	for(int i=1;i<=N_t;i++){
		/*
		if(rank==0){
			printf("%d\n",i);
		}
		*/
		//printf("In loop, rank = %d, i = %d\n",rank,i);

		//STEP 1
		//Forward transform
		fftw_execute(pf_main);
		
		// Differentiation, composition of RHS
		for(int j=0;j<localN;j++){
			for(int k=0;k<N;k++){
				idx=j*N+k;
				int sx, sy;
				sx = (j+local0)<=N/2 ? j+local0 : j+local0-N;
				sy = k<=N/2 ? k : k-N;
				dataout[idx]=dataout[idx]/N/N;
				first_derivx[idx]= sx==N/2 ? 0. : I*sx*dataout[idx];
				first_derivy[idx]= sy==N/2 ? 0. : I*sy*dataout[idx];
				second_deriv[idx]=-(sx*sx+sy*sy)*dataout[idx];
			}
		}
		//Backwards transforms of derivatives
		fftw_execute(pb_deriv1x);
		fftw_execute(pb_deriv1y);
		fftw_execute(pb_deriv2);
		
		//Update for RK4 step 1
		for(int j=0;j<localN;j++){
			for(int k=0;k<N;k++){
				idx = j*N+k;
				F1[idx] = datain[idx]+delta_t/4*(-v1*first_derivx[idx]-v2*first_derivy[idx]+b*(second_deriv[idx]+datain[idx]*(1-datain[idx]*datain[idx])/(Wsq)));
			}
		}

		//STEP 2
		//Forward transform
		fftw_execute(pf1);
		
		// Differentiation, composition of RHS
		for(int j=0;j<localN;j++){
			for(int k=0;k<N;k++){
				idx=j*N+k;
				int sx, sy;
				sx = (j+local0)<=N/2 ? j+local0 : j+local0-N;
				sy = k<=N/2 ? k : k-N;
				dataout[idx]=dataout[idx]/N/N;
				first_derivx[idx]= sx==N/2 ? 0. : I*sx*dataout[idx];
				first_derivy[idx]= sy==N/2 ? 0. : I*sy*dataout[idx];
				second_deriv[idx]=-(sx*sx+sy*sy)*dataout[idx];
			}
		}
		//Backwards transforms of derivatives
		fftw_execute(pb_deriv1x);
		fftw_execute(pb_deriv1y);
		fftw_execute(pb_deriv2);
		
		//Update for RK4 step 2
		for(int j=0;j<localN;j++){
			for(int k=0;k<N;k++){
				idx = j*N+k;
				F2[idx] = datain[idx]+delta_t/3*(-v1*first_derivx[idx]-v2*first_derivy[idx]+b*(second_deriv[idx]+F1[idx]*(1-F1[idx]*F1[idx])/(Wsq)));
			}
		}

		//STEP 3
		//Forward transform
		fftw_execute(pf2);
		
		// Differentiation, composition of RHS
		for(int j=0;j<localN;j++){
			for(int k=0;k<N;k++){
				idx=j*N+k;
				int sx, sy;
				sx = (j+local0)<=N/2 ? j+local0 : j+local0-N;
				sy = k<=N/2 ? k : k-N;
				dataout[idx]=dataout[idx]/N/N;
				first_derivx[idx]= sx==N/2 ? 0. : I*sx*dataout[idx];
				first_derivy[idx]= sy==N/2 ? 0. : I*sy*dataout[idx];
				second_deriv[idx]=-(sx*sx+sy*sy)*dataout[idx];
			}
		}
		//Backwards transforms of derivatives
		fftw_execute(pb_deriv1x);
		fftw_execute(pb_deriv1y);
		fftw_execute(pb_deriv2);
		
		//Update for RK4 step 3
		for(int j=0;j<localN;j++){
			for(int k=0;k<N;k++){
				idx = j*N+k;
				F1[idx] = datain[idx]+delta_t/2*(-v1*first_derivx[idx]-v2*first_derivy[idx]+b*(second_deriv[idx]+F2[idx]*(1-F2[idx]*F2[idx])/(Wsq)));
			}
		}

		//STEP 4
		//Forward transform
		fftw_execute(pf1);
		
		// Differentiation, composition of RHS
		for(int j=0;j<localN;j++){
			for(int k=0;k<N;k++){
				idx=j*N+k;
				int sx, sy;
				sx = (j+local0)<=N/2 ? j+local0 : j+local0-N;
				sy = k<=N/2 ? k : k-N;
				dataout[idx]=dataout[idx]/N/N;
				first_derivx[idx]= sx==N/2 ? 0. : I*sx*dataout[idx];
				first_derivy[idx]= sy==N/2 ? 0. : I*sy*dataout[idx];
				second_deriv[idx]=-(sx*sx+sy*sy)*dataout[idx];
			}
		}
		//Backwards transforms of derivatives
		fftw_execute(pb_deriv1x);
		fftw_execute(pb_deriv1y);
		fftw_execute(pb_deriv2);
		
		//Update for RK4 step 4
		for(int j=0;j<localN;j++){
			for(int k=0;k<N;k++){
				idx = j*N+k;
				datain[idx] = datain[idx]+delta_t*(-v1*first_derivx[idx]-v2*first_derivy[idx]+b*(second_deriv[idx]+F1[idx]*(1-F1[idx]*F1[idx])/(Wsq)));
			}
		}

		//Gather the updated data to rank=0
			MPI_Gather(datain,localN*N,MPI_DOUBLE_COMPLEX,fulldata,localN*N,MPI_DOUBLE_COMPLEX,0,MPI_COMM_WORLD);

		//Write the results to file
		if(fabs(i*delta_t-(float)(skip*count))<eq_tol){
		if(rank==0){
			for(int j=0;j<N;j++){
				for(int k=0;k<N;k++){
					double real = creal(fulldata[j*N+k]);
					//printf("%f+i%f\n",real,cimag(fulldata[j*N+k]));
					fwrite(&real,sizeof(double),1,fileid);
				}
			}
			count++;
		}
		}
	/*
		else{
			if(rank==0){
			for(int j=0;j<N;j++){
				for(int k=0;k<N;k++){
					double real = creal(fulldata[j*N+k]);
					printf("%f+i%f\n",real,cimag(fulldata[j*N+k]));
					fwrite(&real,sizeof(double),1,fileid);
				}
			}
		}
		}
	*/
		
	}
	//END RK4 LOOP

	//Close files, free memory and destroy plans
	if(rank==0){
		fclose(fileid);
		free(fulldata);
	}
	
	fftw_destroy_plan(pf_main); fftw_destroy_plan(pf1); fftw_destroy_plan(pf2);
	fftw_destroy_plan(pb_deriv1x); fftw_destroy_plan(pb_deriv1y); fftw_destroy_plan(pb_deriv2);
	
	fftw_free(datain); fftw_free(dataout); fftw_free(first_derivx);
	fftw_free(first_derivy); fftw_free(second_deriv);
	fftw_free(F1); fftw_free(F2);

	//Stop the clock, print to screen
	double time_elapsed = MPI_Wtime()-starttime;
	if(rank==0){
		printf("Execution time = %le seconds, with precision %le seconds", time_elapsed, precision);
	}

	//Finalize MPI
	MPI_Finalize();
}
