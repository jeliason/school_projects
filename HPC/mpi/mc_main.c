/*
  	Monte Carlo Integration - MPI

        Written by Joel Eliason

        A program that performs MC integration to evaluate an integral

        Inputs:
               	int N: number of samples
                double lambda: expectation of exponential distribution
                long int seed: seed for random number generator (optional)

        Outputs:
                Returns performance measure

        Editing History:
                5/16/2018: Initial draft

*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

int main(int argc, char* argv[])
{
        //MPI initialization
        MPI_Init(&argc,&argv);
	//printf("Finishzed MPI_INIT\n");
	//Start the clock
	double precision = MPI_Wtick();
	double starttime = MPI_Wtime();
	
	//Initialize rank and number of processors
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	int size;
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	//printf("Finished rank and size\n");

        //parameter initialization
	int N;
	double lambda;
	long int seed;
	if(rank==0){
        	N = atoi(argv[1]);
        	lambda = atof(argv[2]);
		seed = 123;
		if(argc==4){
			seed = atoi(argv[3]);
		}
	}
	//printf("Finished initialization\n");
	//Broadcast parameters
	MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&lambda,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&seed,1,MPI_LONG,0,MPI_COMM_WORLD);
	//printf("Finished broadcasting\n");
	//Initialize random seed
	seed=seed+rank;
	srand48(seed);
	
	//localN is number of samples per processor
	int localN = N/size + (N%size > rank ? 1 : 0);

	//x is the random variable sampled from exponential distribution
	//cosx is f(x)
	//sumx is local sum of function values
	//sum_all is total sum of function values across all processors
	//double x; double cosx;
	long double sumx=0;
	long double sum_all=0;

	//START MC SAMPLING
	for(int i=0;i<localN;i++){
		//x=drand48();
		sumx=sumx+cos(-log(drand48())/lambda)/lambda;
		//cosx=cos(x)/lambda;
		//sumx=sumx+cosx;

	}	
	//printf("Finished sampling\n");
	//Sum all function values on single processor
	MPI_Reduce(&sumx,&sum_all,1,MPI_LONG_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	//printf("Finished reduction");
	long double error;
	if(rank==0){
		sum_all=sum_all/N;
		//printf("Expectation = %1.5Lf\n",sum_all);
		error=fabs(lambda/(1+lambda*lambda)-sum_all);
		//printf("Error = %1.10Lf\n",error);
	}
	//Stop the clock, print to screen
	double time_elapsed = MPI_Wtime()-starttime;

	if(rank==0){
		printf("Execution time = %le seconds, with precision %le seconds\n", time_elapsed, precision);
	}

	MPI_Finalize();
	return 0;
}
