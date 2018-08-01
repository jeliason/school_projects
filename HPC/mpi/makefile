
INCLUDEDIR = -I/software/FFTW/3.3.3-RH7/include
LD = -L/software/FFTW/3.3.3-RH7/lib


allen_main.o: allen_main.c
	mpicc -std=gnu99 -c $(INCLUDEDIR) allen_main.c
allen: allen_main.o
	mpicc -std=c99 -o allen $(LD) allen_main.o -lfftw3_mpi -lfftw3 -lm
mc:mc_main.c
	mpicc -std=gnu99 -o mc mc_main.c -lm

