# Note: for openmp pgfortran (pgi 14) or later is not working (stack overflow) when system size is too large
#       but could work for smaller size with ncpu < max per node; 
#       no such problem for pgi 12.6 or 12.10
#
#TYPE = pgfortran
#TYPE = /wsu/apps/compilers/pgi/pgicdk-1210/linux86-64/12.10/bin/pgfortran
TYPE = /wsu/el7/pgi/12.10/linux86-64/12.10/bin/pgfortran

## -tp x64: Unified AMD/Intel 64-bit mode; amd64: same as k8-64; amd64e: same as k8-64e
FFLAGS= -O3 -tp x64,amd64,amd64e,sandybridge-64

NAME = bampl1D_AB
FILES = $(NAME).F03 rand-d.f90

# -Mconcur: automatic shared-memory parallel (SMP), for loops
# -mp: for OpenMP
LIB_FFTW = -mp -Mconcur -L/wsu/apps/pgi-14.6/fftw/fftw-3.3.4-openmp/lib/ -lfftw3_omp -lfftw3 -lm

pgi: 
	$(TYPE) $(FFLAGS) $(FILES) $(LIB_FFTW) -o aa.out;
	rm *.o
	rm *.mod
