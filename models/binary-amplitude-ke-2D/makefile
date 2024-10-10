TYPE = mpicc
# for compiling in zfh nodes and submitting jobs there (to be more efficient):
#     after login a node, first need to load module: module load pgi-14.6/compiler pgi-14.6/compiler-mpi 
#
FFLAGS= -Wno-unused-variable -O3 -tp sandybridge-64 # mpicc doesn't permit targeting multiple processors

NAME = amppfc2d_surf_st
# FILES = $(NAME).c
FILES = CN_LS2.c

LIB_FFTW = -L/opt/ohpc/pub/libs/gnu7/openmpi/fftw/3.3.7/lib -lfftw3_mpi -lfftw3 -lm
LIB_INC = -I/opt/ohpc/pub/libs/gnu7/openmpi/fftw/3.3.7/include

mpi: 
	$(TYPE) $(FFLAGS) $(FILES) $(LIB_INC) $(LIB_FFTW) -o a.out;
