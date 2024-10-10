// mpicc hbn_Vo.c -lm -eftw3_mpi -eftw3 -o hbn_Vo
#include <stdlib.h>
#include <time.h>
#include <fenv.h>
#include <complex.h>
#include <fftw3-mpi.h>
#include <math.h>
//    #include <stdio.h>
//    #include <stddef.h>

ptrdiff_t Ly, Lx;

int main(int argc, char **argv)
{
    printf("Hello world\n");
    MPI_Status status;
    double fx, fy, kA1f, kA2f, kA3f, kB1f, kB2f, kB3f, dx, dy, dt, muA, muB;
    double eng, enl, Sf, twopi, Ssq, Tsq, kx, ky, fn, ffac, ao, gamma, ksq;
    double A1sq, A2sq, A3sq, B1sq, B2sq, B3sq, vA, vB, epsA, epsB, gA, gB;
    double alphaAB, betaB, omega, mu, NA, NB, arg, nl, ns, theta;
    double vA3, vB3, gnAo, gnBo, aonm, constants, Asqcoeff, Bsqcoeff;
    double Vo, alpha, Ap, Bp, ABs, naf, nbf, nAa, nBa;
    double dGxA1, dGyA1, dGxA2, dGyA2, dGxA3, dGyA3;
    double dGxB1, dGyB1, dGxB2, dGyB2, dGxB3, dGyB3;
    double G1xo, G1yo, G2xo, G2yo, G3xo, G3yo;
    double G1x, G1y, G2x, G2y, G3x, G3y;
    double aa, aai, bb, bbi, cc, cci;
    // double *nA,*nB,kAl,kAn,kBl,kBn;
    double *kAl, *kAn, *kBl, *kBn;
    double *kA1l, *kA1n, *kA2l, *kA2n, *kA3l, *kA3n;
    double *kB1l, *kB1n, *kB2l, *kB2n, *kB3l, *kB3n, *engij;
    int rand(void);
    clock_t start;
    double cpuTime;
    char run[4];
    int ig, myid, np, n, nend, nout, neng, neng2, ij, pc, is, p2, ntype, ibeg, iend, iint, ich;
    int itheta;
    fftw_plan plan1An, plan1Ab;
    fftw_plan plan2An, plan2Ab;
    fftw_plan plan3An, plan3Ab;
    fftw_plan plan1Bn, plan1Bb;
    fftw_plan plan2Bn, plan2Bb;
    fftw_plan plan3Bn, plan3Bb;
    fftw_plan plannA, plannB;
    fftw_plan plannAb, plannBb;
    fftw_plan plannAn, plannBn;
    fftw_complex ampA, ampB, A1o, A2o, A3o, B1o, B2o, B3o;
    fftw_complex *A1, *Ak1, *An1, *A2, *Ak2, *An2, *A3, *Ak3, *An3;
    fftw_complex *B1, *Bk1, *Bn1, *B2, *Bk2, *Bn2, *B3, *Bk3, *Bn3;
    fftw_complex *nA, *nB, *nAk, *nBk, *nAn, *nBn;

    ptrdiff_t alloc_local, Lxl, local_0_start, i, j;

    double noise; // 0.10; // 0.001;
    double dtS;
    int nend_smalldt = 100000; // 100000;
    int neng_smalldt = 10000;

    FILE *in;
    in = fopen("CN_LS2.in", "r");
    if (in == NULL) {
        printf("Cannot open file");
        perror("Cannot open file");
    }
    else
    {
        fscanf(in, "%s", run);
        fscanf(in, "%le %le %le\n", &dx, &dy, &dt);
        fscanf(in, "%le %le\n", &alpha, &Vo);
        fscanf(in, "%le %le %le %le %le %le\n", &ns, &nl, &epsA, &epsB, &vA, &vB);
        fscanf(in, "%le %le %le %le %le %le\n", &gA, &gB, &betaB, &alphaAB, &omega, &mu);
        fscanf(in, "%d %d %d %d\n", &nend, &nout, &neng2, &neng);
        fscanf(in, "%d %d %d \n", &ntype, &Lx, &Ly);
        fscanf(in, "%d  \n", &itheta);
        fscanf(in, "%le  %le %le %le %le %le", &aa, &aai, &bb, &bbi, &cc, &cci);
        A1o = aa + I * aai;
        A2o = bb + I * bbi;
        A3o = cc + I * cci;
        fscanf(in, "%le  %le %le %le %le %le", &aa, &aai, &bb, &bbi, &cc, &cci);
        B1o = aa + I * aai;
        B2o = bb + I * bbi;
        B3o = cc + I * cci;
        fscanf(in, "%le %le\n", &noise, &dtS);
        fscanf(in, "%d %d\n", &nend_smalldt, &neng_smalldt);
    }
    fclose(in);
    // fn=(3.67704032591281e-02+2.58282403054995e-01)/2.0;
    // fn=(0.258303076+0.03768087551)/2.0; printf(" fn = %e \n",fn);
    fn = 0.1479919553;
    ffac = 2.74;
    ao = 2.51;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    fftw_mpi_init();
    if (myid == 0)
    {
        A1sq = A1o * conj(A1o);
        A2sq = A2o * conj(A2o);
        A3sq = A3o * conj(A3o);
        B1sq = B1o * conj(B1o);
        B2sq = B2o * conj(B2o);
        B3sq = B3o * conj(B3o);
        Ssq = 2.0 * (A1sq + A2sq + A3sq);
        Tsq = 2.0 * (B1sq + B2sq + B3sq);
        printf(" myid = %i \n", myid);
        printf(" Ajsq = %e %e %e \n", A1sq, A2sq, A3sq);
        printf(" Bjsq = %e %e %e \n", B1sq, B2sq, B3sq);
        printf(" Ssq Tsq = %e %e \n", Ssq, Tsq);
        printf(" Lx, Ly, dx,dt = %i %i %e %e \n", Lx, Ly, dt, dx);
        printf(" Aj = %e %e %e %e %e %e \n", A1o, A2o, A3o);
        printf(" Bj = %e %e %e %e %e %e \n", B1o, B2o, B3o);
        printf(" noise = %e\n", noise);
        printf(" small dt = %e\n", dt*dtS);
        printf(" small steps checkpoint = %i %i\n", nend_smalldt, neng_smalldt);
    }
    // alphaAB=0.0;  omega=0.0; mu=0.0;
    /*	 for(io=ibeg;(io<iend+1&&io>-1);io=io+iint) {
            if(ntype==11) {Lx=io; Ly=(int)(Lx*sqrt(3.0)+.5); dy=sqrt(3.0)*Lx*dx/Ly;
                printf("Lx, Ly = %i %i %i %i %e \n",Lx,Ly,io,iint,dy);}
            if(ntype==12) {Lx=io; Ly=2;
                printf("Lx, Ly = %i %i %i %i %e \n",Lx,Ly,io,iint,dy);}*/
    alloc_local = fftw_mpi_local_size_2d(Lx, Ly, MPI_COMM_WORLD, &Lxl, &local_0_start);
    // printf("Lxl = %i \n",Lxl);
    nA = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);
    nAk = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);
    nAn = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);
    A1 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);
    A2 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);
    A3 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);
    Ak1 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);
    Ak2 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);
    Ak3 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);
    An1 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);
    An2 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);
    An3 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);

    nB = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);
    nBk = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);
    nBn = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);
    B1 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);
    B2 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);
    B3 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);
    Bk1 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);
    Bk2 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);
    Bk3 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);
    Bn1 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);
    Bn2 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);
    Bn3 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * alloc_local);

    kA1l = (double *)fftw_malloc(sizeof(double) * alloc_local);
    kA2l = (double *)fftw_malloc(sizeof(double) * alloc_local);
    kA3l = (double *)fftw_malloc(sizeof(double) * alloc_local);
    kA1n = (double *)fftw_malloc(sizeof(double) * alloc_local);
    kA2n = (double *)fftw_malloc(sizeof(double) * alloc_local);
    kA3n = (double *)fftw_malloc(sizeof(double) * alloc_local);

    kAl = (double *)fftw_malloc(sizeof(double) * alloc_local);
    kAn = (double *)fftw_malloc(sizeof(double) * alloc_local);
    kBl = (double *)fftw_malloc(sizeof(double) * alloc_local);
    kBn = (double *)fftw_malloc(sizeof(double) * alloc_local);
    kB1l = (double *)fftw_malloc(sizeof(double) * alloc_local);
    kB2l = (double *)fftw_malloc(sizeof(double) * alloc_local);
    kB3l = (double *)fftw_malloc(sizeof(double) * alloc_local);
    kB1n = (double *)fftw_malloc(sizeof(double) * alloc_local);
    kB2n = (double *)fftw_malloc(sizeof(double) * alloc_local);
    kB3n = (double *)fftw_malloc(sizeof(double) * alloc_local);
    engij = (double *)fftw_malloc(sizeof(double) * alloc_local);

    plannA = fftw_mpi_plan_dft_2d(Lx, Ly, nA, nAk, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
    plan1An = fftw_mpi_plan_dft_2d(Lx, Ly, An1, An1, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
    plan2An = fftw_mpi_plan_dft_2d(Lx, Ly, An2, An2, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
    plan3An = fftw_mpi_plan_dft_2d(Lx, Ly, An3, An3, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
    plannAn = fftw_mpi_plan_dft_2d(Lx, Ly, nAn, nAn, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);

    plan1Ab = fftw_mpi_plan_dft_2d(Lx, Ly, Ak1, A1, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
    plan2Ab = fftw_mpi_plan_dft_2d(Lx, Ly, Ak2, A2, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
    plan3Ab = fftw_mpi_plan_dft_2d(Lx, Ly, Ak3, A3, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
    plannAb = fftw_mpi_plan_dft_2d(Lx, Ly, nAk, nA, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);

    plannB = fftw_mpi_plan_dft_2d(Lx, Ly, nB, nBk, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
    plan1Bn = fftw_mpi_plan_dft_2d(Lx, Ly, Bn1, Bn1, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
    plan2Bn = fftw_mpi_plan_dft_2d(Lx, Ly, Bn2, Bn2, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
    plan3Bn = fftw_mpi_plan_dft_2d(Lx, Ly, Bn3, Bn3, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
    plannBn = fftw_mpi_plan_dft_2d(Lx, Ly, nBn, nBn, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);

    plan1Bb = fftw_mpi_plan_dft_2d(Lx, Ly, Bk1, B1, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
    plan2Bb = fftw_mpi_plan_dft_2d(Lx, Ly, Bk2, B2, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
    plan3Bb = fftw_mpi_plan_dft_2d(Lx, Ly, Bk3, B3, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
    plannBb = fftw_mpi_plan_dft_2d(Lx, Ly, nBk, nB, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);

    twopi = 2.0 * acos(-1.0);
    fx = twopi / (dx * Lx);
    fy = twopi / (dy * Ly);
    Sf = 1.0 / (Lx * Ly);
    // for (itheta = 0; itheta < 1; itheta++)
    {
        theta = itheta * twopi / 360.0;

        ich = 0;
        /* random
             srand(time(NULL)+myid);
                 for(i=0;i<Lxl;++i){for(j=0;j< Ly;++j){ij=i*Ly+j;
                A1[ij]=.05+0.5*((0.5-(rand() & 1000)/1000.)+I*(0.5-(rand() & 1000)/1000.));
                A2[ij]=.05+0.5*((0.5-(rand() & 1000)/1000.)+I*(0.5-(rand() & 1000)/1000.));
                A3[ij]=.05+0.5*((0.5-(rand() & 1000)/1000.)+I*(0.5-(rand() & 1000)/1000.));
                B1[ij]=.05+0.5*((0.5-(rand() & 1000)/1000.)+I*(0.5-(rand() & 1000)/1000.));
                B2[ij]=.05+0.5*((0.5-(rand() & 1000)/1000.)+I*(0.5-(rand() & 1000)/1000.));
                B3[ij]=.05+0.5*((0.5-(rand() & 1000)/1000.)+I*(0.5-(rand() & 1000)/1000.));
             }}
          */
        /*
                 for(i=0;i<Lxl;++i){for(j=0;j<Ly;++j){ij=i*Ly+j;
                A1[ij]=0.0; A2[ij]=0.0; A3[ij]=0.0;
                B1[ij]=0.0; B2[ij]=0.0; B3[ij]=0.0;
             }}
        */
        /* middle xtal
                 for(i=0;i<Lxl;++i){for(j=0;j<Ly;++j){ij=i*Ly+j;
                A1[ij]=(tanh(1.0*(j-Ly/2-10.))-tanh(1.0*(j-Ly/2+10.)))/2.0*(0.220364-0.023415*I);
                B1[ij]=(tanh(1.0*(j-Ly/2-10.))-tanh(1.0*(j-Ly/2+10.)))/2.0*(-0.130640+0.179134*I);
                A2[ij]=A1[ij]; A3[ij]=A1[ij];
                B2[ij]=B1[ij]; B3[ij]=B1[ij];
             }}
        */
        /* circle xtal  */
        if (ntype == 1)
        {
            for (i = 0; i < Lxl; ++i)
            {
                ig = i + myid * Lxl;
                for (j = 0; j < Ly; ++j)
                {
                    ij = i * Ly + j;
                    arg = sqrt((ig - Lx / 2) * (ig - Lx / 2) + (j - Ly / 2) * (j - Ly / 2));
                    A1[ij] = (tanh(arg - 2000.) - tanh(arg + 2000)) / 2.0 * (0.220364 - 0.023415 * I);
                    B1[ij] = (tanh(arg - 2000.) - tanh(arg + 2000.)) / 2.0 * (-0.130640 + 0.179134 * I);
                    A2[ij] = A1[ij];
                    A3[ij] = A1[ij];
                    B2[ij] = B1[ij];
                    B3[ij] = B1[ij];
                }
            }
        }
        if (ntype == 2)
        {
            for (i = 0; i < Lxl; ++i)
            {
                ig = i + myid * Lxl;
                for (j = 0; j < Ly; ++j)
                {
                    ij = i * Ly + j;
                    A1[ij] = 0.18;
                    B1[ij] = 0.18 * (cos(twopi / 3) - I * sin(twopi / 3));
                    A2[ij] = A1[ij];
                    A3[ij] = A1[ij];
                    B2[ij] = B1[ij];
                    B3[ij] = B1[ij];
                }
            }
        }
        if (ntype == 3)
        {
            printf(" ntype == %i \n", ntype);
            for (i = 0; i < Lxl; ++i)
            {
                ig = i + myid * Lxl;
                for (j = 0; j < Ly; ++j)
                {
                    ij = i * Ly + j;
                    A1[ij] = 0.18 * (cos(twopi / 3) + I * sin(twopi / 3));
                    B1[ij] = 0.18 * (cos(twopi / 3) - I * sin(twopi / 3));
                    A2[ij] = A1[ij];
                    A3[ij] = A1[ij];
                    B2[ij] = B1[ij];
                    B3[ij] = B1[ij];
                }
            }
        }
        if (ntype == 4)
        {
            if (myid == 0)
            {
                printf(" ntype == %i \n", ntype);
            }
            for (i = 0; i < Lxl; ++i)
            {
                ig = i + myid * Lxl;
                for (j = 0; j < Ly; ++j)
                {
                    ij = i * Ly + j;
                    A1[ij] = 0.0;
                    B1[ij] = 0.0;
                    A2[ij] = A1[ij];
                    A3[ij] = A1[ij];
                    B2[ij] = B1[ij];
                    B3[ij] = B1[ij];
                }
            }
        }
        if (ntype == 34)
        {
            if (myid == 0)
            {
                printf(" ntype == %i \n", ntype);
            }
            for (i = 0; i < Lxl; ++i)
            {
                ig = i + myid * Lxl;
                for (j = 0; j < Ly; ++j)
                {
                    ij = i * Ly + j;
                    if (ig < Lx / 2)
                    {
                        /*		A1[ij]=0.18*(cos(twopi/3)+I*sin(twopi/3)); B1[ij]=0.18*(cos(twopi/3)-I*sin(twopi/3));
                                A2[ij]=A1[ij]; A3[ij]=A1[ij];
                                B2[ij]=B1[ij]; B3[ij]=B1[ij];*/
                        A1[ij] = A1o;
                        A2[ij] = A2o;
                        A3[ij] = A3o;
                        B1[ij] = B1o;
                        B2[ij] = B2o;
                        B3[ij] = B3o;
                        nA[ij] = -0.26822407;
                        nB[ij] = nA[ij];
                    }
                    else
                    {
                        A1[ij] = 0.0;
                        B1[ij] = 0.0;
                        A2[ij] = A1[ij];
                        A3[ij] = A1[ij];
                        B2[ij] = B1[ij];
                        B3[ij] = B1[ij];
                        nA[ij] = -0.443516463;
                        nB[ij] = nA[ij];
                    }
                }
            }
        }
        if (ntype == 26)
        { // solid equil
            printf(" ntype == %i \n", ntype);
            // printf(" nl, ns = %f %f\n",nl,ns);
            for (i = 0; i < Lxl; ++i)
            {
                ig = i + myid * Lxl;
                for (j = 0; j < Ly; ++j)
                {
                    ij = i * Ly + j;
                    A1[ij] = A1o;
                    A2[ij] = A2o;
                    A3[ij] = A3o;
                    B1[ij] = B1o;
                    B2[ij] = B2o;
                    B3[ij] = B3o;
                    nA[ij] = 0.0;
                    nB[ij] = 0.0;
                    printf("A1 = %e %e \n", A1[ij]);
                }
            }
        }
        if (ntype == 25)
        {   // liquid equil
            // printf(" nl, ns = %f %f\n",nl,ns);
            for (i = 0; i < Lxl; ++i)
            {
                ig = i + myid * Lxl;
                for (j = 0; j < Ly; ++j)
                {
                    ij = i * Ly + j;
                    A1[ij] = 0.0;
                    A2[ij] = 0.0;
                    A3[ij] = 0.0;
                    B1[ij] = 0.0;
                    B2[ij] = 0.0;
                    B3[ij] = 0.0;
                    nA[ij] = 0.0;
                    nB[ij] = 0.0;
                }
            }
        }
        if (ntype == 99)
        {
            if (myid == 0)
            {
                printf(" ntype == %i \n", ntype);
            }
            for (i = 0; i < Lxl; ++i)
            {
                ig = i + myid * Lxl;
                for (j = 0; j < Ly; ++j)
                {
                    ij = i * Ly + j;
                    arg = sqrt((ig - Lx / 2) * (ig - Lx / 2) + (j - Ly / 2) * (j - Ly / 2));
                    if (arg < Lx / 4)
                    {
                        /*		A1[ij]=0.18*(cos(twopi/3)+I*sin(twopi/3)); B1[ij]=0.18*(cos(twopi/3)-I*sin(twopi/3));
                                A2[ij]=A1[ij]; A3[ij]=A1[ij];
                                B2[ij]=B1[ij]; B3[ij]=B1[ij];*/
                        A1[ij] = A1o;
                        A2[ij] = A2o;
                        A3[ij] = A3o;
                        B1[ij] = B1o;
                        B2[ij] = B2o;
                        B3[ij] = B3o;
                        nA[ij] = -0.26822407;
                        nB[ij] = nA[ij];
                    }
                    else
                    {
                        A1[ij] = 0.0;
                        B1[ij] = 0.0;
                        A2[ij] = A1[ij];
                        A3[ij] = A1[ij];
                        B2[ij] = B1[ij];
                        B3[ij] = B1[ij];
                        nA[ij] = -0.443516463;
                        nB[ij] = nA[ij];
                    }
                }
            }
        }
        /* random  */
        {
            srand(time(NULL)+myid);
            printf("Ajsq %e %e %e \n", A1sq, A2sq, A3sq);
            printf("Bjsq %e %e %e \n", B1sq, B2sq, B3sq);
            printf("rand p/m 10 percent %e \n", 0.1 * ((rand() % 1000) / 500. - 1.));
            for(i=0;i<Lxl;++i)
            {
                ig = i + myid * Lxl;
                for(j=0;j< Ly;++j)
                {
                    // if (ig < Lx / 2)
                    // {
                    ij=i*Ly+j;
                    A1[ij]+=(A1o + conj(A1o))/2. * noise * ((rand() % 1000)/500. - 1.) + (A1o - conj(A1o))/2. * noise * ((rand() % 1000)/500. - 1.);
                    A2[ij]+=(A2o + conj(A2o))/2. * noise * ((rand() % 1000)/500. - 1.) + (A2o - conj(A2o))/2. * noise * ((rand() % 1000)/500. - 1.);
                    A3[ij]+=(A3o + conj(A3o))/2. * noise * ((rand() % 1000)/500. - 1.) + (A3o - conj(A3o))/2. * noise * ((rand() % 1000)/500. - 1.);
                    B1[ij]+=(B1o + conj(B1o))/2. * noise * ((rand() % 1000)/500. - 1.) + (B1o - conj(B1o))/2. * noise * ((rand() % 1000)/500. - 1.);
                    B2[ij]+=(B2o + conj(B2o))/2. * noise * ((rand() % 1000)/500. - 1.) + (B2o - conj(B2o))/2. * noise * ((rand() % 1000)/500. - 1.);
                    B3[ij]+=(B3o + conj(B3o))/2. * noise * ((rand() % 1000)/500. - 1.) + (B3o - conj(B3o))/2. * noise * ((rand() % 1000)/500. - 1.);
                    nA[ij]+=nA[ij]*noise*((rand() % 1000)/500. - 1.);
                    nB[ij]+=nB[ij]*noise*((rand() % 1000)/500. - 1.);
                    // }
                }
            }
        }
        for (i = 0; i < Lxl; ++i)
        {
            for (j = 0; j < Ly; ++j)
            {
                ij = i * Ly + j;
                An1[ij] = A1[ij];
                An2[ij] = A2[ij];
                An3[ij] = A3[ij];
                Bn1[ij] = B1[ij];
                Bn2[ij] = B2[ij];
                Bn3[ij] = B3[ij];
            }
        }
        /*
                 for(i=0;i<Lxl;++i){for(j=0;j<Ly;++j){ij=i*Ly+j;
                A1[ij]=0.0; A2[ij]=0.0; A3[ij]=0.0;
                B1[ij]=0.0; B2[ij]=0.0; B3[ij]=0.0;
             }}
        */
        fftw_execute(plannA);
        fftw_execute(plannB);
        fftw_execute(plan1An);
        fftw_execute(plan2An);
        fftw_execute(plan3An);
        fftw_execute(plan1Bn);
        fftw_execute(plan2Bn);
        fftw_execute(plan3Bn);
        for (i = 0; i < Lxl; ++i)
        {
            for (j = 0; j < Ly; ++j)
            {
                ij = i * Ly + j;
                Ak1[ij] = An1[ij] * Sf;
                Ak2[ij] = An2[ij] * Sf;
                Ak3[ij] = An3[ij] * Sf;
                Bk1[ij] = Bn1[ij] * Sf;
                Bk2[ij] = Bn2[ij] * Sf;
                Bk3[ij] = Bn3[ij] * Sf;
                nAk[ij] = nAk[ij] * Sf;
                nBk[ij] = nBk[ij] * Sf;
            }
        }
        G1xo = -sqrt(3.0) / 2.0;
        G1yo = -1.0 / 2.0;
        G2xo = 0.0;
        G2yo = 1.0;
        G3xo = sqrt(3.0) / 2.0;
        G3yo = -1.0 / 2.0;
        G1x = G1xo * cos(theta) - G1yo * sin(theta);
        G1y = G1xo * sin(theta) + G1yo * cos(theta);
        G2x = G2xo * cos(theta) - G2yo * sin(theta);
        G2y = G2xo * sin(theta) + G2yo * cos(theta);
        G3x = G3xo * cos(theta) - G3yo * sin(theta);
        G3y = G3xo * sin(theta) + G3yo * cos(theta);
        {
            // run for a few seconds with dt/10 time steps
            dt *= dtS;
            for (i = 0; i < Lxl; ++i)
            {
                ig = i + myid * Lxl;
                if (ig < Lx / 2)
                {
                    kx = ig * fx;
                }
                else
                {
                    kx = (ig - Lx) * fx;
                }
                for (j = 0; j < Ly; ++j)
                {
                    ij = i * Ly + j;
                    if (j < Ly / 2)
                    {
                        ky = j * fy;
                    }
                    else
                    {
                        ky = (j - Ly) * fy;
                    }
                    ksq = kx * kx + ky * ky;
                    kA1f = -1.0 * (-epsA + (-ksq - 2.0 * alpha * (G1x * kx + G1y * ky) + 1. - alpha * alpha) * (-ksq - 2.0 * alpha * (G1x * kx + G1y * ky) + 1. - alpha * alpha));
                    kA2f = -1.0 * (-epsA + (-ksq - 2.0 * alpha * (G2x * kx + G2y * ky) + 1. - alpha * alpha) * (-ksq - 2.0 * alpha * (G2x * kx + G2y * ky) + 1. - alpha * alpha));
                    kA3f = -1.0 * (-epsA + (-ksq - 2.0 * alpha * (G3x * kx + G3y * ky) + 1. - alpha * alpha) * (-ksq - 2.0 * alpha * (G3x * kx + G3y * ky) + 1. - alpha * alpha));
                    kB1f = -1.0 * (-epsB + betaB * (-ksq - 2.0 * alpha * (G1x * kx + G1y * ky) + 1. - alpha * alpha) * (-ksq - 2.0 * alpha * (G1x * kx + G1y * ky) + 1. - alpha * alpha));
                    kB2f = -1.0 * (-epsB + betaB * (-ksq - 2.0 * alpha * (G2x * kx + G2y * ky) + 1. - alpha * alpha) * (-ksq - 2.0 * alpha * (G2x * kx + G2y * ky) + 1. - alpha * alpha));
                    kB3f = -1.0 * (-epsB + betaB * (-ksq - 2.0 * alpha * (G3x * kx + G3y * ky) + 1. - alpha * alpha) * (-ksq - 2.0 * alpha * (G3x * kx + G3y * ky) + 1. - alpha * alpha));
                    kA1l[ij] = exp(kA1f * dt);
                    kA2l[ij] = exp(kA2f * dt);
                    kA3l[ij] = exp(kA3f * dt);
                    kB1l[ij] = exp(kB1f * dt);
                    kB2l[ij] = exp(kB2f * dt);
                    kB3l[ij] = exp(kB3f * dt);
                    if (kA1f == 0)
                    {
                        kA1n[ij] = -dt * Sf;
                    }
                    else
                    {
                        kA1n[ij] = ((1.0 - exp(kA1f * dt)) / kA1f) * Sf;
                    }
                    if (kA2f == 0)
                    {
                        kA2n[ij] = -dt * Sf;
                    }
                    else
                    {
                        kA2n[ij] = ((1.0 - exp(kA2f * dt)) / kA2f) * Sf;
                    }
                    if (kA3f == 0)
                    {
                        kA3n[ij] = -dt * Sf;
                    }
                    else
                    {
                        kA3n[ij] = ((1.0 - exp(kA3f * dt)) / kA3f) * Sf;
                    }
                    if (kB1f == 0)
                    {
                        kB1n[ij] = -dt * Sf;
                    }
                    else
                    {
                        kB1n[ij] = ((1.0 - exp(kB1f * dt)) / kB1f) * Sf;
                    }
                    if (kB2f == 0)
                    {
                        kB2n[ij] = -dt * Sf;
                    }
                    else
                    {
                        kB2n[ij] = ((1.0 - exp(kB2f * dt)) / kB2f) * Sf;
                    }
                    if (kB3f == 0)
                    {
                        kB3n[ij] = -dt * Sf;
                    }
                    else
                    {
                        kB3n[ij] = ((1.0 - exp(kB3f * dt)) / kB3f) * Sf;
                    }
                }
            }
            kA3f = -epsA + 1.0;
            kB3f = -epsB + betaB;
            for (i = 0; i < Lxl; ++i)
            {
                ig = i + myid * Lxl;
                if (ig < Lx / 2)
                {
                    kx = ig * fx;
                }
                else
                {
                    kx = (ig - Lx) * fx;
                }
                for (j = 0; j < Ly; ++j)
                {
                    ij = i * Ly + j;
                    if (j < Ly / 2)
                    {
                        ky = j * fy;
                    }
                    else
                    {
                        ky = (j - Ly) * fy;
                    }
                    ksq = kx * kx + ky * ky;
                    kAl[ij] = exp(-ksq * kA3f * dt);
                    if (ksq == 0)
                    {
                        kAn[ij] = 0;
                    }
                    else
                    {
                        kAn[ij] = (exp(-ksq * kA3f * dt) - 1.0) / kA3f * Sf;
                    }
                    kBl[ij] = exp(-ksq * kB3f * dt);
                    if (ksq == 0)
                    {
                        kBn[ij] = 0;
                    }
                    else
                    {
                        kBn[ij] = (exp(-ksq * kB3f * dt) - 1.0) / kB3f * Sf;
                    }
                }
            }

            // ouptut initial fields
            {
                char filename[BUFSIZ], fileB[BUFSIZ];
                FILE *fout, *foutB;
                sprintf(filename, "%s_init_smalldt_%d.Adat", run, itheta);
                sprintf(fileB, "%s_init_small_%d.Bdat", run, itheta);
                //		      if(myid==0) {printf("output n, elapsed time = %i %e \n",n,cpuTime);}
                MPI_Barrier(MPI_COMM_WORLD);
                for (pc = 0; pc < np; pc++)
                {
                    MPI_Barrier(MPI_COMM_WORLD);
                    if (myid == pc)
                    {
                        if (myid == 0)
                        {
                            fout = fopen(filename, "w");
                            foutB = fopen(fileB, "w");
                        }
                        else
                        {
                            fout = fopen(filename, "a");
                            foutB = fopen(fileB, "a");
                        }
                        for (i = 0; i < Lxl; ++i)
                        {
                            ig = i + myid * Lxl;
                            for (j = 0; j < Ly; ++j)
                            {
                                ij = i * Ly + j;
                                A1sq = A1[ij] * conj(A1[ij]);
                                A2sq = A2[ij] * conj(A2[ij]);
                                A3sq = A3[ij] * conj(A3[ij]);
                                B1sq = B1[ij] * conj(B1[ij]);
                                B2sq = B2[ij] * conj(B2[ij]);
                                B3sq = B3[ij] * conj(B3[ij]);
                                Ssq = 2.0 * (A1sq + A2sq + A3sq);
                                Tsq = 2.0 * (B1sq + B2sq + B3sq);
                                fprintf(fout, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nA[ij], A1[ij], A2[ij], A3[ij], Ssq);
                                fprintf(foutB, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nB[ij], B1[ij], B2[ij], B3[ij], Tsq);
                            }
                        }
                        fclose(fout);
                        fclose(foutB);
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            }
            { // every nout time steps output data
                char filename[BUFSIZ], fileB[BUFSIZ];
                FILE *fout, *foutB;
                sprintf(filename, "%s_init_smalldt_%d.Andat", run, itheta);
                sprintf(fileB, "%s_init_smalldt_%d.Bndat", run, itheta);
                //		      if(myid==0) {printf("output n, elapsed time = %i %e \n",n,cpuTime);}
                MPI_Barrier(MPI_COMM_WORLD);
                for (pc = 0; pc < np; pc++)
                {
                    MPI_Barrier(MPI_COMM_WORLD);
                    if (myid == pc)
                    {
                        if (myid == 0)
                        {
                            fout = fopen(filename, "w");
                            foutB = fopen(fileB, "w");
                        }
                        else
                        {
                            fout = fopen(filename, "a");
                            foutB = fopen(fileB, "a");
                        }
                        for (i = 0; i < Lxl; ++i)
                        {
                            ig = i + myid * Lxl;
                            for (j = 0; j < Ly; ++j)
                            {
                                ij = i * Ly + j;
                                A1sq = A1[ij] * conj(A1[ij]);
                                A2sq = A2[ij] * conj(A2[ij]);
                                A3sq = A3[ij] * conj(A3[ij]);
                                B1sq = B1[ij] * conj(B1[ij]);
                                B2sq = B2[ij] * conj(B2[ij]);
                                B3sq = B3[ij] * conj(B3[ij]);
                                Ssq = 2.0 * (A1sq + A2sq + A3sq);
                                Tsq = 2.0 * (B1sq + B2sq + B3sq);
                                fprintf(fout, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nAn[ij], An1[ij], An2[ij], An3[ij], Ssq);
                                fprintf(foutB, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nBn[ij], Bn1[ij], Bn2[ij], Bn3[ij], Tsq);
                            }
                        }
                        fclose(fout);
                        fclose(foutB);
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                }
                start = clock();
            }
            { // every nout time steps output data
                char filename[BUFSIZ], fileB[BUFSIZ];
                FILE *fout, *foutB;
                sprintf(filename, "%s_init_smalldt_%d.Akdat", run, itheta);
                sprintf(fileB, "%s_init_smalldt_%d.Bkdat", run, itheta);
                //		      if(myid==0) {printf("output n, elapsed time = %i %e \n",n,cpuTime);}
                MPI_Barrier(MPI_COMM_WORLD);
                for (pc = 0; pc < np; pc++)
                {
                    MPI_Barrier(MPI_COMM_WORLD);
                    if (myid == pc)
                    {
                        if (myid == 0)
                        {
                            fout = fopen(filename, "w");
                            foutB = fopen(fileB, "w");
                        }
                        else
                        {
                            fout = fopen(filename, "a");
                            foutB = fopen(fileB, "a");
                        }
                        for (i = 0; i < Lxl; ++i)
                        {
                            ig = i + myid * Lxl;
                            for (j = 0; j < Ly; ++j)
                            {
                                ij = i * Ly + j;
                                A1sq = A1[ij] * conj(A1[ij]);
                                A2sq = A2[ij] * conj(A2[ij]);
                                A3sq = A3[ij] * conj(A3[ij]);
                                B1sq = B1[ij] * conj(B1[ij]);
                                B2sq = B2[ij] * conj(B2[ij]);
                                B3sq = B3[ij] * conj(B3[ij]);
                                Ssq = 2.0 * (A1sq + A2sq + A3sq);
                                Tsq = 2.0 * (B1sq + B2sq + B3sq);
                                fprintf(fout, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nAk[ij], Ak1[ij], Ak2[ij], Ak3[ij], Ssq);
                                fprintf(foutB, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nBk[ij], Bk1[ij], Bk2[ij], Bk3[ij], Tsq);
                            }
                        }
                        fclose(fout);
                        fclose(foutB);
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                }
                start = clock();
            }
    
            vA3 = 3.0 * vA;
            vB3 = 3.0 * vB;
            start = clock();
            muA = 0.0;
            muB = 0.0;
            for (n = 0; n < nend_smalldt + 1; ++n)
            {
                for (i = 0; i < Lxl; ++i)
                {
                    for (j = 0; j < Ly; ++j)
                    {
                        ij = i * Ly + j;
                        A1sq = A1[ij] * conj(A1[ij]);
                        A2sq = A2[ij] * conj(A2[ij]);
                        A3sq = A3[ij] * conj(A3[ij]);
                        B1sq = B1[ij] * conj(B1[ij]);
                        B2sq = B2[ij] * conj(B2[ij]);
                        B3sq = B3[ij] * conj(B3[ij]);
                        Ssq = 2.0 * (A1sq + A2sq + A3sq);
                        Tsq = 2.0 * (B1sq + B2sq + B3sq);
                        gnAo = 2.0 * (3.0 * vA * nA[ij] - gA); // gnAo = 2.0 * (3.0 * nA[ij] - gA);
                        gnBo = 2.0 * (3.0 * vA * nB[ij] - gB); // gnBo = 2.0 * (3.0 * nB[ij] - gB);
                        aonm = alphaAB + omega * nA[ij] + mu * nB[ij];
                        naf = nA[ij] * (vA3 * nA[ij] - 2 * gA) + omega * nB[ij];
                        nbf = nB[ij] * (vB3 * nB[ij] - 2 * gB) + mu * nA[ij];

                        // Eq. 19 check 08/05/23
                        An1[ij] = (vA3 * (Ssq - A1sq) + naf) * A1[ij] + gnAo * conj(A2[ij] * A3[ij]) + aonm * B1[ij] + omega * (conj(A2[ij] * B3[ij]) + conj(A3[ij] * B2[ij])) + mu * conj(B2[ij] * B3[ij]) + Vo; //+VA1[ij];

                        An2[ij] = (vA3 * (Ssq - A2sq) + naf) * A2[ij] + gnAo * conj(A1[ij] * A3[ij]) + aonm * B2[ij] + omega * (conj(A1[ij] * B3[ij]) + conj(A3[ij] * B1[ij])) + mu * conj(B1[ij] * B3[ij]) + Vo; //+VA2[ij];

                        An3[ij] = (vA3 * (Ssq - A3sq) + naf) * A3[ij] + gnAo * conj(A1[ij] * A2[ij]) + aonm * B3[ij] + omega * (conj(A1[ij] * B2[ij]) + conj(A2[ij] * B1[ij])) + mu * conj(B1[ij] * B2[ij]) + Vo; //+VA3[ij];

                        // Eq. 21 check 08/05/23
                        Bn1[ij] = (vB3 * (Tsq - B1sq) + nbf) * B1[ij] + gnBo * conj(B2[ij] * B3[ij]) + aonm * A1[ij] + mu * (conj(A2[ij] * B3[ij]) + conj(A3[ij] * B2[ij])) + omega * (conj(A2[ij] * A3[ij])) + Vo; //+VB1[ij];

                        Bn2[ij] = (vB3 * (Tsq - B2sq) + nbf) * B2[ij] + gnBo * conj(B1[ij] * B3[ij]) + aonm * A2[ij] + mu * (conj(A1[ij] * B3[ij]) + conj(A3[ij] * B1[ij])) + omega * conj(A1[ij] * A3[ij]) + Vo; //+VB2[ij];

                        Bn3[ij] = (vB3 * (Tsq - B3sq) + nbf) * B3[ij] + gnBo * conj(B1[ij] * B2[ij]) + aonm * A3[ij] + mu * (conj(A1[ij] * B2[ij]) + conj(A2[ij] * B1[ij])) + omega * conj(A1[ij] * A2[ij]) + Vo; //+VB3[ij];

                        // Eq. 22
                        nAn[ij] = nA[ij] * nA[ij] * (-gA + vA * nA[ij]) + 0.5 * (gnAo * Ssq + mu * Tsq) + 6.0 * vA * (A1[ij] * A2[ij] * A3[ij] + conj(A1[ij] * A2[ij] * A3[ij])) + omega * (A1[ij] * conj(B1[ij]) + A2[ij] * conj(B2[ij]) + A3[ij] * conj(B3[ij]) + B1[ij] * conj(A1[ij]) + B2[ij] * conj(A2[ij]) + B3[ij] * conj(A3[ij])) + nB[ij] * (alphaAB + omega * nA[ij] + 0.5 * mu * nB[ij]);
                        // Eq. 24
                        nBn[ij] = nB[ij] * nB[ij] * (-gB + vB * nB[ij]) + 0.5 * (gnBo * Tsq + omega * Ssq) + 6.0 * vB * (B1[ij] * B2[ij] * B3[ij] + conj(B1[ij] * B2[ij] * B3[ij])) + mu * (A1[ij] * conj(B1[ij]) + A2[ij] * conj(B2[ij]) + A3[ij] * conj(B3[ij]) + B1[ij] * conj(A1[ij]) + B2[ij] * conj(A2[ij]) + B3[ij] * conj(A3[ij])) + nA[ij] * (alphaAB + mu * nB[ij] + 0.5 * omega * nA[ij]);
                        // printf("%i %i %e %e %e %e\n",i,j,nA[ij],nB[ij],kAl,kAn);
                        // printf("%i %i %e %e \n",i,j,nA[ij],nB[ij]);
                        // printf("%i %i %e %e %e %e\n",i,j,nA[ij],nB[ij],kAl,kAn);
                        // printf("%i %i %e %e \n",i,j,nA[ij],nB[ij]);
                    }
                }
                //        for(i=0;i<Lxl;++i){for(j=0;j< Ly;++j){ij=i*Ly+j;
                //		printf("%i %i %e %e %e %e \n",i,j,An1[ij],Bn1[ij]);
                //	     }}
                fftw_execute(plannAn);
                fftw_execute(plannBn);
                fftw_execute(plan1An);
                fftw_execute(plan2An);
                fftw_execute(plan3An);
                fftw_execute(plan1Bn);
                fftw_execute(plan2Bn);
                fftw_execute(plan3Bn);
                for (i = 0; i < Lxl; ++i)
                {
                    for (j = 0; j < Ly; ++j)
                    {
                        ij = i * Ly + j;
                        Ak1[ij] = kA1l[ij] * Ak1[ij] + kA1n[ij] * An1[ij];
                        Ak2[ij] = kA2l[ij] * Ak2[ij] + kA2n[ij] * An2[ij];
                        Ak3[ij] = kA3l[ij] * Ak3[ij] + kA3n[ij] * An3[ij];
                        Bk1[ij] = kB1l[ij] * Bk1[ij] + kB1n[ij] * Bn1[ij];
                        Bk2[ij] = kB2l[ij] * Bk2[ij] + kB2n[ij] * Bn2[ij];
                        Bk3[ij] = kB3l[ij] * Bk3[ij] + kB3n[ij] * Bn3[ij];
                        nAk[ij] = kAl[ij] * nAk[ij] + kAn[ij] * nAn[ij];
                        nBk[ij] = kBl[ij] * nBk[ij] + kBn[ij] * nBn[ij];
                    }
                }
                fftw_execute(plannAb);
                fftw_execute(plannBb);
                fftw_execute(plan1Ab);
                fftw_execute(plan2Ab);
                fftw_execute(plan3Ab);
                fftw_execute(plan1Bb);
                fftw_execute(plan2Bb);
                fftw_execute(plan3Bb);
                //         for(i=0;i<Lxl;++i){for(j=0;j< Ly;++j){ij=i*Ly+j;
                //		printf("A %i %i %e %e %e %e \n",i,j,An1[ij],Bn1[ij]);
                //	     }}
                /*         for(i=0;i<Lxl;++i){for(j=0;j< Ly;++j){ij=i*Ly+j;
                            An1[ij]=A1[ij]; An2[ij]=A2[ij]; An3[ij]=A3[ij];
                            Bn1[ij]=B1[ij]; Bn2[ij]=B2[ij]; Bn3[ij]=B3[ij];
                        }}
                    fftw_execute(plannA); fftw_execute(plannB);
                    fftw_execute(plan1An); fftw_execute(plan2An); fftw_execute(plan3An);
                    fftw_execute(plan1Bn); fftw_execute(plan2Bn); fftw_execute(plan3Bn);
                        for(i=0;i<Lxl;++i){for(j=0;j< Ly;++j){ij=i*Ly+j;
                            Ak1[ij]=An1[ij]*Sf; Ak2[ij]=An2[ij]*Sf; Ak3[ij]=An3[ij]*Sf;
                            Bk1[ij]=Bn1[ij]*Sf; Bk2[ij]=Bn2[ij]*Sf; Bk3[ij]=Bn3[ij]*Sf;
                            nAk[ij]=nAk[ij]*Sf; nBk[ij]=nBk[ij]*Sf;
                    }}  */

                if (n % neng_smalldt == 0) // if (n % nout == 0)
                { // every nout time steps output data
                    cpuTime = (clock() - start) / (CLOCKS_PER_SEC);
                    char filename[BUFSIZ], fileB[BUFSIZ];
                    FILE *fout, *foutB;
                    sprintf(filename, "%s_%d_smalldt_%d.Adat", run, n, itheta);
                    sprintf(fileB, "%s_%d_smalldt_%d.Bdat", run, n, itheta);
                    //		      if(myid==0) {printf("output n, elapsed time = %i %e \n",n,cpuTime);}
                    MPI_Barrier(MPI_COMM_WORLD);
                    for (pc = 0; pc < np; pc++)
                    {
                        MPI_Barrier(MPI_COMM_WORLD);
                        if (myid == pc)
                        {
                            if (myid == 0)
                            {
                                fout = fopen(filename, "w");
                                foutB = fopen(fileB, "w");
                            }
                            else
                            {
                                fout = fopen(filename, "a");
                                foutB = fopen(fileB, "a");
                            }
                            for (i = 0; i < Lxl; ++i)
                            {
                                ig = i + myid * Lxl;
                                for (j = 0; j < Ly; ++j)
                                {
                                    ij = i * Ly + j;
                                    A1sq = A1[ij] * conj(A1[ij]);
                                    A2sq = A2[ij] * conj(A2[ij]);
                                    A3sq = A3[ij] * conj(A3[ij]);
                                    B1sq = B1[ij] * conj(B1[ij]);
                                    B2sq = B2[ij] * conj(B2[ij]);
                                    B3sq = B3[ij] * conj(B3[ij]);
                                    Ssq = 2.0 * (A1sq + A2sq + A3sq);
                                    Tsq = 2.0 * (B1sq + B2sq + B3sq);
                                    fprintf(fout, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nA[ij], A1[ij], A2[ij], A3[ij], Ssq);
                                    fprintf(foutB, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nB[ij], B1[ij], B2[ij], B3[ij], Tsq);
                                }
                            }
                            fclose(fout);
                            fclose(foutB);
                        }
                        MPI_Barrier(MPI_COMM_WORLD);
                    }
                }
                // if (n % neng_smalldt == 0) // if (n % nout == 0)
                // { // every nout time steps output data
                //     cpuTime = (clock() - start) / (CLOCKS_PER_SEC);
                //     char filename[BUFSIZ], fileB[BUFSIZ];
                //     FILE *fout, *foutB;
                //     sprintf(filename, "%s_%d_smalldt_%d.Andat", run, n, itheta);
                //     sprintf(fileB, "%s_%d_smalldt_%d.Bndat", run, n, itheta);
                //     //		      if(myid==0) {printf("output n, elapsed time = %i %e \n",n,cpuTime);}
                //     MPI_Barrier(MPI_COMM_WORLD);
                //     for (pc = 0; pc < np; pc++)
                //     {
                //         MPI_Barrier(MPI_COMM_WORLD);
                //         if (myid == pc)
                //         {
                //             if (myid == 0)
                //             {
                //                 fout = fopen(filename, "w");
                //                 foutB = fopen(fileB, "w");
                //             }
                //             else
                //             {
                //                 fout = fopen(filename, "a");
                //                 foutB = fopen(fileB, "a");
                //             }
                //             for (i = 0; i < Lxl; ++i)
                //             {
                //                 ig = i + myid * Lxl;
                //                 for (j = 0; j < Ly; ++j)
                //                 {
                //                     ij = i * Ly + j;
                //                     A1sq = A1[ij] * conj(A1[ij]);
                //                     A2sq = A2[ij] * conj(A2[ij]);
                //                     A3sq = A3[ij] * conj(A3[ij]);
                //                     B1sq = B1[ij] * conj(B1[ij]);
                //                     B2sq = B2[ij] * conj(B2[ij]);
                //                     B3sq = B3[ij] * conj(B3[ij]);
                //                     Ssq = 2.0 * (A1sq + A2sq + A3sq);
                //                     Tsq = 2.0 * (B1sq + B2sq + B3sq);
                //                     fprintf(fout, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nAn[ij], An1[ij], An2[ij], An3[ij], Ssq);
                //                     fprintf(foutB, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nBn[ij], Bn1[ij], Bn2[ij], Bn3[ij], Tsq);
                //                 }
                //             }
                //             fclose(fout);
                //             fclose(foutB);
                //         }
                //         MPI_Barrier(MPI_COMM_WORLD);
                //     }
                // }
                // if (n % neng_smalldt == 0) // if (n % nout == 0)
                // { // every nout time steps output data
                //     cpuTime = (clock() - start) / (CLOCKS_PER_SEC);
                //     char filename[BUFSIZ], fileB[BUFSIZ];
                //     FILE *fout, *foutB;
                //     sprintf(filename, "%s_%d_smalldt_%d.Akdat", run, n, itheta);
                //     sprintf(fileB, "%s_%d_smalldt_%d.Bkdat", run, n, itheta);
                //     //		      if(myid==0) {printf("output n, elapsed time = %i %e \n",n,cpuTime);}
                //     MPI_Barrier(MPI_COMM_WORLD);
                //     for (pc = 0; pc < np; pc++)
                //     {
                //         MPI_Barrier(MPI_COMM_WORLD);
                //         if (myid == pc)
                //         {
                //             if (myid == 0)
                //             {
                //                 fout = fopen(filename, "w");
                //                 foutB = fopen(fileB, "w");
                //             }
                //             else
                //             {
                //                 fout = fopen(filename, "a");
                //                 foutB = fopen(fileB, "a");
                //             }
                //             for (i = 0; i < Lxl; ++i)
                //             {
                //                 ig = i + myid * Lxl;
                //                 for (j = 0; j < Ly; ++j)
                //                 {
                //                     ij = i * Ly + j;
                //                     A1sq = A1[ij] * conj(A1[ij]);
                //                     A2sq = A2[ij] * conj(A2[ij]);
                //                     A3sq = A3[ij] * conj(A3[ij]);
                //                     B1sq = B1[ij] * conj(B1[ij]);
                //                     B2sq = B2[ij] * conj(B2[ij]);
                //                     B3sq = B3[ij] * conj(B3[ij]);
                //                     Ssq = 2.0 * (A1sq + A2sq + A3sq);
                //                     Tsq = 2.0 * (B1sq + B2sq + B3sq);
                //                     fprintf(fout, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nAk[ij], Ak1[ij], Ak2[ij], Ak3[ij], Ssq);
                //                     fprintf(foutB, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nBk[ij], Bk1[ij], Bk2[ij], Bk3[ij], Tsq);
                //                 }
                //             }
                //             fclose(fout);
                //             fclose(foutB);
                //         }
                //         MPI_Barrier(MPI_COMM_WORLD);
                //     }
                // }
                if (n % neng_smalldt == 0) // if (n % neng2 == 0)
                {
                    char filename[BUFSIZ];
                    FILE *fout;
                    enl = 0.0;
                    // alternate (slower) method to calculate gradient energy
                    for (i = 0; i < Lxl; ++i)
                    {
                        ig = i + myid * Lxl;
                        if (ig < Lx / 2)
                        {
                            kx = ig * fx;
                        }
                        else
                        {
                            kx = (ig - Lx) * fx;
                        }
                        for (j = 0; j < Ly; ++j)
                        {
                            ij = i * Ly + j;
                            if (j < Ly / 2)
                            {
                                ky = j * fy;
                            }
                            else
                            {
                                ky = (j - Ly) * fy;
                            }

                            kA1f = -(kx * kx + ky * ky) - 2.0 * (G1x * kx + G1y * ky) * alpha + 1.0 - alpha * alpha;
                            kB1f = kA1f;
                            // kB1f=-(kx*kx+ky*ky)-2.0*(G1x*kx+G1y*ky)*alpha+1.0-alpha*alpha;

                            kA2f = -(kx * kx + ky * ky) - 2.0 * (G2x * kx + G2y * ky) * alpha + 1.0 - alpha * alpha;
                            kB2f = kA2f;
                            // kB2f=-(kx*kx+ky*ky)-2.0*(G2x*kx+G2y*ky)*alpha+1.0-alpha*alpha;

                            kA3f = -(kx * kx + ky * ky) - 2.0 * (G3x * kx + G3y * ky) * alpha + 1.0 - alpha * alpha;
                            kB3f = kA3f;
                            // kB3f=-(kx*kx+ky*ky)-2.0*(G3x*kx+G3y*ky)*alpha+1.0-alpha*alpha;
                            //
                            An1[ij] = A1[ij];
                            An2[ij] = A2[ij];
                            An3[ij] = A3[ij];
                            Bn1[ij] = B1[ij];
                            Bn2[ij] = B2[ij];
                            Bn3[ij] = B3[ij];
                            Ak1[ij] = kA1f * Ak1[ij];
                            Ak2[ij] = kA2f * Ak2[ij];
                            Ak3[ij] = kA3f * Ak3[ij];
                            Bk1[ij] = kB1f * Bk1[ij];
                            Bk2[ij] = kB2f * Bk2[ij];
                            Bk3[ij] = kB3f * Bk3[ij];
                        }
                    }
                    fftw_execute(plan1Ab);
                    fftw_execute(plan2Ab);
                    fftw_execute(plan3Ab);
                    fftw_execute(plan1Bb);
                    fftw_execute(plan2Bb);
                    fftw_execute(plan3Bb);
                    for (i = 0; i < Lxl; ++i)
                    {
                        for (j = 0; j < Ly; ++j)
                        {
                            ij = i * Ly + j;
                            engij[ij] = A1[ij] * conj(A1[ij]) + A2[ij] * conj(A2[ij]) + A3[ij] * conj(A3[ij]);
                            engij[ij] = engij[ij] + B1[ij] * conj(B1[ij]) + B2[ij] * conj(B2[ij]) + B3[ij] * conj(B3[ij]);
                        }
                    }
                    for (i = 0; i < Lxl; ++i)
                    {
                        for (j = 0; j < Ly; ++j)
                        {
                            ij = i * Ly + j;
                            A1[ij] = An1[ij];
                            A2[ij] = An2[ij];
                            A3[ij] = An3[ij];
                            B1[ij] = Bn1[ij];
                            B2[ij] = Bn2[ij];
                            B3[ij] = Bn3[ij];
                            An1[ij] = A1[ij];
                            An2[ij] = A2[ij];
                            An3[ij] = A3[ij];
                            Bn1[ij] = B1[ij];
                            Bn2[ij] = B2[ij];
                            Bn3[ij] = B3[ij];
                        }
                    }
                    fftw_execute(plan1An);
                    fftw_execute(plan2An);
                    fftw_execute(plan3An);
                    fftw_execute(plan1Bn);
                    fftw_execute(plan2Bn);
                    fftw_execute(plan3Bn);
                    for (i = 0; i < Lxl; ++i)
                    {
                        for (j = 0; j < Ly; ++j)
                        {
                            ij = i * Ly + j;
                            Ak1[ij] = An1[ij] * Sf;
                            Ak2[ij] = An2[ij] * Sf;
                            Ak3[ij] = An3[ij] * Sf;
                            Bk1[ij] = Bn1[ij] * Sf;
                            Bk2[ij] = Bn2[ij] * Sf;
                            Bk3[ij] = Bn3[ij] * Sf;
                        }
                    }
                    /*                 for(i=0;i<Lxl;++i){ig=i+myid*Lxl;
                                    if(ig< Lx/2){kx=ig*fx;} else {kx=(ig-Lx)*fx;}
                                        for(j=0;j< Ly;++j){ij=i*Ly+j;
                                    if(j< Ly/2){ky=j*fy;} else {ky=(j-Ly)*fy;}

                                kA1f=-(kx*kx+ky*ky)-2.0*(G1x*kx+G1y*ky)*alpha+1.0-alpha*alpha; kB1f=kA1f;

                                kA2f=-(kx*kx+ky*ky)-2.0*(G2x*kx+G2y*ky)*alpha+1.0-alpha*alpha; kB2f=kA2f;

                                kA3f=-(kx*kx+ky*ky)-2.0*(G3x*kx+G3y*ky)*alpha+1.0-alpha*alpha; kB3f=kA3f;

                                    engij[ij]=kA1f*kA1f*Ak1[ij]*conj(Ak1[ij])+kA2f*kA2f*Ak2[ij]*conj(Ak2[ij])
                                        +kA3f*kA3f*Ak3[ij]*conj(Ak3[ij])
                                    +betaB*(kB1f*kB1f*Bk1[ij]*conj(Bk1[ij])+kB2f*kB2f*Bk2[ij]*conj(Bk2[ij])
                                        +kB3f*kB3f*Bk3[ij]*conj(Bk3[ij]));
                                engij[ij]=engij[ij]*Lx*Ly;
                                    }}                 */
                    for (i = 0; i < Lxl; ++i)
                    {
                        for (j = 0; j < Ly; ++j)
                        {
                            ij = i * Ly + j;
                            // Eq. 18  check 08/05/23
                            Ssq = A1[ij] * conj(A1[ij]) + A2[ij] * conj(A2[ij]) + A3[ij] * conj(A3[ij]);
                            Tsq = B1[ij] * conj(B1[ij]) + B2[ij] * conj(B2[ij]) + B3[ij] * conj(B3[ij]);
                            gnAo = 2.0 * (3.0 * nA[ij] * vA - gA);
                            gnBo = 2.0 * (3.0 * nB[ij] * vB - gB);
                            aonm = alphaAB + omega * nA[ij] + mu * nB[ij];
                            Asqcoeff = -epsA + nA[ij] * (3.0 * vA * nA[ij] - 2.0 * gA) + omega * nB[ij];
                            Bsqcoeff = -epsB + nB[ij] * (3.0 * vB * nB[ij] - 2.0 * gB) + mu * nA[ij];
                            engij[ij] = engij[ij] + Asqcoeff * Ssq + Bsqcoeff * Tsq + gnAo * (A1[ij] * A2[ij] * A3[ij] + conj(A1[ij] * A2[ij] * A3[ij])) + gnBo * (B1[ij] * B2[ij] * B3[ij] + conj(B1[ij] * B2[ij] * B3[ij])) + vA3 * Ssq * Ssq + vB3 * Tsq * Tsq - 1.5 * vA * (A1[ij] * conj(A1[ij]) * A1[ij] * conj(A1[ij]) + A2[ij] * conj(A2[ij]) * A2[ij] * conj(A2[ij]) + A3[ij] * conj(A3[ij]) * A3[ij] * conj(A3[ij])) - 1.5 * vB * (B1[ij] * conj(B1[ij]) * B1[ij] * conj(B1[ij]) + B2[ij] * conj(B2[ij]) * B2[ij] * conj(B2[ij]) + B3[ij] * conj(B3[ij]) * B3[ij] * conj(B3[ij])) + aonm * (A1[ij] * conj(B1[ij]) + B1[ij] * conj(A1[ij]) + A2[ij] * conj(B2[ij]) + B2[ij] * conj(A2[ij]) + A3[ij] * conj(B3[ij]) + B3[ij] * conj(A3[ij])) + omega * (A1[ij] * A2[ij] * B3[ij] + conj(A1[ij] * A2[ij] * B3[ij]) + A1[ij] * A3[ij] * B2[ij] + conj(A1[ij] * A3[ij] * B2[ij]) + A2[ij] * A3[ij] * B1[ij] + conj(A2[ij] * A3[ij] * B1[ij])) + mu * (A1[ij] * B2[ij] * B3[ij] + conj(A1[ij] * B2[ij] * B3[ij]) + A2[ij] * B1[ij] * B3[ij] + conj(A2[ij] * B1[ij] * B3[ij]) + A3[ij] * B1[ij] * B2[ij] + conj(A3[ij] * B1[ij] * B2[ij])) + nA[ij] * nA[ij] * (0.5 * (-epsA + 1.0) + nA[ij] * (-gA / 3. + 0.25 * vA * nA[ij])) + nB[ij] * nB[ij] * (0.5 * (-epsB + betaB) + nB[ij] * (-gB / 3. + 0.25 * vB * nB[ij])) + nA[ij] * nB[ij] * (alphaAB + 0.5 * (omega * nA[ij] + mu * nB[ij])); //
                        }
                    }
                    sprintf(filename, "%s_%d_smalldt_%d.deng", run, n, itheta);
                    //		      if(myid==0) {printf("output n, elapsed time = %i %e \n",n,cpuTime);}
                    MPI_Barrier(MPI_COMM_WORLD);
                    for (pc = 0; pc < np; pc++)
                    {
                        MPI_Barrier(MPI_COMM_WORLD);
                        if (myid == pc)
                        {
                            if (myid == 0)
                            {
                                fout = fopen(filename, "w");
                            }
                            else
                            {
                                fout = fopen(filename, "a");
                            }
                            for (i = 0; i < Lxl; ++i)
                            {
                                ig = i + myid * Lxl;
                                for (j = 0; j < Ly; ++j)
                                {
                                    ij = i * Ly + j;
                                    A1sq = A1[ij] * conj(A1[ij]);
                                    A2sq = A2[ij] * conj(A2[ij]);
                                    A3sq = A3[ij] * conj(A3[ij]);
                                    B1sq = B1[ij] * conj(B1[ij]);
                                    B2sq = B2[ij] * conj(B2[ij]);
                                    B3sq = B3[ij] * conj(B3[ij]);
                                    Ssq = 2.0 * (A1sq + A2sq + A3sq);
                                    Tsq = 2.0 * (B1sq + B2sq + B3sq);
                                    fprintf(fout, "%i %i %22.14e %e %e %e \n", ig, j, engij[ij], Tsq + Ssq, sqrt(Tsq / 6.0), sqrt(Ssq / 6.0));
                                }
                            }
                            fclose(fout);
                        }
                        MPI_Barrier(MPI_COMM_WORLD);
                    }
                }
                if (n % neng_smalldt == 0)
                {
                    char filename[BUFSIZ];
                    FILE *fout;
                    enl = 0.0;
                    // calculate gradient energy
                    for (i = 0; i < Lxl; ++i)
                    {
                        ig = i + myid * Lxl;
                        if (ig < Lx / 2)
                        {
                            kx = ig * fx;
                        }
                        else
                        {
                            kx = (ig - Lx) * fx;
                        }
                        for (j = 0; j < Ly; ++j)
                        {
                            ij = i * Ly + j;
                            if (j < Ly / 2)
                            {
                                ky = j * fy;
                            }
                            else
                            {
                                ky = (j - Ly) * fy;
                            }

                            kA1f = -(kx * kx + ky * ky) - 2.0 * (G1x * kx + G1y * ky) * alpha + 1.0 - alpha * alpha;
                            kB1f = kA1f;
                            // kB1f=-(kx*kx+ky*ky)-2.0*(G1x*kx+G1y*ky)*alpha+1.0-alpha*alpha;

                            kA2f = -(kx * kx + ky * ky) - 2.0 * (G2x * kx + G2y * ky) * alpha + 1.0 - alpha * alpha;
                            kB2f = kA2f;
                            // kB2f=-(kx*kx+ky*ky)-2.0*(G2x*kx+G2y*ky)*alpha+1.0-alpha*alpha;

                            kA3f = -(kx * kx + ky * ky) - 2.0 * (G3x * kx + G3y * ky) * alpha + 1.0 - alpha * alpha;
                            kB3f = kA3f;
                            // kB3f=-(kx*kx+ky*ky)-2.0*(G3x*kx+G3y*ky)*alpha+1.0-alpha*alpha;

                            enl = enl + kA1f * kA1f * Ak1[ij] * conj(Ak1[ij]) + kA2f * kA2f * Ak2[ij] * conj(Ak2[ij]) + kA3f * kA3f * Ak3[ij] * conj(Ak3[ij]) + betaB * (kB1f * kB1f * Bk1[ij] * conj(Bk1[ij]) + kB2f * kB2f * Bk2[ij] * conj(Bk2[ij]) + kB3f * kB3f * Bk3[ij] * conj(Bk3[ij]));
                        }
                    }
                    enl = enl * Lx * Ly; // enl=0.0;
                    // alternate (slower) method to calculate gradient energy
                    /*                 	for(i=0;i<Lxl;++i){ig=i+myid*Lxl;
                                    if(ig< Lx/2){kx=ig*fx;} else {kx=(ig-Lx)*fx;}
                                        for(j=0;j< Ly;++j){ij=i*Ly+j;
                                    if(j< Ly/2){ky=j*fy;} else {ky=(j-Ly)*fy;}

                                kA1f=-(kx*kx+ky*ky)-2.0*(G1x*kx+G1y*ky)*alpha+1.0-alpha*alpha; kB1f=kA1f;
                                //kB1f=-(kx*kx+ky*ky)-2.0*(G1x*kx+G1y*ky)*alpha+1.0-alpha*alpha;

                                kA2f=-(kx*kx+ky*ky)-2.0*(G2x*kx+G2y*ky)*alpha+1.0-alpha*alpha; kB2f=kA2f;
                                //kB2f=-(kx*kx+ky*ky)-2.0*(G2x*kx+G2y*ky)*alpha+1.0-alpha*alpha;

                                kA3f=-(kx*kx+ky*ky)-2.0*(G3x*kx+G3y*ky)*alpha+1.0-alpha*alpha; kB3f=kA3f;
                                //kB3f=-(kx*kx+ky*ky)-2.0*(G3x*kx+G3y*ky)*alpha+1.0-alpha*alpha;
                                //
                                An1[ij]=A1[ij]; An2[ij]=A2[ij]; An3[ij]=A3[ij];
                                Bn1[ij]=B1[ij]; Bn2[ij]=B2[ij]; Bn3[ij]=B3[ij];
                                Ak1[ij]=kA1f*Ak1[ij]; Ak2[ij]=kA2f*Ak2[ij]; Ak3[ij]=kA3f*Ak3[ij];
                                Bk1[ij]=kB1f*Bk1[ij]; Bk2[ij]=kB2f*Bk2[ij]; Bk3[ij]=kB3f*Bk3[ij];
                                    }}
                                    fftw_execute(plan1Ab);  fftw_execute(plan2Ab);  fftw_execute(plan3Ab);
                                    fftw_execute(plan1Bb);  fftw_execute(plan2Bb);  fftw_execute(plan3Bb);
                                    for(i=0;i<Lxl;++i){ for(j=0;j< Ly;++j){ij=i*Ly+j;
                                enl=enl+A1[ij]*conj(A1[ij])+A2[ij]*conj(A2[ij])+A3[ij]*conj(A3[ij]);
                                enl=enl+B1[ij]*conj(B1[ij])+B2[ij]*conj(B2[ij])+B3[ij]*conj(B3[ij]);
                            }}
                                    for(i=0;i<Lxl;++i){ for(j=0;j< Ly;++j){ij=i*Ly+j;
                                A1[ij]=An1[ij]; A2[ij]=An2[ij]; A3[ij]=An3[ij];
                                B1[ij]=Bn1[ij]; B2[ij]=Bn2[ij]; B3[ij]=Bn3[ij];
                                An1[ij]=A1[ij]; An2[ij]=A2[ij]; An3[ij]=A3[ij];
                                Bn1[ij]=B1[ij]; Bn2[ij]=B2[ij]; Bn3[ij]=B3[ij];
                            }}
                                    fftw_execute(plan1An);  fftw_execute(plan2An);  fftw_execute(plan3An);
                                    fftw_execute(plan1Bn);  fftw_execute(plan2Bn);  fftw_execute(plan3Bn);
                                    for(i=0;i<Lxl;++i){ for(j=0;j< Ly;++j){ij=i*Ly+j;
                                Ak1[ij]=An1[ij]*Sf; Ak2[ij]=An2[ij]*Sf; Ak3[ij]=An3[ij]*Sf;
                                Bk1[ij]=Bn1[ij]*Sf; Bk2[ij]=Bn2[ij]*Sf; Bk3[ij]=Bn3[ij]*Sf;
                            }} */
                    // printf(" grad energy = %e \n",enl);
                    for (i = 0; i < Lxl; ++i)
                    {
                        for (j = 0; j < Ly; ++j)
                        {
                            ij = i * Ly + j;
                            // Eq. 18  check 08/05/23
                            Ssq = A1[ij] * conj(A1[ij]) + A2[ij] * conj(A2[ij]) + A3[ij] * conj(A3[ij]);
                            Tsq = B1[ij] * conj(B1[ij]) + B2[ij] * conj(B2[ij]) + B3[ij] * conj(B3[ij]);
                            gnAo = 2.0 * (3.0 * nA[ij] * vA - gA);
                            gnBo = 2.0 * (3.0 * nB[ij] * vB - gB);
                            aonm = alphaAB + omega * nA[ij] + mu * nB[ij];
                            Asqcoeff = -epsA + nA[ij] * (3.0 * vA * nA[ij] - 2.0 * gA) + omega * nB[ij];
                            Bsqcoeff = -epsB + nB[ij] * (3.0 * vB * nB[ij] - 2.0 * gB) + mu * nA[ij];
                            enl = enl + Asqcoeff * Ssq + Bsqcoeff * Tsq + gnAo * (A1[ij] * A2[ij] * A3[ij] + conj(A1[ij] * A2[ij] * A3[ij])) + gnBo * (B1[ij] * B2[ij] * B3[ij] + conj(B1[ij] * B2[ij] * B3[ij])) + vA3 * Ssq * Ssq + vB3 * Tsq * Tsq - 1.5 * vA * (A1[ij] * conj(A1[ij]) * A1[ij] * conj(A1[ij]) + A2[ij] * conj(A2[ij]) * A2[ij] * conj(A2[ij]) + A3[ij] * conj(A3[ij]) * A3[ij] * conj(A3[ij])) - 1.5 * vB * (B1[ij] * conj(B1[ij]) * B1[ij] * conj(B1[ij]) + B2[ij] * conj(B2[ij]) * B2[ij] * conj(B2[ij]) + B3[ij] * conj(B3[ij]) * B3[ij] * conj(B3[ij])) + aonm * (A1[ij] * conj(B1[ij]) + B1[ij] * conj(A1[ij]) + A2[ij] * conj(B2[ij]) + B2[ij] * conj(A2[ij]) + A3[ij] * conj(B3[ij]) + B3[ij] * conj(A3[ij])) + omega * (A1[ij] * A2[ij] * B3[ij] + conj(A1[ij] * A2[ij] * B3[ij]) + A1[ij] * A3[ij] * B2[ij] + conj(A1[ij] * A3[ij] * B2[ij]) + A2[ij] * A3[ij] * B1[ij] + conj(A2[ij] * A3[ij] * B1[ij])) + mu * (A1[ij] * B2[ij] * B3[ij] + conj(A1[ij] * B2[ij] * B3[ij]) + A2[ij] * B1[ij] * B3[ij] + conj(A2[ij] * B1[ij] * B3[ij]) + A3[ij] * B1[ij] * B2[ij] + conj(A3[ij] * B1[ij] * B2[ij])) + nA[ij] * nA[ij] * (0.5 * (-epsA + 1.0) + nA[ij] * (-gA / 3. + 0.25 * vA * nA[ij])) + nB[ij] * nB[ij] * (0.5 * (-epsB + betaB) + nB[ij] * (-gB / 3. + 0.25 * vB * nB[ij])) + nA[ij] * nB[ij] * (alphaAB + 0.5 * (omega * nA[ij] + mu * nB[ij])); //
                        }
                    }
                    if (myid == 0)
                    {
                        eng = enl; // printf("np, enl = %i %e \n",np,enl);
                        for (p2 = 1; p2 < np; p2++)
                        {
                            MPI_Recv(&enl, 1, MPI_DOUBLE, p2, 0, MPI_COMM_WORLD, &status);
                            eng = eng + enl;
                            // printf(" p2 enl  = %i %e %e\n",p2,enl,eng);
                        }
                        sprintf(filename, "%s_smalldt_%d.eng", run, itheta);
                        if (ich == 0)
                        {
                            fout = fopen(filename, "w");
                        }
                        else
                        {
                            fout = fopen(filename, "a");
                        }
                        ich = 1;
                        gamma = ffac * (2 * twopi / sqrt(3)) / (2.51) * (eng * Sf - fn) * Lx * dx;
                        fprintf(fout, "%e %22.14e %22.14e \n", n * dt, eng * Sf, gamma);
                        printf("smalldt: %e %22.14e %22.14e \n", n * dt, eng * Sf, gamma);
                        fclose(fout);
                    }
                    else
                    {
                        MPI_Send(&enl, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                    }
                }
            }
            dt /= dtS;
        }
        for (i = 0; i < Lxl; ++i)
        {
            ig = i + myid * Lxl;
            if (ig < Lx / 2)
            {
                kx = ig * fx;
            }
            else
            {
                kx = (ig - Lx) * fx;
            }
            for (j = 0; j < Ly; ++j)
            {
                ij = i * Ly + j;
                if (j < Ly / 2)
                {
                    ky = j * fy;
                }
                else
                {
                    ky = (j - Ly) * fy;
                }
                ksq = kx * kx + ky * ky;
                kA1f = -1.0 * (-epsA + (-ksq - 2.0 * alpha * (G1x * kx + G1y * ky) + 1. - alpha * alpha) * (-ksq - 2.0 * alpha * (G1x * kx + G1y * ky) + 1. - alpha * alpha));
                kA2f = -1.0 * (-epsA + (-ksq - 2.0 * alpha * (G2x * kx + G2y * ky) + 1. - alpha * alpha) * (-ksq - 2.0 * alpha * (G2x * kx + G2y * ky) + 1. - alpha * alpha));
                kA3f = -1.0 * (-epsA + (-ksq - 2.0 * alpha * (G3x * kx + G3y * ky) + 1. - alpha * alpha) * (-ksq - 2.0 * alpha * (G3x * kx + G3y * ky) + 1. - alpha * alpha));
                kB1f = -1.0 * (-epsB + betaB * (-ksq - 2.0 * alpha * (G1x * kx + G1y * ky) + 1. - alpha * alpha) * (-ksq - 2.0 * alpha * (G1x * kx + G1y * ky) + 1. - alpha * alpha));
                kB2f = -1.0 * (-epsB + betaB * (-ksq - 2.0 * alpha * (G2x * kx + G2y * ky) + 1. - alpha * alpha) * (-ksq - 2.0 * alpha * (G2x * kx + G2y * ky) + 1. - alpha * alpha));
                kB3f = -1.0 * (-epsB + betaB * (-ksq - 2.0 * alpha * (G3x * kx + G3y * ky) + 1. - alpha * alpha) * (-ksq - 2.0 * alpha * (G3x * kx + G3y * ky) + 1. - alpha * alpha));
                kA1l[ij] = exp(kA1f * dt);
                kA2l[ij] = exp(kA2f * dt);
                kA3l[ij] = exp(kA3f * dt);
                kB1l[ij] = exp(kB1f * dt);
                kB2l[ij] = exp(kB2f * dt);
                kB3l[ij] = exp(kB3f * dt);
                if (kA1f == 0)
                {
                    kA1n[ij] = -dt * Sf;
                }
                else
                {
                    kA1n[ij] = ((1.0 - exp(kA1f * dt)) / kA1f) * Sf;
                }
                if (kA2f == 0)
                {
                    kA2n[ij] = -dt * Sf;
                }
                else
                {
                    kA2n[ij] = ((1.0 - exp(kA2f * dt)) / kA2f) * Sf;
                }
                if (kA3f == 0)
                {
                    kA3n[ij] = -dt * Sf;
                }
                else
                {
                    kA3n[ij] = ((1.0 - exp(kA3f * dt)) / kA3f) * Sf;
                }
                if (kB1f == 0)
                {
                    kB1n[ij] = -dt * Sf;
                }
                else
                {
                    kB1n[ij] = ((1.0 - exp(kB1f * dt)) / kB1f) * Sf;
                }
                if (kB2f == 0)
                {
                    kB2n[ij] = -dt * Sf;
                }
                else
                {
                    kB2n[ij] = ((1.0 - exp(kB2f * dt)) / kB2f) * Sf;
                }
                if (kB3f == 0)
                {
                    kB3n[ij] = -dt * Sf;
                }
                else
                {
                    kB3n[ij] = ((1.0 - exp(kB3f * dt)) / kB3f) * Sf;
                }
            }
        }
        kA3f = -epsA + 1.0;
        kB3f = -epsB + betaB;
        for (i = 0; i < Lxl; ++i)
        {
            ig = i + myid * Lxl;
            if (ig < Lx / 2)
            {
                kx = ig * fx;
            }
            else
            {
                kx = (ig - Lx) * fx;
            }
            for (j = 0; j < Ly; ++j)
            {
                ij = i * Ly + j;
                if (j < Ly / 2)
                {
                    ky = j * fy;
                }
                else
                {
                    ky = (j - Ly) * fy;
                }
                ksq = kx * kx + ky * ky;
                kAl[ij] = exp(-ksq * kA3f * dt);
                if (ksq == 0)
                {
                    kAn[ij] = 0;
                }
                else
                {
                    kAn[ij] = (exp(-ksq * kA3f * dt) - 1.0) / kA3f * Sf;
                }
                kBl[ij] = exp(-ksq * kB3f * dt);
                if (ksq == 0)
                {
                    kBn[ij] = 0;
                }
                else
                {
                    kBn[ij] = (exp(-ksq * kB3f * dt) - 1.0) / kB3f * Sf;
                }
            }
        }

        // ouptut initial fields
        {
            char filename[BUFSIZ], fileB[BUFSIZ];
            FILE *fout, *foutB;
            sprintf(filename, "%s_init_%d.Adat", run, itheta);
            sprintf(fileB, "%s_init_%d.Bdat", run, itheta);
            //		      if(myid==0) {printf("output n, elapsed time = %i %e \n",n,cpuTime);}
            MPI_Barrier(MPI_COMM_WORLD);
            for (pc = 0; pc < np; pc++)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                if (myid == pc)
                {
                    if (myid == 0)
                    {
                        fout = fopen(filename, "w");
                        foutB = fopen(fileB, "w");
                    }
                    else
                    {
                        fout = fopen(filename, "a");
                        foutB = fopen(fileB, "a");
                    }
                    for (i = 0; i < Lxl; ++i)
                    {
                        ig = i + myid * Lxl;
                        for (j = 0; j < Ly; ++j)
                        {
                            ij = i * Ly + j;
                            A1sq = A1[ij] * conj(A1[ij]);
                            A2sq = A2[ij] * conj(A2[ij]);
                            A3sq = A3[ij] * conj(A3[ij]);
                            B1sq = B1[ij] * conj(B1[ij]);
                            B2sq = B2[ij] * conj(B2[ij]);
                            B3sq = B3[ij] * conj(B3[ij]);
                            Ssq = 2.0 * (A1sq + A2sq + A3sq);
                            Tsq = 2.0 * (B1sq + B2sq + B3sq);
                            fprintf(fout, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nA[ij], A1[ij], A2[ij], A3[ij], Ssq);
                            fprintf(foutB, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nB[ij], B1[ij], B2[ij], B3[ij], Tsq);
                        }
                    }
                    fclose(fout);
                    fclose(foutB);
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }
        { // every nout time steps output data
            char filename[BUFSIZ], fileB[BUFSIZ];
            FILE *fout, *foutB;
            sprintf(filename, "%s_init_%d.Andat", run, itheta);
            sprintf(fileB, "%s_init_%d.Bndat", run, itheta);
            //		      if(myid==0) {printf("output n, elapsed time = %i %e \n",n,cpuTime);}
            MPI_Barrier(MPI_COMM_WORLD);
            for (pc = 0; pc < np; pc++)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                if (myid == pc)
                {
                    if (myid == 0)
                    {
                        fout = fopen(filename, "w");
                        foutB = fopen(fileB, "w");
                    }
                    else
                    {
                        fout = fopen(filename, "a");
                        foutB = fopen(fileB, "a");
                    }
                    for (i = 0; i < Lxl; ++i)
                    {
                        ig = i + myid * Lxl;
                        for (j = 0; j < Ly; ++j)
                        {
                            ij = i * Ly + j;
                            A1sq = A1[ij] * conj(A1[ij]);
                            A2sq = A2[ij] * conj(A2[ij]);
                            A3sq = A3[ij] * conj(A3[ij]);
                            B1sq = B1[ij] * conj(B1[ij]);
                            B2sq = B2[ij] * conj(B2[ij]);
                            B3sq = B3[ij] * conj(B3[ij]);
                            Ssq = 2.0 * (A1sq + A2sq + A3sq);
                            Tsq = 2.0 * (B1sq + B2sq + B3sq);
                            fprintf(fout, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nAn[ij], An1[ij], An2[ij], An3[ij], Ssq);
                            fprintf(foutB, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nBn[ij], Bn1[ij], Bn2[ij], Bn3[ij], Tsq);
                        }
                    }
                    fclose(fout);
                    fclose(foutB);
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
            start = clock();
        }
        { // every nout time steps output data
            char filename[BUFSIZ], fileB[BUFSIZ];
            FILE *fout, *foutB;
            sprintf(filename, "%s_init_%d.Akdat", run, itheta);
            sprintf(fileB, "%s_init_%d.Bkdat", run, itheta);
            //		      if(myid==0) {printf("output n, elapsed time = %i %e \n",n,cpuTime);}
            MPI_Barrier(MPI_COMM_WORLD);
            for (pc = 0; pc < np; pc++)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                if (myid == pc)
                {
                    if (myid == 0)
                    {
                        fout = fopen(filename, "w");
                        foutB = fopen(fileB, "w");
                    }
                    else
                    {
                        fout = fopen(filename, "a");
                        foutB = fopen(fileB, "a");
                    }
                    for (i = 0; i < Lxl; ++i)
                    {
                        ig = i + myid * Lxl;
                        for (j = 0; j < Ly; ++j)
                        {
                            ij = i * Ly + j;
                            A1sq = A1[ij] * conj(A1[ij]);
                            A2sq = A2[ij] * conj(A2[ij]);
                            A3sq = A3[ij] * conj(A3[ij]);
                            B1sq = B1[ij] * conj(B1[ij]);
                            B2sq = B2[ij] * conj(B2[ij]);
                            B3sq = B3[ij] * conj(B3[ij]);
                            Ssq = 2.0 * (A1sq + A2sq + A3sq);
                            Tsq = 2.0 * (B1sq + B2sq + B3sq);
                            fprintf(fout, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nAk[ij], Ak1[ij], Ak2[ij], Ak3[ij], Ssq);
                            fprintf(foutB, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nBk[ij], Bk1[ij], Bk2[ij], Bk3[ij], Tsq);
                        }
                    }
                    fclose(fout);
                    fclose(foutB);
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
            start = clock();
        }
 

        vA3 = 3.0 * vA;
        vB3 = 3.0 * vB;
        start = clock();
        muA = 0.0;
        muB = 0.0;
        for (n = 0; n < nend + 1; ++n)
        {
            for (i = 0; i < Lxl; ++i)
            {
                for (j = 0; j < Ly; ++j)
                {
                    ij = i * Ly + j;
                    A1sq = A1[ij] * conj(A1[ij]);
                    A2sq = A2[ij] * conj(A2[ij]);
                    A3sq = A3[ij] * conj(A3[ij]);
                    B1sq = B1[ij] * conj(B1[ij]);
                    B2sq = B2[ij] * conj(B2[ij]);
                    B3sq = B3[ij] * conj(B3[ij]);
                    Ssq = 2.0 * (A1sq + A2sq + A3sq);
                    Tsq = 2.0 * (B1sq + B2sq + B3sq);
                    gnAo = 2.0 * (3.0 * vA * nA[ij] - gA); // gnAo = 2.0 * (3.0 * nA[ij] - gA);
                    gnBo = 2.0 * (3.0 * vA * nB[ij] - gB); // gnBo = 2.0 * (3.0 * nB[ij] - gB);
                    aonm = alphaAB + omega * nA[ij] + mu * nB[ij];
                    naf = nA[ij] * (vA3 * nA[ij] - 2 * gA) + omega * nB[ij];
                    nbf = nB[ij] * (vB3 * nB[ij] - 2 * gB) + mu * nA[ij];

                    // Eq. 19 check 08/05/23
                    An1[ij] = (vA3 * (Ssq - A1sq) + naf) * A1[ij] + gnAo * conj(A2[ij] * A3[ij]) + aonm * B1[ij] + omega * (conj(A2[ij] * B3[ij]) + conj(A3[ij] * B2[ij])) + mu * conj(B2[ij] * B3[ij]) + Vo; //+VA1[ij];

                    An2[ij] = (vA3 * (Ssq - A2sq) + naf) * A2[ij] + gnAo * conj(A1[ij] * A3[ij]) + aonm * B2[ij] + omega * (conj(A1[ij] * B3[ij]) + conj(A3[ij] * B1[ij])) + mu * conj(B1[ij] * B3[ij]) + Vo; //+VA2[ij];

                    An3[ij] = (vA3 * (Ssq - A3sq) + naf) * A3[ij] + gnAo * conj(A1[ij] * A2[ij]) + aonm * B3[ij] + omega * (conj(A1[ij] * B2[ij]) + conj(A2[ij] * B1[ij])) + mu * conj(B1[ij] * B2[ij]) + Vo; //+VA3[ij];

                    // Eq. 21 check 08/05/23
                    Bn1[ij] = (vB3 * (Tsq - B1sq) + nbf) * B1[ij] + gnBo * conj(B2[ij] * B3[ij]) + aonm * A1[ij] + mu * (conj(A2[ij] * B3[ij]) + conj(A3[ij] * B2[ij])) + omega * (conj(A2[ij] * A3[ij])) + Vo; //+VB1[ij];

                    Bn2[ij] = (vB3 * (Tsq - B2sq) + nbf) * B2[ij] + gnBo * conj(B1[ij] * B3[ij]) + aonm * A2[ij] + mu * (conj(A1[ij] * B3[ij]) + conj(A3[ij] * B1[ij])) + omega * conj(A1[ij] * A3[ij]) + Vo; //+VB2[ij];

                    Bn3[ij] = (vB3 * (Tsq - B3sq) + nbf) * B3[ij] + gnBo * conj(B1[ij] * B2[ij]) + aonm * A3[ij] + mu * (conj(A1[ij] * B2[ij]) + conj(A2[ij] * B1[ij])) + omega * conj(A1[ij] * A2[ij]) + Vo; //+VB3[ij];

                    // Eq. 22
                    nAn[ij] = nA[ij] * nA[ij] * (-gA + vA * nA[ij]) + 0.5 * (gnAo * Ssq + mu * Tsq) + 6.0 * vA * (A1[ij] * A2[ij] * A3[ij] + conj(A1[ij] * A2[ij] * A3[ij])) + omega * (A1[ij] * conj(B1[ij]) + A2[ij] * conj(B2[ij]) + A3[ij] * conj(B3[ij]) + B1[ij] * conj(A1[ij]) + B2[ij] * conj(A2[ij]) + B3[ij] * conj(A3[ij])) + nB[ij] * (alphaAB + omega * nA[ij] + 0.5 * mu * nB[ij]);
                    // Eq. 24
                    nBn[ij] = nB[ij] * nB[ij] * (-gB + vB * nB[ij]) + 0.5 * (gnBo * Tsq + omega * Ssq) + 6.0 * vB * (B1[ij] * B2[ij] * B3[ij] + conj(B1[ij] * B2[ij] * B3[ij])) + mu * (A1[ij] * conj(B1[ij]) + A2[ij] * conj(B2[ij]) + A3[ij] * conj(B3[ij]) + B1[ij] * conj(A1[ij]) + B2[ij] * conj(A2[ij]) + B3[ij] * conj(A3[ij])) + nA[ij] * (alphaAB + mu * nB[ij] + 0.5 * omega * nA[ij]);
                    // printf("%i %i %e %e %e %e\n",i,j,nA[ij],nB[ij],kAl,kAn);
                    // printf("%i %i %e %e \n",i,j,nA[ij],nB[ij]);
                    // printf("%i %i %e %e %e %e\n",i,j,nA[ij],nB[ij],kAl,kAn);
                    // printf("%i %i %e %e \n",i,j,nA[ij],nB[ij]);
                }
            }
            //        for(i=0;i<Lxl;++i){for(j=0;j< Ly;++j){ij=i*Ly+j;
            //		printf("%i %i %e %e %e %e \n",i,j,An1[ij],Bn1[ij]);
            //	     }}
            fftw_execute(plannAn);
            fftw_execute(plannBn);
            fftw_execute(plan1An);
            fftw_execute(plan2An);
            fftw_execute(plan3An);
            fftw_execute(plan1Bn);
            fftw_execute(plan2Bn);
            fftw_execute(plan3Bn);
            for (i = 0; i < Lxl; ++i)
            {
                for (j = 0; j < Ly; ++j)
                {
                    ij = i * Ly + j;
                    Ak1[ij] = kA1l[ij] * Ak1[ij] + kA1n[ij] * An1[ij];
                    Ak2[ij] = kA2l[ij] * Ak2[ij] + kA2n[ij] * An2[ij];
                    Ak3[ij] = kA3l[ij] * Ak3[ij] + kA3n[ij] * An3[ij];
                    Bk1[ij] = kB1l[ij] * Bk1[ij] + kB1n[ij] * Bn1[ij];
                    Bk2[ij] = kB2l[ij] * Bk2[ij] + kB2n[ij] * Bn2[ij];
                    Bk3[ij] = kB3l[ij] * Bk3[ij] + kB3n[ij] * Bn3[ij];
                    nAk[ij] = kAl[ij] * nAk[ij] + kAn[ij] * nAn[ij];
                    nBk[ij] = kBl[ij] * nBk[ij] + kBn[ij] * nBn[ij];
                }
            }
            fftw_execute(plannAb);
            fftw_execute(plannBb);
            fftw_execute(plan1Ab);
            fftw_execute(plan2Ab);
            fftw_execute(plan3Ab);
            fftw_execute(plan1Bb);
            fftw_execute(plan2Bb);
            fftw_execute(plan3Bb);
            //         for(i=0;i<Lxl;++i){for(j=0;j< Ly;++j){ij=i*Ly+j;
            //		printf("A %i %i %e %e %e %e \n",i,j,An1[ij],Bn1[ij]);
            //	     }}
            /*         for(i=0;i<Lxl;++i){for(j=0;j< Ly;++j){ij=i*Ly+j;
                         An1[ij]=A1[ij]; An2[ij]=A2[ij]; An3[ij]=A3[ij];
                         Bn1[ij]=B1[ij]; Bn2[ij]=B2[ij]; Bn3[ij]=B3[ij];
                     }}
                 fftw_execute(plannA); fftw_execute(plannB);
                 fftw_execute(plan1An); fftw_execute(plan2An); fftw_execute(plan3An);
                 fftw_execute(plan1Bn); fftw_execute(plan2Bn); fftw_execute(plan3Bn);
                     for(i=0;i<Lxl;++i){for(j=0;j< Ly;++j){ij=i*Ly+j;
                        Ak1[ij]=An1[ij]*Sf; Ak2[ij]=An2[ij]*Sf; Ak3[ij]=An3[ij]*Sf;
                        Bk1[ij]=Bn1[ij]*Sf; Bk2[ij]=Bn2[ij]*Sf; Bk3[ij]=Bn3[ij]*Sf;
                        nAk[ij]=nAk[ij]*Sf; nBk[ij]=nBk[ij]*Sf;
                 }}  */

            if (n % nout == 0) // if (n % nout == 0)
            { // every nout time steps output data
                cpuTime = (clock() - start) / (CLOCKS_PER_SEC);
                char filename[BUFSIZ], fileB[BUFSIZ];
                FILE *fout, *foutB;
                sprintf(filename, "%s_%d_%d.Adat", run, n, itheta);
                sprintf(fileB, "%s_%d_%d.Bdat", run, n, itheta);
                //		      if(myid==0) {printf("output n, elapsed time = %i %e \n",n,cpuTime);}
                MPI_Barrier(MPI_COMM_WORLD);
                for (pc = 0; pc < np; pc++)
                {
                    MPI_Barrier(MPI_COMM_WORLD);
                    if (myid == pc)
                    {
                        if (myid == 0)
                        {
                            fout = fopen(filename, "w");
                            foutB = fopen(fileB, "w");
                        }
                        else
                        {
                            fout = fopen(filename, "a");
                            foutB = fopen(fileB, "a");
                        }
                        for (i = 0; i < Lxl; ++i)
                        {
                            ig = i + myid * Lxl;
                            for (j = 0; j < Ly; ++j)
                            {
                                ij = i * Ly + j;
                                A1sq = A1[ij] * conj(A1[ij]);
                                A2sq = A2[ij] * conj(A2[ij]);
                                A3sq = A3[ij] * conj(A3[ij]);
                                B1sq = B1[ij] * conj(B1[ij]);
                                B2sq = B2[ij] * conj(B2[ij]);
                                B3sq = B3[ij] * conj(B3[ij]);
                                Ssq = 2.0 * (A1sq + A2sq + A3sq);
                                Tsq = 2.0 * (B1sq + B2sq + B3sq);
                                fprintf(fout, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nA[ij], A1[ij], A2[ij], A3[ij], Ssq);
                                fprintf(foutB, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nB[ij], B1[ij], B2[ij], B3[ij], Tsq);
                            }
                        }
                        fclose(fout);
                        fclose(foutB);
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                }
                start = clock();
            }
            // if (n % nout == 0) // if (n % nout == 0)
            // { // every nout time steps output data
            //     cpuTime = (clock() - start) / (CLOCKS_PER_SEC);
            //     char filename[BUFSIZ], fileB[BUFSIZ];
            //     FILE *fout, *foutB;
            //     sprintf(filename, "%s_%d_%d.Andat", run, n, itheta);
            //     sprintf(fileB, "%s_%d_%d.Bndat", run, n, itheta);
            //     //		      if(myid==0) {printf("output n, elapsed time = %i %e \n",n,cpuTime);}
            //     MPI_Barrier(MPI_COMM_WORLD);
            //     for (pc = 0; pc < np; pc++)
            //     {
            //         MPI_Barrier(MPI_COMM_WORLD);
            //         if (myid == pc)
            //         {
            //             if (myid == 0)
            //             {
            //                 fout = fopen(filename, "w");
            //                 foutB = fopen(fileB, "w");
            //             }
            //             else
            //             {
            //                 fout = fopen(filename, "a");
            //                 foutB = fopen(fileB, "a");
            //             }
            //             for (i = 0; i < Lxl; ++i)
            //             {
            //                 ig = i + myid * Lxl;
            //                 for (j = 0; j < Ly; ++j)
            //                 {
            //                     ij = i * Ly + j;
            //                     A1sq = A1[ij] * conj(A1[ij]);
            //                     A2sq = A2[ij] * conj(A2[ij]);
            //                     A3sq = A3[ij] * conj(A3[ij]);
            //                     B1sq = B1[ij] * conj(B1[ij]);
            //                     B2sq = B2[ij] * conj(B2[ij]);
            //                     B3sq = B3[ij] * conj(B3[ij]);
            //                     Ssq = 2.0 * (A1sq + A2sq + A3sq);
            //                     Tsq = 2.0 * (B1sq + B2sq + B3sq);
            //                     fprintf(fout, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nAn[ij], An1[ij], An2[ij], An3[ij], Ssq);
            //                     fprintf(foutB, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nBn[ij], Bn1[ij], Bn2[ij], Bn3[ij], Tsq);
            //                 }
            //             }
            //             fclose(fout);
            //             fclose(foutB);
            //         }
            //         MPI_Barrier(MPI_COMM_WORLD);
            //     }
            //     start = clock();
            // }
            // if (n % nout == 0) // if (n % nout == 0)
            // { // every nout time steps output data
            //     cpuTime = (clock() - start) / (CLOCKS_PER_SEC);
            //     char filename[BUFSIZ], fileB[BUFSIZ];
            //     FILE *fout, *foutB;
            //     sprintf(filename, "%s_%d_%d.Akdat", run, n, itheta);
            //     sprintf(fileB, "%s_%d_%d.Bkdat", run, n, itheta);
            //     //		      if(myid==0) {printf("output n, elapsed time = %i %e \n",n,cpuTime);}
            //     MPI_Barrier(MPI_COMM_WORLD);
            //     for (pc = 0; pc < np; pc++)
            //     {
            //         MPI_Barrier(MPI_COMM_WORLD);
            //         if (myid == pc)
            //         {
            //             if (myid == 0)
            //             {
            //                 fout = fopen(filename, "w");
            //                 foutB = fopen(fileB, "w");
            //             }
            //             else
            //             {
            //                 fout = fopen(filename, "a");
            //                 foutB = fopen(fileB, "a");
            //             }
            //             for (i = 0; i < Lxl; ++i)
            //             {
            //                 ig = i + myid * Lxl;
            //                 for (j = 0; j < Ly; ++j)
            //                 {
            //                     ij = i * Ly + j;
            //                     A1sq = A1[ij] * conj(A1[ij]);
            //                     A2sq = A2[ij] * conj(A2[ij]);
            //                     A3sq = A3[ij] * conj(A3[ij]);
            //                     B1sq = B1[ij] * conj(B1[ij]);
            //                     B2sq = B2[ij] * conj(B2[ij]);
            //                     B3sq = B3[ij] * conj(B3[ij]);
            //                     Ssq = 2.0 * (A1sq + A2sq + A3sq);
            //                     Tsq = 2.0 * (B1sq + B2sq + B3sq);
            //                     fprintf(fout, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nAk[ij], Ak1[ij], Ak2[ij], Ak3[ij], Ssq);
            //                     fprintf(foutB, "%i %i %e %e %e %e %e %e %e %e %e\n", ig, j, nBk[ij], Bk1[ij], Bk2[ij], Bk3[ij], Tsq);
            //                 }
            //             }
            //             fclose(fout);
            //             fclose(foutB);
            //         }
            //         MPI_Barrier(MPI_COMM_WORLD);
            //     }
            //     start = clock();
            // }
            if (n % neng2 == 0) // if (n % neng2 == 0)
            {
                char filename[BUFSIZ];
                FILE *fout;
                enl = 0.0;
                // alternate (slower) method to calculate gradient energy
                for (i = 0; i < Lxl; ++i)
                {
                    ig = i + myid * Lxl;
                    if (ig < Lx / 2)
                    {
                        kx = ig * fx;
                    }
                    else
                    {
                        kx = (ig - Lx) * fx;
                    }
                    for (j = 0; j < Ly; ++j)
                    {
                        ij = i * Ly + j;
                        if (j < Ly / 2)
                        {
                            ky = j * fy;
                        }
                        else
                        {
                            ky = (j - Ly) * fy;
                        }

                        kA1f = -(kx * kx + ky * ky) - 2.0 * (G1x * kx + G1y * ky) * alpha + 1.0 - alpha * alpha;
                        kB1f = kA1f;
                        // kB1f=-(kx*kx+ky*ky)-2.0*(G1x*kx+G1y*ky)*alpha+1.0-alpha*alpha;

                        kA2f = -(kx * kx + ky * ky) - 2.0 * (G2x * kx + G2y * ky) * alpha + 1.0 - alpha * alpha;
                        kB2f = kA2f;
                        // kB2f=-(kx*kx+ky*ky)-2.0*(G2x*kx+G2y*ky)*alpha+1.0-alpha*alpha;

                        kA3f = -(kx * kx + ky * ky) - 2.0 * (G3x * kx + G3y * ky) * alpha + 1.0 - alpha * alpha;
                        kB3f = kA3f;
                        // kB3f=-(kx*kx+ky*ky)-2.0*(G3x*kx+G3y*ky)*alpha+1.0-alpha*alpha;
                        //
                        An1[ij] = A1[ij];
                        An2[ij] = A2[ij];
                        An3[ij] = A3[ij];
                        Bn1[ij] = B1[ij];
                        Bn2[ij] = B2[ij];
                        Bn3[ij] = B3[ij];
                        Ak1[ij] = kA1f * Ak1[ij];
                        Ak2[ij] = kA2f * Ak2[ij];
                        Ak3[ij] = kA3f * Ak3[ij];
                        Bk1[ij] = kB1f * Bk1[ij];
                        Bk2[ij] = kB2f * Bk2[ij];
                        Bk3[ij] = kB3f * Bk3[ij];
                    }
                }
                fftw_execute(plan1Ab);
                fftw_execute(plan2Ab);
                fftw_execute(plan3Ab);
                fftw_execute(plan1Bb);
                fftw_execute(plan2Bb);
                fftw_execute(plan3Bb);
                for (i = 0; i < Lxl; ++i)
                {
                    for (j = 0; j < Ly; ++j)
                    {
                        ij = i * Ly + j;
                        engij[ij] = A1[ij] * conj(A1[ij]) + A2[ij] * conj(A2[ij]) + A3[ij] * conj(A3[ij]);
                        engij[ij] = engij[ij] + B1[ij] * conj(B1[ij]) + B2[ij] * conj(B2[ij]) + B3[ij] * conj(B3[ij]);
                    }
                }
                for (i = 0; i < Lxl; ++i)
                {
                    for (j = 0; j < Ly; ++j)
                    {
                        ij = i * Ly + j;
                        A1[ij] = An1[ij];
                        A2[ij] = An2[ij];
                        A3[ij] = An3[ij];
                        B1[ij] = Bn1[ij];
                        B2[ij] = Bn2[ij];
                        B3[ij] = Bn3[ij];
                        An1[ij] = A1[ij];
                        An2[ij] = A2[ij];
                        An3[ij] = A3[ij];
                        Bn1[ij] = B1[ij];
                        Bn2[ij] = B2[ij];
                        Bn3[ij] = B3[ij];
                    }
                }
                fftw_execute(plan1An);
                fftw_execute(plan2An);
                fftw_execute(plan3An);
                fftw_execute(plan1Bn);
                fftw_execute(plan2Bn);
                fftw_execute(plan3Bn);
                for (i = 0; i < Lxl; ++i)
                {
                    for (j = 0; j < Ly; ++j)
                    {
                        ij = i * Ly + j;
                        Ak1[ij] = An1[ij] * Sf;
                        Ak2[ij] = An2[ij] * Sf;
                        Ak3[ij] = An3[ij] * Sf;
                        Bk1[ij] = Bn1[ij] * Sf;
                        Bk2[ij] = Bn2[ij] * Sf;
                        Bk3[ij] = Bn3[ij] * Sf;
                    }
                }
                /*                 for(i=0;i<Lxl;++i){ig=i+myid*Lxl;
                                if(ig< Lx/2){kx=ig*fx;} else {kx=(ig-Lx)*fx;}
                                    for(j=0;j< Ly;++j){ij=i*Ly+j;
                                if(j< Ly/2){ky=j*fy;} else {ky=(j-Ly)*fy;}

                            kA1f=-(kx*kx+ky*ky)-2.0*(G1x*kx+G1y*ky)*alpha+1.0-alpha*alpha; kB1f=kA1f;

                            kA2f=-(kx*kx+ky*ky)-2.0*(G2x*kx+G2y*ky)*alpha+1.0-alpha*alpha; kB2f=kA2f;

                            kA3f=-(kx*kx+ky*ky)-2.0*(G3x*kx+G3y*ky)*alpha+1.0-alpha*alpha; kB3f=kA3f;

                                engij[ij]=kA1f*kA1f*Ak1[ij]*conj(Ak1[ij])+kA2f*kA2f*Ak2[ij]*conj(Ak2[ij])
                                       +kA3f*kA3f*Ak3[ij]*conj(Ak3[ij])
                                +betaB*(kB1f*kB1f*Bk1[ij]*conj(Bk1[ij])+kB2f*kB2f*Bk2[ij]*conj(Bk2[ij])
                                       +kB3f*kB3f*Bk3[ij]*conj(Bk3[ij]));
                            engij[ij]=engij[ij]*Lx*Ly;
                                 }}                 */
                for (i = 0; i < Lxl; ++i)
                {
                    for (j = 0; j < Ly; ++j)
                    {
                        ij = i * Ly + j;
                        // Eq. 18  check 08/05/23
                        Ssq = A1[ij] * conj(A1[ij]) + A2[ij] * conj(A2[ij]) + A3[ij] * conj(A3[ij]);
                        Tsq = B1[ij] * conj(B1[ij]) + B2[ij] * conj(B2[ij]) + B3[ij] * conj(B3[ij]);
                        gnAo = 2.0 * (3.0 * nA[ij] * vA - gA);
                        gnBo = 2.0 * (3.0 * nB[ij] * vB - gB);
                        aonm = alphaAB + omega * nA[ij] + mu * nB[ij];
                        Asqcoeff = -epsA + nA[ij] * (3.0 * vA * nA[ij] - 2.0 * gA) + omega * nB[ij];
                        Bsqcoeff = -epsB + nB[ij] * (3.0 * vB * nB[ij] - 2.0 * gB) + mu * nA[ij];
                        engij[ij] = engij[ij] + Asqcoeff * Ssq + Bsqcoeff * Tsq + gnAo * (A1[ij] * A2[ij] * A3[ij] + conj(A1[ij] * A2[ij] * A3[ij])) + gnBo * (B1[ij] * B2[ij] * B3[ij] + conj(B1[ij] * B2[ij] * B3[ij])) + vA3 * Ssq * Ssq + vB3 * Tsq * Tsq - 1.5 * vA * (A1[ij] * conj(A1[ij]) * A1[ij] * conj(A1[ij]) + A2[ij] * conj(A2[ij]) * A2[ij] * conj(A2[ij]) + A3[ij] * conj(A3[ij]) * A3[ij] * conj(A3[ij])) - 1.5 * vB * (B1[ij] * conj(B1[ij]) * B1[ij] * conj(B1[ij]) + B2[ij] * conj(B2[ij]) * B2[ij] * conj(B2[ij]) + B3[ij] * conj(B3[ij]) * B3[ij] * conj(B3[ij])) + aonm * (A1[ij] * conj(B1[ij]) + B1[ij] * conj(A1[ij]) + A2[ij] * conj(B2[ij]) + B2[ij] * conj(A2[ij]) + A3[ij] * conj(B3[ij]) + B3[ij] * conj(A3[ij])) + omega * (A1[ij] * A2[ij] * B3[ij] + conj(A1[ij] * A2[ij] * B3[ij]) + A1[ij] * A3[ij] * B2[ij] + conj(A1[ij] * A3[ij] * B2[ij]) + A2[ij] * A3[ij] * B1[ij] + conj(A2[ij] * A3[ij] * B1[ij])) + mu * (A1[ij] * B2[ij] * B3[ij] + conj(A1[ij] * B2[ij] * B3[ij]) + A2[ij] * B1[ij] * B3[ij] + conj(A2[ij] * B1[ij] * B3[ij]) + A3[ij] * B1[ij] * B2[ij] + conj(A3[ij] * B1[ij] * B2[ij])) + nA[ij] * nA[ij] * (0.5 * (-epsA + 1.0) + nA[ij] * (-gA / 3. + 0.25 * vA * nA[ij])) + nB[ij] * nB[ij] * (0.5 * (-epsB + betaB) + nB[ij] * (-gB / 3. + 0.25 * vB * nB[ij])) + nA[ij] * nB[ij] * (alphaAB + 0.5 * (omega * nA[ij] + mu * nB[ij])); //
                    }
                }
                sprintf(filename, "%s_%d_%d.deng", run, n, itheta);
                //		      if(myid==0) {printf("output n, elapsed time = %i %e \n",n,cpuTime);}
                MPI_Barrier(MPI_COMM_WORLD);
                for (pc = 0; pc < np; pc++)
                {
                    MPI_Barrier(MPI_COMM_WORLD);
                    if (myid == pc)
                    {
                        if (myid == 0)
                        {
                            fout = fopen(filename, "w");
                        }
                        else
                        {
                            fout = fopen(filename, "a");
                        }
                        for (i = 0; i < Lxl; ++i)
                        {
                            ig = i + myid * Lxl;
                            for (j = 0; j < Ly; ++j)
                            {
                                ij = i * Ly + j;
                                A1sq = A1[ij] * conj(A1[ij]);
                                A2sq = A2[ij] * conj(A2[ij]);
                                A3sq = A3[ij] * conj(A3[ij]);
                                B1sq = B1[ij] * conj(B1[ij]);
                                B2sq = B2[ij] * conj(B2[ij]);
                                B3sq = B3[ij] * conj(B3[ij]);
                                Ssq = 2.0 * (A1sq + A2sq + A3sq);
                                Tsq = 2.0 * (B1sq + B2sq + B3sq);
                                fprintf(fout, "%i %i %22.14e %e %e %e \n", ig, j, engij[ij], Tsq + Ssq, sqrt(Tsq / 6.0), sqrt(Ssq / 6.0));
                            }
                        }
                        fclose(fout);
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            }
            if (n % neng == 0)
            {
                char filename[BUFSIZ];
                FILE *fout;
                enl = 0.0;
                // calculate gradient energy
                for (i = 0; i < Lxl; ++i)
                {
                    ig = i + myid * Lxl;
                    if (ig < Lx / 2)
                    {
                        kx = ig * fx;
                    }
                    else
                    {
                        kx = (ig - Lx) * fx;
                    }
                    for (j = 0; j < Ly; ++j)
                    {
                        ij = i * Ly + j;
                        if (j < Ly / 2)
                        {
                            ky = j * fy;
                        }
                        else
                        {
                            ky = (j - Ly) * fy;
                        }

                        kA1f = -(kx * kx + ky * ky) - 2.0 * (G1x * kx + G1y * ky) * alpha + 1.0 - alpha * alpha;
                        kB1f = kA1f;
                        // kB1f=-(kx*kx+ky*ky)-2.0*(G1x*kx+G1y*ky)*alpha+1.0-alpha*alpha;

                        kA2f = -(kx * kx + ky * ky) - 2.0 * (G2x * kx + G2y * ky) * alpha + 1.0 - alpha * alpha;
                        kB2f = kA2f;
                        // kB2f=-(kx*kx+ky*ky)-2.0*(G2x*kx+G2y*ky)*alpha+1.0-alpha*alpha;

                        kA3f = -(kx * kx + ky * ky) - 2.0 * (G3x * kx + G3y * ky) * alpha + 1.0 - alpha * alpha;
                        kB3f = kA3f;
                        // kB3f=-(kx*kx+ky*ky)-2.0*(G3x*kx+G3y*ky)*alpha+1.0-alpha*alpha;

                        enl = enl + kA1f * kA1f * Ak1[ij] * conj(Ak1[ij]) + kA2f * kA2f * Ak2[ij] * conj(Ak2[ij]) + kA3f * kA3f * Ak3[ij] * conj(Ak3[ij]) + betaB * (kB1f * kB1f * Bk1[ij] * conj(Bk1[ij]) + kB2f * kB2f * Bk2[ij] * conj(Bk2[ij]) + kB3f * kB3f * Bk3[ij] * conj(Bk3[ij]));
                    }
                }
                enl = enl * Lx * Ly; // enl=0.0;
                // alternate (slower) method to calculate gradient energy
                /*                 	for(i=0;i<Lxl;++i){ig=i+myid*Lxl;
                                if(ig< Lx/2){kx=ig*fx;} else {kx=(ig-Lx)*fx;}
                                    for(j=0;j< Ly;++j){ij=i*Ly+j;
                                if(j< Ly/2){ky=j*fy;} else {ky=(j-Ly)*fy;}

                            kA1f=-(kx*kx+ky*ky)-2.0*(G1x*kx+G1y*ky)*alpha+1.0-alpha*alpha; kB1f=kA1f;
                            //kB1f=-(kx*kx+ky*ky)-2.0*(G1x*kx+G1y*ky)*alpha+1.0-alpha*alpha;

                            kA2f=-(kx*kx+ky*ky)-2.0*(G2x*kx+G2y*ky)*alpha+1.0-alpha*alpha; kB2f=kA2f;
                            //kB2f=-(kx*kx+ky*ky)-2.0*(G2x*kx+G2y*ky)*alpha+1.0-alpha*alpha;

                            kA3f=-(kx*kx+ky*ky)-2.0*(G3x*kx+G3y*ky)*alpha+1.0-alpha*alpha; kB3f=kA3f;
                            //kB3f=-(kx*kx+ky*ky)-2.0*(G3x*kx+G3y*ky)*alpha+1.0-alpha*alpha;
                            //
                            An1[ij]=A1[ij]; An2[ij]=A2[ij]; An3[ij]=A3[ij];
                            Bn1[ij]=B1[ij]; Bn2[ij]=B2[ij]; Bn3[ij]=B3[ij];
                            Ak1[ij]=kA1f*Ak1[ij]; Ak2[ij]=kA2f*Ak2[ij]; Ak3[ij]=kA3f*Ak3[ij];
                            Bk1[ij]=kB1f*Bk1[ij]; Bk2[ij]=kB2f*Bk2[ij]; Bk3[ij]=kB3f*Bk3[ij];
                                 }}
                                 fftw_execute(plan1Ab);  fftw_execute(plan2Ab);  fftw_execute(plan3Ab);
                                 fftw_execute(plan1Bb);  fftw_execute(plan2Bb);  fftw_execute(plan3Bb);
                                 for(i=0;i<Lxl;++i){ for(j=0;j< Ly;++j){ij=i*Ly+j;
                            enl=enl+A1[ij]*conj(A1[ij])+A2[ij]*conj(A2[ij])+A3[ij]*conj(A3[ij]);
                            enl=enl+B1[ij]*conj(B1[ij])+B2[ij]*conj(B2[ij])+B3[ij]*conj(B3[ij]);
                         }}
                                 for(i=0;i<Lxl;++i){ for(j=0;j< Ly;++j){ij=i*Ly+j;
                            A1[ij]=An1[ij]; A2[ij]=An2[ij]; A3[ij]=An3[ij];
                            B1[ij]=Bn1[ij]; B2[ij]=Bn2[ij]; B3[ij]=Bn3[ij];
                            An1[ij]=A1[ij]; An2[ij]=A2[ij]; An3[ij]=A3[ij];
                            Bn1[ij]=B1[ij]; Bn2[ij]=B2[ij]; Bn3[ij]=B3[ij];
                         }}
                                 fftw_execute(plan1An);  fftw_execute(plan2An);  fftw_execute(plan3An);
                                 fftw_execute(plan1Bn);  fftw_execute(plan2Bn);  fftw_execute(plan3Bn);
                                 for(i=0;i<Lxl;++i){ for(j=0;j< Ly;++j){ij=i*Ly+j;
                            Ak1[ij]=An1[ij]*Sf; Ak2[ij]=An2[ij]*Sf; Ak3[ij]=An3[ij]*Sf;
                            Bk1[ij]=Bn1[ij]*Sf; Bk2[ij]=Bn2[ij]*Sf; Bk3[ij]=Bn3[ij]*Sf;
                         }} */
                // printf(" grad energy = %e \n",enl);
                for (i = 0; i < Lxl; ++i)
                {
                    for (j = 0; j < Ly; ++j)
                    {
                        ij = i * Ly + j;
                        // Eq. 18  check 08/05/23
                        Ssq = A1[ij] * conj(A1[ij]) + A2[ij] * conj(A2[ij]) + A3[ij] * conj(A3[ij]);
                        Tsq = B1[ij] * conj(B1[ij]) + B2[ij] * conj(B2[ij]) + B3[ij] * conj(B3[ij]);
                        gnAo = 2.0 * (3.0 * nA[ij] * vA - gA);
                        gnBo = 2.0 * (3.0 * nB[ij] * vB - gB);
                        aonm = alphaAB + omega * nA[ij] + mu * nB[ij];
                        Asqcoeff = -epsA + nA[ij] * (3.0 * vA * nA[ij] - 2.0 * gA) + omega * nB[ij];
                        Bsqcoeff = -epsB + nB[ij] * (3.0 * vB * nB[ij] - 2.0 * gB) + mu * nA[ij];
                        enl = enl + Asqcoeff * Ssq + Bsqcoeff * Tsq + gnAo * (A1[ij] * A2[ij] * A3[ij] + conj(A1[ij] * A2[ij] * A3[ij])) + gnBo * (B1[ij] * B2[ij] * B3[ij] + conj(B1[ij] * B2[ij] * B3[ij])) + vA3 * Ssq * Ssq + vB3 * Tsq * Tsq - 1.5 * vA * (A1[ij] * conj(A1[ij]) * A1[ij] * conj(A1[ij]) + A2[ij] * conj(A2[ij]) * A2[ij] * conj(A2[ij]) + A3[ij] * conj(A3[ij]) * A3[ij] * conj(A3[ij])) - 1.5 * vB * (B1[ij] * conj(B1[ij]) * B1[ij] * conj(B1[ij]) + B2[ij] * conj(B2[ij]) * B2[ij] * conj(B2[ij]) + B3[ij] * conj(B3[ij]) * B3[ij] * conj(B3[ij])) + aonm * (A1[ij] * conj(B1[ij]) + B1[ij] * conj(A1[ij]) + A2[ij] * conj(B2[ij]) + B2[ij] * conj(A2[ij]) + A3[ij] * conj(B3[ij]) + B3[ij] * conj(A3[ij])) + omega * (A1[ij] * A2[ij] * B3[ij] + conj(A1[ij] * A2[ij] * B3[ij]) + A1[ij] * A3[ij] * B2[ij] + conj(A1[ij] * A3[ij] * B2[ij]) + A2[ij] * A3[ij] * B1[ij] + conj(A2[ij] * A3[ij] * B1[ij])) + mu * (A1[ij] * B2[ij] * B3[ij] + conj(A1[ij] * B2[ij] * B3[ij]) + A2[ij] * B1[ij] * B3[ij] + conj(A2[ij] * B1[ij] * B3[ij]) + A3[ij] * B1[ij] * B2[ij] + conj(A3[ij] * B1[ij] * B2[ij])) + nA[ij] * nA[ij] * (0.5 * (-epsA + 1.0) + nA[ij] * (-gA / 3. + 0.25 * vA * nA[ij])) + nB[ij] * nB[ij] * (0.5 * (-epsB + betaB) + nB[ij] * (-gB / 3. + 0.25 * vB * nB[ij])) + nA[ij] * nB[ij] * (alphaAB + 0.5 * (omega * nA[ij] + mu * nB[ij])); //
                    }
                }
                if (myid == 0)
                {
                    eng = enl; // printf("np, enl = %i %e \n",np,enl);
                    for (p2 = 1; p2 < np; p2++)
                    {
                        MPI_Recv(&enl, 1, MPI_DOUBLE, p2, 0, MPI_COMM_WORLD, &status);
                        eng = eng + enl;
                        // printf(" p2 enl  = %i %e %e\n",p2,enl,eng);
                    }
                    sprintf(filename, "%s_%d.eng", run, itheta);
                    if (ich == 0)
                    {
                        fout = fopen(filename, "w");
                    }
                    else
                    {
                        fout = fopen(filename, "a");
                    }
                    ich = 1;
                    gamma = ffac * (2 * twopi / sqrt(3)) / (2.51) * (eng * Sf - fn) * Lx * dx;
                    fprintf(fout, "%e %22.14e %22.14e \n", n * dt, eng * Sf, gamma);
                    printf("%e %22.14e %22.14e \n", n * dt, eng * Sf, gamma);
                    fclose(fout);
                }
                else
                {
                    MPI_Send(&enl, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                }
            }
        }
    }
    //	 }
    fftw_destroy_plan(plan1An);
    fftw_destroy_plan(plan2An);
    fftw_destroy_plan(plan3An);
    fftw_destroy_plan(plan1Ab);
    fftw_destroy_plan(plan2Ab);
    fftw_destroy_plan(plan3Ab);
    fftw_destroy_plan(plan1Bn);
    fftw_destroy_plan(plan2Bn);
    fftw_destroy_plan(plan3Bn);
    fftw_destroy_plan(plan1Bb);
    fftw_destroy_plan(plan2Bb);
    fftw_destroy_plan(plan3Bb);
    MPI_Finalize();
    printf("Goodbye world\n");
    return (0);
}
