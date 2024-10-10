! random number (uniform deviation): between 0 and 1
!  ran3 for iris: use static vars.
      function ran3(idum)
      integer idum
      integer mbig,mseed,mz
      double precision ran3,fac
!         parameter (mbig=4000000.,mseed=1618033.,mz=0.,fac=2.5e-7)
      parameter (mbig=1000000000,mseed=161803398,mz=0,fac=1.d0/mbig)
      integer mj,mk,ma(55)
      integer i,iff,ii,inext,inextp,k
      save iff,inext,inextp,ma
      data iff /0/
1      if(idum.lt.0.or.iff.eq.0)then
        iff=1
        mj=mseed-iabs(idum)
        mj=mod(mj,mbig)
        ma(55)=mj
        mk=1
        do 11 i=1,54
          ii=mod(21*i,55)
          ma(ii)=mk
          mk=mj-mk
          if(mk.lt.mz)mk=mk+mbig
          mj=ma(ii)
11      continue
        do 13 k=1,4
          do 12 i=1,55
            ma(i)=ma(i)-ma(1+mod(i+30,55))
            if (ma(i).lt.mz) ma(i)=ma(i)+mbig
12        continue
13      continue
        inext=0
        inextp=31
        idum=1
      endif
      inext=inext+1
      if (inext.eq.56) inext=1
      inextp=inextp+1
      if (inextp.eq.56) inextp=1
      mj=ma(inext)-ma(inextp)
      if (mj.lt.mz) mj=mj+mbig
      ma(inext)=mj
      ran3=mj*fac
      if (ran3.le.0.or.ran3.ge.1) goto 1
      return
      end                              

! Normal (Gaussian) distribution with zero mean and unit variance
! for a Gaussian variable y with mean y0 and variance sig^2,
!     (y-y0)/sig = x =gasdev, and hence y=sig*x + y0
      FUNCTION gasdev(idum)
      INTEGER idum
      double precision gasdev
!    USES ran3
      INTEGER iset
      double precision fac,gset,rsq,v1,v2,ran3
      SAVE iset,gset
      DATA iset/0/
      if (iset.eq.0) then
1       v1=2.d0*ran3(idum)-1.d0
        v2=2.d0*ran3(idum)-1.d0
        rsq=v1**2+v2**2
        if(rsq.ge.1.d0.or.rsq.eq.0.d0)goto 1
        fac=sqrt(-2.d0*log(rsq)/rsq)
        gset=v1*fac
        gasdev=v2*fac
        iset=1
      else
        gasdev=gset
        iset=0
      endif
      return
      END
