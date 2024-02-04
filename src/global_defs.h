#ifndef GLOBAL_DEFS_H
#define GLOBAL_DEFS_H

#include <complex.h>

#define STRINGLENGTH 500

#define _FILE_OFFSET_BITS 64
#define EPS_float 1E-6
#define EPS_double 1E-14

#define FOR2(e) \
    {           \
        e e     \
    }
#define FOR3(e) \
    {           \
        e e e   \
    }
#define FOR4(e) \
    {           \
        e e e e \
    }
#define FOR10(e)            \
    {                       \
        e e e e e e e e e e \
    }
#define FOR20(e)                                \
    {                                           \
        e e e e e e e e e e e e e e e e e e e e \
    }
#define FOR40(e)                                                                        \
    {                                                                                   \
        e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e \
    }
#define FOR6(e)     \
    {               \
        e e e e e e \
    }
#define FOR12(e)                \
    {                           \
        e e e e e e e e e e e e \
    }
#define FOR24(e)                                        \
    {                                                   \
        e e e e e e e e e e e e e e e e e e e e e e e e \
    }
#define FOR36(e)          \
    {                     \
        FOR12(e)          \
        FOR12(e) FOR12(e) \
    }
#define FOR42(e) \
    {            \
        FOR36(e) \
        FOR6(e)  \
    }

#define SQUARE(e) (e) * (e)
#define NORM_SQUARE_float(e) SQUARE(crealf(e)) + SQUARE(cimagf(e))
#define NORM_SQUARE_double(e) SQUARE(creal(e)) + SQUARE(cimag(e))
#define CSPLIT(e) creal(e), cimag(e)

#define MPI_double MPI_DOUBLE
#define MPI_float MPI_FLOAT
#define MPI_COMPLEX_double MPI_DOUBLE_COMPLEX
#define MPI_COMPLEX_float MPI_COMPLEX

#ifdef IMPORT_FROM_EXTERN_C
// anything to add here ?
#else
#define I _Complex_I
#endif

#define conj_double conj
#define conj_float conjf
#define cabs_double cabs
#define cabs_float cabsf
#define creal_double creal
#define creal_float crealf
#define cimag_double cimag
#define cimag_float cimagf
#define csqrt_double csqrt
#define csqrt_float csqrtf
#define cpow_double cpow
#define cpow_float cpowf
#define pow_double pow
#define pow_float powf
#define abs_float fabs
#define abs_double abs

#if defined(GCRODR) || defined(POLYPREC)
  #define geev_double LAPACKE_zgeev 
  #define geev_float LAPACKE_cgeev 
  #define ggev_double LAPACKE_zggev
  #define ggev_float LAPACKE_cggev
  #define geqr2_double LAPACKE_zgeqr2
  #define geqr2_float LAPACKE_cgeqr2
  #define ungqr_double LAPACKE_zungqr
  #define ungqr_float LAPACKE_cungqr
  #define trtri_double LAPACKE_ztrtri
  #define trtri_float LAPACKE_ctrtri
  #define gesv_double LAPACKE_zgesv
  #define gesv_float LAPACKE_cgesv
  #define gels_double LAPACKE_zgels
  #define gels_float LAPACKE_cgels
  //#define pgeqr2_double pzgeqr2_
  //#define pgeqr2_float pcgeqr2_
  //#define pung2r_double pzung2r_
  //#define pung2r_float pcung2r_

  //#define MKL_float MKL_Complex8
  //#define MKL_double MKL_Complex16
#endif

#endif // GLOBAL_DEFS_H
