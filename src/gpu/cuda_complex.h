#ifndef CUDA_COMPLEX_H
#define CUDA_COMPLEX_H
#include <cuComplex.h>

typedef cuDoubleComplex cu_cmplx_double;
typedef cuFloatComplex cu_cmplx_float;
typedef cuDoubleComplex cu_config_double;
typedef cuFloatComplex cu_config_float;

// MACROS for the creation of complex numbers
#define make_cu_cmplx_double(cur, cui) make_cuDoubleComplex(cur, cui)
#define make_cu_cmplx_float(cur, cui) make_cuFloatComplex(cur, cui)
// MACROS of functions acting ON complex numbers
#define cu_creal_double(c_nr) cuCreal(c_nr)
#define cu_cimag_double(c_nr) cuCimag(c_nr)
#define cu_creal_float(c_nr) cuCrealf(c_nr)
#define cu_cimag_float(c_nr) cuCimagf(c_nr)
#define cu_cmul_float(c_nr1, c_nr2) cuCmulf(c_nr1, c_nr2)
#define cu_cmul_double(c_nr1, c_nr2) cuCmul(c_nr1, c_nr2)
#define cu_cdiv_float(c_nr1, c_nr2) cuCdivf(c_nr1, c_nr2)
#define cu_cdiv_double(c_nr1, c_nr2) cuCdiv(c_nr1, c_nr2)
#define cu_csub_float(c_nr1, c_nr2) cuCsubf(c_nr1, c_nr2)
#define cu_csub_double(c_nr1, c_nr2) cuCsub(c_nr1, c_nr2)
#define cu_cadd_float(c_nr1, c_nr2) cuCaddf(c_nr1, c_nr2)
#define cu_cadd_double(c_nr1, c_nr2) cuCadd(c_nr1, c_nr2)
#define cu_conj_float(c_nr) cuConjf(c_nr)
#define cu_conj_double(c_nr) cuConj(c_nr)

#endif //CUDA_COMPLEX_H
