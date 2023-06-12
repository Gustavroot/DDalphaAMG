#include "cuda_complex_operators_PRECISION.h"
#include "cuda_mvm_PRECISION.h"

__device__ void cuda_mvm_PRECISION(cu_cmplx_PRECISION *y, cu_cmplx_PRECISION const *M,
                                   cu_cmplx_PRECISION const *x) {
  y[0] = M[0] * x[0];
  y[0] += M[1] * x[1];
  y[0] += M[2] * x[2];
  y[1] = M[3] * x[0];
  y[1] += M[4] * x[1];
  y[1] += M[5] * x[2];
  y[2] = M[6] * x[0];
  y[2] += M[7] * x[1];
  y[2] += M[8] * x[2];
}

__device__ void cuda_mvmh_PRECISION(cu_cmplx_PRECISION *y, cu_cmplx_PRECISION const *M,
                                    cu_cmplx_PRECISION const *x) {
  y[0] = cu_conj_PRECISION(M[0]) * x[0];
  y[1] = cu_conj_PRECISION(M[1]) * x[0];
  y[2] = cu_conj_PRECISION(M[2]) * x[0];
  y[0] += cu_conj_PRECISION(M[3]) * x[1];
  y[1] += cu_conj_PRECISION(M[4]) * x[1];
  y[2] += cu_conj_PRECISION(M[5]) * x[1];
  y[0] += cu_conj_PRECISION(M[6]) * x[2];
  y[1] += cu_conj_PRECISION(M[7]) * x[2];
  y[2] += cu_conj_PRECISION(M[8]) * x[2];
}

__device__ void cuda_mvm_componentwise_PRECISION(ComponentAccess<cu_cmplx_PRECISION> y,
                                                 ComponentAccess<cu_cmplx_PRECISION const> M,
                                                 cu_cmplx_PRECISION const *x) {
  y[0] = M[0] * x[0];
  y[0] += M[1] * x[1];
  y[0] += M[2] * x[2];
  y[1] = M[3] * x[0];
  y[1] += M[4] * x[1];
  y[1] += M[5] * x[2];
  y[2] = M[6] * x[0];
  y[2] += M[7] * x[1];
  y[2] += M[8] * x[2];
}

__device__ void cuda_mvmh_componentwise_PRECISION(cu_cmplx_PRECISION *y,
                                                  ComponentAccess<cu_cmplx_PRECISION const> M,
                                                  ComponentAccess<cu_cmplx_PRECISION const> x) {
  y[0] = cu_conj_PRECISION(M[0]) * x[0];
  y[1] = cu_conj_PRECISION(M[1]) * x[0];
  y[2] = cu_conj_PRECISION(M[2]) * x[0];
  y[0] += cu_conj_PRECISION(M[3]) * x[1];
  y[1] += cu_conj_PRECISION(M[4]) * x[1];
  y[2] += cu_conj_PRECISION(M[5]) * x[1];
  y[0] += cu_conj_PRECISION(M[6]) * x[2];
  y[1] += cu_conj_PRECISION(M[7]) * x[2];
  y[2] += cu_conj_PRECISION(M[8]) * x[2];
}
