#ifndef CUDA_MVM_PRECISION_H
#define CUDA_MVM_PRECISION_H

#include "cuda_complex.h"

__device__ void cuda_mvm_PRECISION(cu_cmplx_PRECISION *y, cu_cmplx_PRECISION const *M,
                                   cu_cmplx_PRECISION const *x);

__device__ void cuda_mvmh_PRECISION(cu_cmplx_PRECISION *y, cu_cmplx_PRECISION const *M,
                                    cu_cmplx_PRECISION const *x);

#endif  // CUDA_MVM_PRECISION_H