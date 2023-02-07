#ifndef CUDA_COMPLEX_OPERATORS_PRECISION_H
#define CUDA_COMPLEX_OPERATORS_PRECISION_H

#include "cuda_complex.h"

__host__ __device__ cu_cmplx_PRECISION operator*(cu_cmplx_PRECISION const& lhs, int const& rhs);
__host__ __device__ cu_cmplx_PRECISION operator*(int const& lhs, cu_cmplx_PRECISION const& rhs);
__host__ __device__ cu_cmplx_PRECISION operator*(cu_cmplx_PRECISION const& lhs,
                                                 cu_cmplx_PRECISION const& rhs);

__host__ __device__ cu_cmplx_PRECISION operator-(cu_cmplx_PRECISION const& lhs,
                                                 cu_cmplx_PRECISION const& rhs);
__host__ __device__ cu_cmplx_PRECISION operator+(cu_cmplx_PRECISION const& lhs,
                                                 cu_cmplx_PRECISION const& rhs);
__host__ __device__ cu_cmplx_PRECISION operator+=(cu_cmplx_PRECISION const& lhs,
                                                  cu_cmplx_PRECISION const& rhs);
__host__ __device__ cu_cmplx_PRECISION operator-(cu_cmplx_PRECISION const& value);

#endif  // CUDA_COMPLEX_OPERATORS_PRECISION_H
