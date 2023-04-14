#ifndef CUDA_COMPLEX_OPERATORS_H
#define CUDA_COMPLEX_OPERATORS_H

#include "cuda_complex.h"

// mixed precision operations which need to do type promotion of one of the operands


__host__ __device__ cu_cmplx_double operator*(cu_cmplx_double const& lhs,
                                              cu_cmplx_float const& rhs);
__host__ __device__ cu_cmplx_double operator*(cu_cmplx_float const& lhs,
                                              cu_cmplx_double const& rhs);
__host__ __device__ cu_cmplx_double operator-(cu_cmplx_float const& lhs,
                                              cu_cmplx_double const& rhs);
__host__ __device__ cu_cmplx_double operator-(cu_cmplx_double const& lhs,
                                              cu_cmplx_float const& rhs);

#endif // CUDA_COMPLEX_OPERATORS_H
