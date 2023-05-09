#include "cuda_complex_operators_PRECISION.h"
#include <stdio.h>

__host__ __device__ cu_cmplx_PRECISION operator*(cu_cmplx_PRECISION const& lhs, int const& rhs) {
  return cu_cmul_PRECISION(lhs, make_cu_cmplx_PRECISION(rhs, 0));
}

__host__ __device__ cu_cmplx_PRECISION operator*(int const& lhs, cu_cmplx_PRECISION const& rhs) {
  return cu_cmul_PRECISION(make_cu_cmplx_PRECISION(lhs, 0), rhs);
}

__host__ __device__ cu_cmplx_PRECISION operator*(cu_cmplx_PRECISION const& lhs,
                                                 cu_cmplx_PRECISION const& rhs) {
  return cu_cmul_PRECISION(lhs, rhs);
}

__host__ __device__ cu_cmplx_PRECISION operator-(cu_cmplx_PRECISION const& lhs,
                                                 cu_cmplx_PRECISION const& rhs) {
  return cu_csub_PRECISION(lhs, rhs);
}

__host__ __device__ cu_cmplx_PRECISION operator+(cu_cmplx_PRECISION const& lhs,
                                                 cu_cmplx_PRECISION const& rhs) {
  return cu_cadd_PRECISION(lhs, rhs);
}

__host__ __device__ cu_cmplx_PRECISION & operator+=(cu_cmplx_PRECISION & lhs,
                                                    cu_cmplx_PRECISION const& rhs) {
  lhs = cu_cadd_PRECISION(lhs, rhs);
  return lhs;
}

__host__ __device__ cu_cmplx_PRECISION & operator-=(cu_cmplx_PRECISION & lhs,
                                                    cu_cmplx_PRECISION const& rhs) {
  lhs = cu_csub_PRECISION(lhs, rhs);
  return lhs;
}

__host__ __device__ cu_cmplx_PRECISION operator-(cu_cmplx_PRECISION const& value) {
  return {-value.x, -value.y};
}
