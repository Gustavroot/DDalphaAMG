#include "cuda_complex_operators.h"

__host__ __device__ cu_cmplx_double operator*(cu_cmplx_double const& lhs,
                                              cu_cmplx_float const& rhs) {
  const cu_cmplx_double rhs_double = make_cu_cmplx_double(cu_creal_float(rhs), cu_cimag_float(rhs));
  return cu_cmul_double(lhs, rhs_double);
}

__host__ __device__ cu_cmplx_double operator*(cu_cmplx_float const& lhs,
                                              cu_cmplx_double const& rhs) {
  const cu_cmplx_double lhs_double = make_cu_cmplx_double(cu_creal_float(lhs), cu_cimag_float(lhs));
  return cu_cmul_double(lhs_double, rhs);
}

__host__ __device__ cu_cmplx_double operator-(cu_cmplx_float const& lhs,
                                              cu_cmplx_double const& rhs) {
  const cu_cmplx_double lhs_double = make_cu_cmplx_double(cu_creal_float(lhs), cu_cimag_float(lhs));
  return cu_cmul_double(lhs_double, rhs);
}

__host__ __device__ cu_cmplx_double operator-(cu_cmplx_double const& lhs,
                                              cu_cmplx_float const& rhs) {
  const cu_cmplx_double rhs_double = make_cu_cmplx_double(cu_creal_float(rhs), cu_cimag_float(rhs));
  return cu_cmul_double(lhs, rhs_double);
}
