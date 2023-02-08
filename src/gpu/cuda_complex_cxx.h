#ifndef CUDA_COMPLEX_CXX_H
#define CUDA_COMPLEX_CXX_H

#include "cuda_complex.h"
#include <complex.h>

// As the _Complex types are not known in CUDA C++, this may only be a host function.
constexpr cu_cmplx_double to_cuda_cmplx_double(double _Complex from) {
  return {creal(from), cimag(from)};
}

constexpr cu_cmplx_float to_cuda_cmplx_float(float _Complex from) {
  return {crealf(from), cimagf(from)};
}

__host__ __device__ constexpr cu_cmplx_double to_cuda_cmplx_double(cu_cmplx_double from) {
  return {from.x, from.y};
}
__host__ __device__ constexpr cu_cmplx_double to_cuda_cmplx_double(cu_cmplx_float from) {
  return {from.x, from.y};
}
__host__ __device__ constexpr cu_cmplx_float to_cuda_cmplx_float(cu_cmplx_double from) {
  return {(float)from.x, (float)from.y};
}
__host__ __device__ constexpr cu_cmplx_float to_cuda_cmplx_float(cu_cmplx_float from) {
  return {from.x, from.y};
}

constexpr cu_cmplx_double CU_CMPLX_double_ONE       = { 1.0, 0.0};
constexpr cu_cmplx_double CU_CMPLX_double_MINUS_ONE = {-1.0, 0.0};
constexpr cu_cmplx_double CU_CMPLX_double_ZERO      = { 0.0, 0.0};
constexpr cu_cmplx_float  CU_CMPLX_float_ONE        = { 1.0, 0.0};
constexpr cu_cmplx_float  CU_CMPLX_float_MINUS_ONE  = {-1.0, 0.0};
constexpr cu_cmplx_float  CU_CMPLX_float_ZERO       = { 1.0, 0.0};

#endif //CUDA_COMPLEX_CXX_H