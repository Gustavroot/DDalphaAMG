/**
 * \file cuda_complex_cxx.h
 * 
 * \brief The part of cuda_complex.h that makes extensive use of C++ language features.
 * 
 * \warning Not to be included in C code or in an 'extern "C"' context.
 */


#ifndef CUDA_COMPLEX_CXX_H
#define CUDA_COMPLEX_CXX_H

#include "cuda_complex.h"
#include <complex.h>

/**
 * \brief Create a CUDA complex number from a C style complex number in double precision.
 * 
 * As the _Complex types are not known in CUDA C++, this may only be a host function.
 * 
 * \param from  The (C style) complex number to be converted.
 * \returns     A CUDA C++ style complex number of the same value as 'from'.
 */
constexpr cu_cmplx_double to_cuda_cmplx_double(double _Complex from) {
  return {creal(from), cimag(from)};
}

/**
 * \brief Create a CUDA complex number from a C style complex number in single precision.
 * 
 * As the _Complex types are not known in CUDA C++, this may only be a host function.
 * 
 * \param from  The (C style) complex number to be converted.
 * \returns     A CUDA C++ style complex number of the same value as 'from'.
 */
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
