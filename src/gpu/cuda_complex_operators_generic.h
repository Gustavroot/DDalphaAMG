/**
 * \file cuda_complex_operators_generic.h
 * 
 * C++ operator overloads for CUDA complex numbers.
 */


#ifndef CUDA_COMPLEX_OPERATORS_PRECISION_H
#define CUDA_COMPLEX_OPERATORS_PRECISION_H

#include "cuda_complex.h"

/**
 * \brief Create a literal which is of type cu_cmplx_precision.
 * 
 * This only yields an imaginary part. Thus a term like '1.0_cu_i_PRECISION' yields a complex
 * number 0.0 + 1.0i, which is processable on the GPU (i.e. of type 'cu_cmplx_PRECISION'). Both
 * float and double is supported.
 * 
 * This method is not to be called directly but provides an operator as per C++ specification.
 * 
 * The method is a constexpr and thus capable to populate variables with static storage duration.
 * 
 * \param imag    The literal to process. All conversion is automatic and the type 'long double'
 *                should only be taken in a symbolic sense and is part of the processing that occurs
 *                when writing a literal like '1.0_cu_i_PRECISION'.
 * \return        A complex number with an imaginary part of value 'imag'.
 */
__host__ __device__ constexpr cu_cmplx_PRECISION operator ""_cu_i_PRECISION(long double imag) {
  return {0, (PRECISION)imag};
}

__host__ __device__ cu_cmplx_PRECISION operator*(cu_cmplx_PRECISION const& lhs, int const& rhs);
__host__ __device__ cu_cmplx_PRECISION operator*(int const& lhs, cu_cmplx_PRECISION const& rhs);
__host__ __device__ cu_cmplx_PRECISION operator*(cu_cmplx_PRECISION const& lhs,
                                                 cu_cmplx_PRECISION const& rhs);

__host__ __device__ cu_cmplx_PRECISION operator-(cu_cmplx_PRECISION const& lhs,
                                                 cu_cmplx_PRECISION const& rhs);
__host__ __device__ cu_cmplx_PRECISION operator+(cu_cmplx_PRECISION const& lhs,
                                                 cu_cmplx_PRECISION const& rhs);
__host__ __device__ cu_cmplx_PRECISION & operator+=(cu_cmplx_PRECISION & lhs,
                                                  cu_cmplx_PRECISION const& rhs);
__host__ __device__ cu_cmplx_PRECISION & operator-=(cu_cmplx_PRECISION & lhs,
                                                  cu_cmplx_PRECISION const& rhs);
__host__ __device__ cu_cmplx_PRECISION operator-(cu_cmplx_PRECISION const& value);

#endif  // CUDA_COMPLEX_OPERATORS_PRECISION_H
