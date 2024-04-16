/*
 * Copyright (C) 2016, Matthias Rottmann, Artur Strebel, Gustavo Ramirez, Simon Heybrock, Simone Bacchio, Bjoern Leder, Issaku Kanamori, Tilmann Matthaei, Ke-Long Zhang.
 * 
 * This file is part of the DDalphaAMG solver library.
 * 
 * The DDalphaAMG solver library is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * The DDalphaAMG solver library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * 
 * You should have received a copy of the GNU General Public License
 * along with the DDalphaAMG solver library. If not, see http://www.gnu.org/licenses/.
 * 
 */

#ifndef VECTORIZATION_CONTROL_H
#define VECTORIZATION_CONTROL_H

#ifdef SSE

#define SIMD_LENGTH_float  4
#define SIMD_LENGTH_double 2

#define INTERPOLATION_OPERATOR_LAYOUT_OPTIMIZED_float
#define INTERPOLATION_SETUP_LAYOUT_OPTIMIZED_float
#define VECTORIZE_COARSE_OPERATOR_float
#define GRAM_SCHMIDT_VECTORIZED_float
#define OPTIMIZED_NEIGHBOR_COUPLING_float
#define OPTIMIZED_SELF_COUPLING_float
#define OPTIMIZED_NEIGHBOR_COUPLING_double
#define OPTIMIZED_LINALG_float
#define OPTIMIZED_LINALG_double

#include "sse_complex_float_intrinsic.h"
#include "sse_complex_double_intrinsic.h"

#endif // SSE

#define OPERATOR_COMPONENT_OFFSET_float  (SIMD_LENGTH_float * ((l->num_eig_vect + SIMD_LENGTH_float - 1) / SIMD_LENGTH_float))
#define OPERATOR_COMPONENT_OFFSET_double (SIMD_LENGTH_double * ((l->num_eig_vect + SIMD_LENGTH_double - 1) / SIMD_LENGTH_double))

#define OPERATOR_TYPE_float  float
#define OPERATOR_TYPE_double double


/**
 * @brief AVX/AVX512 is based on SSE
 * @brief The option judgment priority of AVX512 should be higher than that of AVX, 
 * because when the -mavx512vl/-mavx512f option is turned on during compilation, 
 * it will be backward compatible with AVX and SSE.
 */

#if defined(AVX512) || defined(AVX2) || defined(AVX)

#if !defined(SSE)
#error(SSE Not Defined! Now AVX? requires SSE support, please set SSE_ENABLER = yes.)
#endif

#if defined(AVX512)

#define AVX_LENGTH_float  16
#define AVX_LENGTH_double 8
#include "vectorized_blas_avx512.h"

#elif defined(AVX2) || defined(AVX)

#define AVX_LENGTH_float  8
#define AVX_LENGTH_double 4
#include "vectorized_blas_avx.h"

#endif // if AVX512 else AVX2
#endif // defined(AVX512) || defined(AVX2) || defined(AVX)

#endif // VECTORIZATION_CONTROL_H
