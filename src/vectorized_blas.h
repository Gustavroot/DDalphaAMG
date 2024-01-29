/*
 * Copyright (C) 2016, Matthias Rottmann, Artur Strebel, Gustavo Ramirez, Simon Heybrock, Simone Bacchio, Bjoern Leder, Issaku Kanamori.
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

#ifndef VECTORIZED_BLAS_H
#define VECTORIZED_BLAS_H

// Macro Flag to switch simd version


/**
 * @brief A: op, A is a lda-N matrix, stored of column-major with discrete real and imag in differen column.
 *   eg: Are_ij = A+2*j*lda+i, Aim_ij = A+(2*j+1)*lda+i
 *  i = [0,lda), j = [0,N)
 *             re im   re im   re im   re im   re im
 *           | r | i | r | i | r | i | r | i | r | i |
 *           | r | i | r | i | r | i | r | i | r | i |
 *           | r | i | r | i | r | i | r | i | r | i |
 * A[i][j] = | r | i | r | i | r | i | r | i | r | i |
 *           | r | i | r | i | r | i | r | i | r | i |
 *           | r | i | r | i | r | i | r | i | r | i | 
 *           | r | i | r | i | r | i | r | i | r | i |
 *           | r | i | r | i | r | i | r | i | r | i | 
 * 
 */

/**
 * @brief B: complex vector, with re and im interleaved, e.g: Bre_j = B[2*j]
 * @brief C: complex vector, with re and im interleaved, e.g: Cre_j = C[2*j]
 * 
 */



#ifdef AVX_BLAS_float

#ifdef AVX512

#include "vectorized_blas_avx512.h"
#define simd_cgemv        avx512_cgemv
#define simd_cgenmv       avx512_cgenmv
#define simd_cgem_inverse avx512_cgem_inverse

#else

#ifdef AVX2

#include "vectorized_blas_avx.h"
#define simd_cgemv        avx_cgemv
#define simd_cgenmv       avx_cgenmv
#define simd_cgem_inverse avx_cgem_inverse

#endif

#endif

// BLAS naming convention: LDA = leading dimension of A

/**
 * @brief  C=A*B+C
 * 
 * @param N 
 * @param A 
 * @param lda 
 * @param B 
 * @param C 
 */
static inline void vectorized_cgemv(const int N, const OPERATOR_TYPE_float *A, int lda, const float *B, float *C)
{
    simd_cgemv(N, A, lda, B, C);
}

/** C=-A*B+C
 * @brief 
 * 
 * @param N 
 * @param A 
 * @param lda 
 * @param B 
 * @param C 
 */
static inline void vectorized_cgenmv(const int N, const OPERATOR_TYPE_float *A, int lda, const float *B, float *C)
{
    simd_cgenmv(N, A, lda, B, C);
}

/**
 * @brief A_inverse
 * 
 * @param N 
 * @param A_inverse 
 * @param A 
 * @param lda 
 */
static inline void vectorized_cgem_inverse(const int N, OPERATOR_TYPE_float *A_inverse, OPERATOR_TYPE_float *A, int lda)
{
    simd_cgem_inverse(N, A_inverse, A, lda);
}

#endif // endif AVX_LENGTH_float

#endif // VECTORIZED_BLAS_H
