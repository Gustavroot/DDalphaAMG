/*
 * Copyright (C) 2016, Matthias Rottmann, Artur Strebel, Gustavo Ramirez, Simon
 * Heybrock, Simone Bacchio, Bjoern Leder, Issaku Kanamori.
 *
 * This file is part of the DDalphaAMG solver library.
 *
 * The DDalphaAMG solver library is free software: you can redistribute it
 * and/or modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the License,
 * or (at your option) any later version.
 *
 * The DDalphaAMG solver library is distributed in the hope that it will be
 * useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *
 * You should have received a copy of the GNU General Public License
 * along with the DDalphaAMG solver library. If not, see
 * http://www.gnu.org/licenses/.
 *
 */

#ifndef VECTORIZED_BLAS_AVX_H
#define VECTORIZED_BLAS_AVX_H

#include <immintrin.h>

#ifndef AVX_LENGTH_float
#define AVX_LENGTH_float 8
#endif
#ifndef AVX_LENGTH_double
#define AVX_LENGTH_double 4
#endif

#ifdef AVX_BLAS_float

static inline void simd_cgem_inverse(const int N, float *A_inverse, float *A, int lda)
{
    // generate LU decomp in A

    int i, j, k;

    complex_float alpha;

    complex_float tmpA[N * N];
    complex_float tmpA_inverse[N * N];

    for (j = 0; j < N; j++) {
        for (i = 0; i < N; i++) {
// FUNCTION BROKEN! _Complex_I is not supported in C++
#ifndef __cplusplus
            tmpA[i + N * j] = A[2 * j * lda + i] + _Complex_I * A[(2 * j + 1) * lda + i];
#endif
        }
    }

    // LU decomp in A
    for (k = 0; k < N - 1; k++) {
        for (i = k + 1; i < N; i++) {
            // alpha = A_ik/A_kk
            alpha           = tmpA[i + k * N] / tmpA[k + k * N];
            tmpA[i + k * N] = alpha;
            for (j = k + 1; j < N; j++) {
                // A_ij = A_ij - alpha * A_kj
                tmpA[i + j * N] -= alpha * tmpA[k + j * N];
            }
        }
    }

    complex_float b[N];
    complex_float *x;

    for (k = 0; k < N; k++) { b[k] = 0; }

    for (k = 0; k < N; k++) {
        x    = tmpA_inverse + k * N;
        b[k] = 1;
        if (k > 0) b[k - 1] = 0;

        for (i = 0; i < N; i++) {
            x[i] = b[i];
            // x_i = x_i - A_ij + x_j
            for (j = 0; j < i; j++) { x[i] = x[i] - tmpA[i + j * N] * x[j]; }
        }

        for (i = N - 1; i >= 0; i--) {
            // x_i = x_i - A_ij * x_j
            for (j = i + 1; j < N; j++) { x[i] = x[i] - tmpA[i + j * N] * x[j]; }
            // x_i = x_i / A_ii
            x[i] = x[i] / tmpA[i + i * N];
        }
    } // k

    for (j = 0; j < N; j++) {
        for (i = 0; i < N; i++) {
            A_inverse[i + 2 * j * lda]       = creal(tmpA_inverse[i + j * N]);
            A_inverse[i + (2 * j + 1) * lda] = cimag(tmpA_inverse[i + j * N]);
        }
        for (i = N; i < lda; i++) {
            A_inverse[i + 2 * j * lda]       = 0.0;
            A_inverse[i + (2 * j + 1) * lda] = 0.0;
        }
    }
}

static inline void simd_cgemv(const int N, const OPERATOR_TYPE_float *A, int lda, const float *B, float *C)
{
    int i, j;
    __m256i idxe = _mm256_setr_epi32(0, 2, 8, 10, 4, 6, 12, 14); // addr of bytes
    __m256i idxo = _mm256_setr_epi32(1, 3, 9, 11, 5, 7, 13, 15);
    __m256 A_re;
    __m256 A_im;
    __m256 B_re;
    __m256 B_im;
    __m256 C_re[lda / AVX_LENGTH_float];
    __m256 C_im[lda / AVX_LENGTH_float];

    // deinterleaved load
    for (i = 0; i < lda; i += AVX_LENGTH_float) {
        C_re[i / AVX_LENGTH_float] = _mm256_i32gather_ps(&C[2 * i], idxe, 4); //idxe * 4 bytes
        C_im[i / AVX_LENGTH_float] = _mm256_i32gather_ps(&C[2 * i], idxo, 4);
    }

    for (j = 0; j < N; j++) {
        // load the j-th complex number in B
        B_re = _mm256_set1_ps(B[2 * j]);
        B_im = _mm256_set1_ps(B[2 * j + 1]);
        for (i = 0; i < lda; i += AVX_LENGTH_float) {
#if defined(DEBUG)
            printf("======================  loadu ... \n");
            if (A == NULL) { printf("======================  A is NULL\n"); }
#endif
            A_re = _mm256_loadu_ps(A + 2 * j * lda + i);
            A_im = _mm256_loadu_ps(A + (2 * j + 1) * lda + i);

            // C += A*B
            C_re[i / AVX_LENGTH_float] = _mm256_fmsub_ps(A_re, B_re, _mm256_fmsub_ps(A_im, B_im, C_re[i / AVX_LENGTH_float]));
            C_im[i / AVX_LENGTH_float] = _mm256_fmadd_ps(A_im, B_re, _mm256_fmadd_ps(A_re, A_im, C_im[i / AVX_LENGTH_float]));
        }
    }


    // interleaves real and imaginary parts and stores the resulting complex numbers in C
    for (i = 0; i < lda; i += AVX_LENGTH_float) {
        // #ifdef AVX512
        //         _mm256_i32scatter_ps(C + 2 * i, idxe, C_re[i / AVX_LENGTH_float], 4);
        //         _mm256_i32scatter_ps(C + 2 * i, idxo, C_im[i / AVX_LENGTH_float], 4);
        // #else
        __m256 Ci_lo = _mm256_unpacklo_ps(C_re[i / AVX_LENGTH_float], C_im[i / AVX_LENGTH_float]);
        __m256 Ci_hi = _mm256_unpackhi_ps(C_re[i / AVX_LENGTH_float], C_im[i / AVX_LENGTH_float]);
        _mm256_storeu_ps(C + 2 * i, Ci_lo);
        _mm256_storeu_ps(C + 2 * i + AVX_LENGTH_float, Ci_hi);
        // #endif
    }
}

static inline void simd_cgenmv(const int N, const OPERATOR_TYPE_float *A, int lda, const float *B, float *C)
{
    int i, j;

    __m256i idxe = _mm256_setr_epi32(0, 2, 8, 10, 4, 6, 12, 14); // addr of bytes
    __m256i idxo = _mm256_setr_epi32(1, 3, 9, 11, 5, 7, 13, 15);

    __m256 A_re, A_im, B_re, B_im;
    __m256 C_re[lda / AVX_LENGTH_float];
    __m256 C_im[lda / AVX_LENGTH_float];

    for (i = 0; i < lda; i += AVX_LENGTH_float) {
        C_re[i / AVX_LENGTH_float] = _mm256_i32gather_ps(&C[2 * i], idxe, 4);
        C_im[i / AVX_LENGTH_float] = _mm256_i32gather_ps(&C[2 * i], idxo, 4);
    }

    for (j = 0; j < N; j++) {
        B_re = _mm256_set1_ps(B[2 * j]);
        B_im = _mm256_set1_ps(B[2 * j + 1]);

        for (i = 0; i < lda; i += AVX_LENGTH_float) {
            A_re = _mm256_load_ps(A + 2 * j * lda + i);
            A_im = _mm256_load_ps(A + (2 * j + 1) * lda + i);

            // C -= A*B
            C_re[i / AVX_LENGTH_float] = _mm256_fnmadd_ps(A_re, B_re, _mm256_fmadd_ps(A_im, B_im, C_re[i / AVX_LENGTH_float]));
            C_im[i / AVX_LENGTH_float] = _mm256_fnmadd_ps(A_re, B_im, _mm256_fnmadd_ps(A_im, B_re, C_im[i / AVX_LENGTH_float]));
        }
    }

    for (i = 0; i < lda; i += AVX_LENGTH_float) {
        __m256 Ci_lo = _mm256_unpacklo_ps(C_re[i / AVX_LENGTH_float], C_im[i / AVX_LENGTH_float]);
        __m256 Ci_hi = _mm256_unpackhi_ps(C_re[i / AVX_LENGTH_float], C_im[i / AVX_LENGTH_float]);
        _mm256_storeu_ps(C + 2 * i, Ci_lo);
        _mm256_storeu_ps(C + 2 * i + AVX_LENGTH_float, Ci_hi);
    }
}

#else

static inline void simd_cgem_inverse(const int N, float *A_inverse, float *A, int lda)
{
    printf("NO SIMD simd_cgem_inverse(), file: %s, line: %d\n", __FILE__, __LINE__);
}

static inline void simd_cgemv(const int N, const OPERATOR_TYPE_float *A, int lda, const float *B, float *C)
{
    printf("NO SIMD simd_cgemv(), file: %s, line: %d\n", __FILE__, __LINE__);
}

static inline void simd_cgenmv(const int N, const OPERATOR_TYPE_float *A, int lda, const float *B, float *C)
{
    printf("NO SIMD simd_cgenmv(), file: %s, line: %d\n", __FILE__, __LINE__);
}

#endif

#endif