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
#ifndef SSE_LENGTH_float
#define SSE_LENGTH_float 4
#endif

#ifndef AVX_BLAS_VERSION_02
#define AVX_BLAS_VERSION_03
#endif

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


#if defined(AVX_BLAS_VERSION_03)

/**
 * @brief 
 * 
 * @param N 
 * @param A A[i][j]: A[i][j].re = A[0:lda][j], A[i][j].im = A[lda:2*lda][j]; matrix is stored in specific way: column-major, Are[]
 * @param lda number of rows;
 * @param B B[j], N complex elems 
 * @param C 
 */
static inline void simd_cgemv(const int N, const float *A, int lda, const float *B, float *C)
{
    int i, j;
    __m256i idxA = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    // here is a trick to keep the data order of result consist with _mm256_unpacklo/hi_ps

    __m256 A_re;
    __m256 A_im;
    __m256 B_re;
    __m256 B_im;
    __m256 C_re[lda / AVX_LENGTH_float];
    __m256 C_im[lda / AVX_LENGTH_float];


    // load Cre Cim
    for (i = 0; i < lda; i += AVX_LENGTH_float) {
        __m128 cvl0 = _mm_loadu_ps(&C[2 * i]); //idxe * 4 bytes
        __m128 cvl1 = _mm_loadu_ps(&C[2 * i + SSE_LENGTH_float]);
        __m128 cvh2 = _mm_loadu_ps(&C[2 * i + SSE_LENGTH_float * 2]);
        __m128 cvh3 = _mm_loadu_ps(&C[2 * i + SSE_LENGTH_float * 3]);
        cvl0        = _mm_permute_ps(cvl0, 0b11011000);
        cvl1        = _mm_permute_ps(cvl1, 0b11011000);
        cvh2        = _mm_permute_ps(cvh2, 0b11011000);
        cvh3        = _mm_permute_ps(cvh3, 0b11011000);

        __m128 cvlre = _mm_movelh_ps(cvl0, cvl1);
        __m128 cvhre = _mm_movelh_ps(cvh2, cvh3);
        __m128 cvlim = _mm_movehl_ps(cvl1, cvl0);
        __m128 cvhim = _mm_movehl_ps(cvh3, cvh2);

        C_re[i / AVX_LENGTH_float] = _mm256_set_m128(cvhre, cvlre);
        C_im[i / AVX_LENGTH_float] = _mm256_set_m128(cvhim, cvlim);
    }

    // apply cgemv with out-product method;
    for (j = 0; j < N; j++) {
        // load the j-th complex number in B
        B_re = _mm256_set1_ps(B[2 * j]);
        B_im = _mm256_set1_ps(B[2 * j + 1]);
        for (i = 0; i < lda; i += AVX_LENGTH_float) {
            A_re = _mm256_loadu_ps(A + 2 * j * lda + i);
            A_im = _mm256_loadu_ps(A + (2 * j + 1) * lda + i);

            // C += A*B
            C_re[i / AVX_LENGTH_float] = _mm256_fmsub_ps(A_re, B_re, _mm256_fmsub_ps(A_im, B_im, C_re[i / AVX_LENGTH_float]));
            C_im[i / AVX_LENGTH_float] = _mm256_fmadd_ps(A_im, B_re, _mm256_fmadd_ps(A_re, B_im, C_im[i / AVX_LENGTH_float]));
        }
    }

    // store to float *C; it has to do some unpack and permute operation; be carefull about A's data ordering.
    for (i = 0; i < lda; i += AVX_LENGTH_float) {
        __m128 rel = _mm256_castps256_ps128(C_re[i / AVX_LENGTH_float]);
        __m128 iml = _mm256_castps256_ps128(C_im[i / AVX_LENGTH_float]);
        __m128 vl0 = _mm_unpacklo_ps(rel, iml);
        __m128 vl1 = _mm_unpackhi_ps(rel, iml);
        _mm_storeu_ps(C + 2 * i, vl0);
        _mm_storeu_ps(C + 2 * i + SSE_LENGTH_float, vl1);

        __m128 reh = _mm256_extractf128_ps(C_re[i / AVX_LENGTH_float], 0b01);
        __m128 imh = _mm256_extractf128_ps(C_im[i / AVX_LENGTH_float], 0b01);
        __m128 vh0 = _mm_unpacklo_ps(reh, imh);
        __m128 vh1 = _mm_unpackhi_ps(reh, imh);
        _mm_storeu_ps(C + 2 * i + 2 * SSE_LENGTH_float, vh0);
        _mm_storeu_ps(C + 2 * i + 3 * SSE_LENGTH_float, vh1);
    }
}


static inline void simd_cgenmv(const int N, const float *A, int lda, const float *B, float *C)
{
    int i, j;

    __m256i idxe = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14); // addr of bytes
    __m256i idxo = _mm256_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15);

    __m256 A_re;
    __m256 A_im;
    __m256 B_re;
    __m256 B_im;
    __m256 C_re[lda / AVX_LENGTH_float];
    __m256 C_im[lda / AVX_LENGTH_float];

    // load Cre Cim
    for (i = 0; i < lda; i += AVX_LENGTH_float) {
        __m128 cvl0 = _mm_loadu_ps(&C[2 * i]); //idxe * 4 bytes
        __m128 cvl1 = _mm_loadu_ps(&C[2 * i + SSE_LENGTH_float]);
        __m128 cvh2 = _mm_loadu_ps(&C[2 * i + SSE_LENGTH_float * 2]);
        __m128 cvh3 = _mm_loadu_ps(&C[2 * i + SSE_LENGTH_float * 3]);
        cvl0        = _mm_permute_ps(cvl0, 0b11011000);
        cvl1        = _mm_permute_ps(cvl1, 0b11011000);
        cvh2        = _mm_permute_ps(cvh2, 0b11011000);
        cvh3        = _mm_permute_ps(cvh3, 0b11011000);

        __m128 cvlre = _mm_movelh_ps(cvl0, cvl1);
        __m128 cvhre = _mm_movelh_ps(cvh2, cvh3);
        __m128 cvlim = _mm_movehl_ps(cvl1, cvl0);
        __m128 cvhim = _mm_movehl_ps(cvh3, cvh2);

        C_re[i / AVX_LENGTH_float] = _mm256_set_m128(cvhre, cvlre);
        C_im[i / AVX_LENGTH_float] = _mm256_set_m128(cvhim, cvlim);
    }

    for (j = 0; j < N; j++) {
        B_re = _mm256_set1_ps(B[2 * j]);
        B_im = _mm256_set1_ps(B[2 * j + 1]);

        for (i = 0; i < lda; i += AVX_LENGTH_float) {
            A_re = _mm256_loadu_ps(A + 2 * j * lda + i);
            A_im = _mm256_loadu_ps(A + (2 * j + 1) * lda + i);

            // C -= A*B
            C_re[i / AVX_LENGTH_float] = _mm256_fnmadd_ps(A_re, B_re, _mm256_fmadd_ps(A_im, B_im, C_re[i / AVX_LENGTH_float]));
            C_im[i / AVX_LENGTH_float] = _mm256_fnmadd_ps(A_re, B_im, _mm256_fnmadd_ps(A_im, B_re, C_im[i / AVX_LENGTH_float]));
        }
    }

    // store to float *C; it has to do some unpack and permute operation; be carefull about A's data ordering.
    for (i = 0; i < lda; i += AVX_LENGTH_float) {
        __m128 rel = _mm256_castps256_ps128(C_re[i / AVX_LENGTH_float]);
        __m128 iml = _mm256_castps256_ps128(C_im[i / AVX_LENGTH_float]);
        __m128 vl0 = _mm_unpacklo_ps(rel, iml);
        __m128 vl1 = _mm_unpackhi_ps(rel, iml);
        _mm_storeu_ps(C + 2 * i, vl0);
        _mm_storeu_ps(C + 2 * i + SSE_LENGTH_float, vl1);

        __m128 reh = _mm256_extractf128_ps(C_re[i / AVX_LENGTH_float], 0b01);
        __m128 imh = _mm256_extractf128_ps(C_im[i / AVX_LENGTH_float], 0b01);
        __m128 vh0 = _mm_unpacklo_ps(reh, imh);
        __m128 vh1 = _mm_unpackhi_ps(reh, imh);
        _mm_storeu_ps(C + 2 * i + 2 * SSE_LENGTH_float, vh0);
        _mm_storeu_ps(C + 2 * i + 3 * SSE_LENGTH_float, vh1);
    }
}

#endif


#if defined(AVX_BLAS_VERSION_02)

static inline void simd_cgemv(const int N, const float *A, int lda, const float *B, float *C)
{
    int i, j;
    // here is a trick to keep the data order of result consist with _mm256_unpacklo/hi_ps
    __m256i idxe = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14); // addr of bytes
    __m256i idxo = _mm256_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15);

    __m256 A_re;
    __m256 A_im;
    __m256 B_re;
    __m256 B_im;
    __m256 C_re[lda / AVX_LENGTH_float];
    __m256 C_im[lda / AVX_LENGTH_float];


    // load Cre Cim
    for (i = 0; i < lda; i += AVX_LENGTH_float) {
        C_re[i / AVX_LENGTH_float] = _mm256_i32gather_ps(&C[2 * i], idxe, 4); //idxe * 4 bytes
        C_im[i / AVX_LENGTH_float] = _mm256_i32gather_ps(&C[2 * i], idxo, 4);
    }

    // apply cgemv with out-product method;
    for (j = 0; j < N; j++) {
        // load the j-th complex number in B
        B_re = _mm256_set1_ps(B[2 * j]);
        B_im = _mm256_set1_ps(B[2 * j + 1]);
        for (i = 0; i < lda; i += AVX_LENGTH_float) {
            A_re = _mm256_loadu_ps(A + 2 * j * lda + i);
            A_im = _mm256_loadu_ps(A + (2 * j + 1) * lda + i);

            // C += A*B
            C_re[i / AVX_LENGTH_float] = _mm256_fmsub_ps(A_re, B_re, _mm256_fmsub_ps(A_im, B_im, C_re[i / AVX_LENGTH_float]));
            C_im[i / AVX_LENGTH_float] = _mm256_fmadd_ps(A_im, B_re, _mm256_fmadd_ps(A_re, B_im, C_im[i / AVX_LENGTH_float]));
        }
    }

    // store to float *C; it has to do some unpack and permute operation; be carefull about A's data ordering.
    __m256i idxA = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    for (i = 0; i < lda; i += AVX_LENGTH_float) {

        A_re = _mm256_permutevar8x32_ps(C_re[i / AVX_LENGTH_float], idxA);
        A_im = _mm256_permutevar8x32_ps(C_im[i / AVX_LENGTH_float], idxA);
        B_re = _mm256_unpacklo_ps(A_re, A_im);
        B_im = _mm256_unpackhi_ps(A_re, A_im);
        _mm256_storeu_ps(C + 2 * i, B_re);
        _mm256_storeu_ps(C + 2 * i + AVX_LENGTH_float, B_im);
    }
}



static inline void simd_cgenmv(const int N, const float *A, int lda, const float *B, float *C)
{
    int i, j;

    __m256i idxe = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14); // addr of bytes
    __m256i idxo = _mm256_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15);

    __m256 A_re;
    __m256 A_im;
    __m256 B_re;
    __m256 B_im;
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
            A_re = _mm256_loadu_ps(A + 2 * j * lda + i);
            A_im = _mm256_loadu_ps(A + (2 * j + 1) * lda + i);

            // C -= A*B
            C_re[i / AVX_LENGTH_float] = _mm256_fnmadd_ps(A_re, B_re, _mm256_fmadd_ps(A_im, B_im, C_re[i / AVX_LENGTH_float]));
            C_im[i / AVX_LENGTH_float] = _mm256_fnmadd_ps(A_re, B_im, _mm256_fnmadd_ps(A_im, B_re, C_im[i / AVX_LENGTH_float]));
        }
    }

    __m256i idxA = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    for (i = 0; i < lda; i += AVX_LENGTH_float) {
        A_re = _mm256_permutevar8x32_ps(C_re[i / AVX_LENGTH_float], idxA);
        A_im = _mm256_permutevar8x32_ps(C_im[i / AVX_LENGTH_float], idxA);
        B_re = _mm256_unpacklo_ps(A_re, A_im);
        B_im = _mm256_unpackhi_ps(A_re, A_im);
        _mm256_storeu_ps(C + 2 * i, B_re);
        _mm256_storeu_ps(C + 2 * i + AVX_LENGTH_float, B_im);
    }
}

#endif 

#endif