/**
 * @file benchmark_vectorized_blas_avx.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-01-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <stdio.h>
#include <immintrin.h>

#ifndef AVX_LENGTH_float
#define AVX_LENGTH_float 8
#endif

static inline void simd_cgemv(const int N, const float *A, int lda, const float *B, float *C)
{
    int i, j;
    // here is a trick to keep the data order of result consist with _mm256_unpacklo/hi_ps
    __m256i idxe = _mm256_setr_epi32(0, 2, 8, 10, 4, 6, 12, 14); // addr of bytes
    __m256i idxo = _mm256_setr_epi32(1, 3, 9, 11, 5, 7, 13, 15);
    __m256i idxA = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);

    // __m256i idxe = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14); // addr of bytes
    // __m256i idxo = _mm256_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15);

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
            // A_re = _mm256_loadu_ps(A + 2 * j * lda + i);
            // A_im = _mm256_loadu_ps(A + (2 * j + 1) * lda + i);
            // use gather with fixed order
            A_re = _mm256_i32gather_ps(A + 2 * j * lda + i, idxA, 4);
            A_im = _mm256_i32gather_ps(A + (2 * j + 1) * lda + i, idxA, 4);

            // C += A*B
            C_re[i / AVX_LENGTH_float] = _mm256_fmsub_ps(A_re, B_re, _mm256_fmsub_ps(A_im, B_im, C_re[i / AVX_LENGTH_float]));
            C_im[i / AVX_LENGTH_float] = _mm256_fmadd_ps(A_im, B_re, _mm256_fmadd_ps(A_re, B_im, C_im[i / AVX_LENGTH_float]));
        }
    }

    // store to float *C
    for (i = 0; i < lda; i += AVX_LENGTH_float) {
        __m256 Ci_lo = _mm256_unpacklo_ps(C_re[i / AVX_LENGTH_float], C_im[i / AVX_LENGTH_float]);
        __m256 Ci_hi = _mm256_unpackhi_ps(C_re[i / AVX_LENGTH_float], C_im[i / AVX_LENGTH_float]);
        _mm256_storeu_ps(C + 2 * i, Ci_lo);
        _mm256_storeu_ps(C + 2 * i + AVX_LENGTH_float, Ci_hi);
    }
}


static inline void simd_cgenmv(const int N, const float *A, int lda, const float *B, float *C)
{
    int i, j;

    __m256i idxe = _mm256_setr_epi32(0, 2, 8, 10, 4, 6, 12, 14); // addr of bytes
    __m256i idxo = _mm256_setr_epi32(1, 3, 9, 11, 5, 7, 13, 15);
    __m256i idxA = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);

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
            A_re = _mm256_i32gather_ps(A + 2 * j * lda + i, idxA, 4);
            A_im = _mm256_i32gather_ps(A + (2 * j + 1) * lda + i, idxA, 4);

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
