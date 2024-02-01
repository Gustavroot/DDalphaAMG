/**
 * @file benchmark_vectorized_blas_sse.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-01-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include "benchmark_complex_float_intrinsic_sse.h"
#include "benchmark_float_intrinsic_sse.h"

#define SSE_LENGTH_float 4

static inline void sse_cgemv(const int N, const float *A, int lda, const float *B, float *C)
{
    int i, j;

    __m128 A_re;
    __m128 A_im;
    __m128 B_re;
    __m128 B_im;
    __m128 C_re[lda / SSE_LENGTH_float];
    __m128 C_im[lda / SSE_LENGTH_float];

    // deinterleaved load
    for (i = 0; i < lda; i += SSE_LENGTH_float) {
        C_re[i / SSE_LENGTH_float] = _mm_setr_ps(C[2 * i], C[2 * i + 2], C[2 * i + 4], C[2 * i + 6]);
        C_im[i / SSE_LENGTH_float] = _mm_setr_ps(C[2 * i + 1], C[2 * i + 3], C[2 * i + 5], C[2 * i + 7]);
    }

    for (j = 0; j < N; j++) {
        // load the j-th complex number in B
        B_re = _mm_set1_ps(B[2 * j]);
        B_im = _mm_set1_ps(B[2 * j + 1]);

        for (i = 0; i < lda; i += SSE_LENGTH_float) {
            A_re = _mm_load_ps(A + 2 * j * lda + i);
            A_im = _mm_load_ps(A + (2 * j + 1) * lda + i);

            // C += A*B
            cfmadd(A_re, A_im, B_re, B_im, &(C_re[i / SSE_LENGTH_float]), &(C_im[i / SSE_LENGTH_float]));
        }
    }

    // interleaves real and imaginary parts and stores the resulting complex numbers in C
    for (i = 0; i < lda; i += SSE_LENGTH_float) {
        A_re = _mm_unpacklo_ps(C_re[i / SSE_LENGTH_float], C_im[i / SSE_LENGTH_float]);
        A_im = _mm_unpackhi_ps(C_re[i / SSE_LENGTH_float], C_im[i / SSE_LENGTH_float]);
        _mm_store_ps(C + 2 * i, A_re);
        _mm_store_ps(C + 2 * i + SSE_LENGTH_float, A_im);
    }
}

static inline void sse_cgenmv(const int N, const float *A, int lda, const float *B, float *C)
{
    int i, j;

    __m128 A_re;
    __m128 A_im;
    __m128 B_re;
    __m128 B_im;
    __m128 C_re[lda / SSE_LENGTH_float];
    __m128 C_im[lda / SSE_LENGTH_float];

    for (i = 0; i < lda; i += SSE_LENGTH_float) {
        C_re[i / SSE_LENGTH_float] = _mm_setr_ps(C[2 * i], C[2 * i + 2], C[2 * i + 4], C[2 * i + 6]);
        C_im[i / SSE_LENGTH_float] = _mm_setr_ps(C[2 * i + 1], C[2 * i + 3], C[2 * i + 5], C[2 * i + 7]);
    }

    for (j = 0; j < N; j++) {

        B_re = _mm_set1_ps(B[2 * j]);
        B_im = _mm_set1_ps(B[2 * j + 1]);

        for (i = 0; i < lda; i += SSE_LENGTH_float) {
            A_re = _mm_load_ps(A + 2 * j * lda + i);
            A_im = _mm_load_ps(A + (2 * j + 1) * lda + i);

            // C -= A*B
            cfnmadd(A_re, A_im, B_re, B_im, &(C_re[i / SSE_LENGTH_float]), &(C_im[i / SSE_LENGTH_float]));
        }
    }

    for (i = 0; i < lda; i += SSE_LENGTH_float) {
        A_re = _mm_unpacklo_ps(C_re[i / SSE_LENGTH_float], C_im[i / SSE_LENGTH_float]);
        A_im = _mm_unpackhi_ps(C_re[i / SSE_LENGTH_float], C_im[i / SSE_LENGTH_float]);
        _mm_store_ps(C + 2 * i, A_re);
        _mm_store_ps(C + 2 * i + SSE_LENGTH_float, A_im);
    }
}
