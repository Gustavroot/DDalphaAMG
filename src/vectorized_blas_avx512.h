/**
 * @file vectorized_blas_avx512.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-01-25
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef VECTORIZED_BLAS_AVX512_H
#define VECTORIZED_BLAS_AVX512_H

#include <immintrin.h>

#if !(AVX_LENGTH_float == 16)
#error(AVX512 needs AVX_LENGTH_float==16)
#endif

#define AVX_LENGTH_float 16
#define SSE_LENGTH_float 4

#define RELEASE

#if defined(RELEASE)

static inline void avx512_cgem_inverse(const int N, float *A_inverse, float *A, int lda)
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


// #if defined(AVX_BLAS_VERSION_03)
// #endif

// #if defined(AVX512_BLAS_VERSION_02)

static inline void avx512_cgemv(const int N, const float *A, int lda, const float *B, float *C)
{
    __m512i idxe = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30); // addr of bytes
    __m512i idxo = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);

    __m512 A_re;
    __m512 A_im;
    __m512 B_re;
    __m512 B_im;
    __m512 C_re[lda / AVX_LENGTH_float];
    __m512 C_im[lda / AVX_LENGTH_float];

    // deinterleaved load
    for (int i = 0; i < lda; i += AVX_LENGTH_float) {
        C_re[i / AVX_LENGTH_float] = _mm512_i32gather_ps(idxe, &C[2 * i], 4); //idxe * 4 bytes
        C_im[i / AVX_LENGTH_float] = _mm512_i32gather_ps(idxo, &C[2 * i], 4);
    }

    // apply cgemv with out-product method;
    for (int j = 0; j < N; j++) {
        B_re = _mm512_set1_ps(B[2 * j]);
        B_im = _mm512_set1_ps(B[2 * j + 1]);
        for (int i = 0; i < lda; i += AVX_LENGTH_float) {
            A_re                       = _mm512_loadu_ps(A + 2 * j * lda + i);
            A_im                       = _mm512_loadu_ps(A + (2 * j + 1) * lda + i);
            // C += A*B
            C_re[i / AVX_LENGTH_float] = _mm512_fmsub_ps(A_re, B_re, _mm512_fmsub_ps(A_im, B_im, C_re[i / AVX_LENGTH_float]));
            C_im[i / AVX_LENGTH_float] = _mm512_fmadd_ps(A_im, B_re, _mm512_fmadd_ps(A_re, B_im, C_im[i / AVX_LENGTH_float]));
        }
    }

// store to float *C
#if defined(RELEASE)
    for (int i = 0; i < lda; i += AVX_LENGTH_float) {
        _mm512_i32scatter_ps(&C[2 * i], idxe, C_re[i / AVX_LENGTH_float], 4);
        _mm512_i32scatter_ps(&C[2 * i], idxo, C_im[i / AVX_LENGTH_float], 4);
    }
#elif defined(DEVEL)
    for (int i = 0; i < lda / AVX_LENGTH_float; i++) {
        float *pC = C + i * 2 * AVX_LENGTH_float;
        __m128 re, im;
        re = _mm512_extractf32x4_ps(C_re[i], 0b00);
        im = _mm512_extractf32x4_ps(C_im[i], 0b00);
        _mm_store_ps(pC, _mm_unpacklo_ps(re, im));
        _mm_store_ps(pC + SSE_LENGTH_float, _mm_unpackhi_ps(re, im));

        pC = pC + 2 * SSE_LENGTH_float;
        re = _mm512_extractf32x4_ps(C_re[i], 0b01);
        im = _mm512_extractf32x4_ps(C_im[i], 0b01);
        _mm_store_ps(pC, _mm_unpacklo_ps(re, im));
        _mm_store_ps(pC + SSE_LENGTH_float, _mm_unpackhi_ps(re, im));

        pC = pC + 2 * SSE_LENGTH_float;
        re = _mm512_extractf32x4_ps(C_re[i], 0b10);
        im = _mm512_extractf32x4_ps(C_im[i], 0b10);
        _mm_store_ps(pC, _mm_unpacklo_ps(re, im));
        _mm_store_ps(pC + SSE_LENGTH_float, _mm_unpackhi_ps(re, im));

        pC = pC + 2 * SSE_LENGTH_float;
        re = _mm512_extractf32x4_ps(C_re[i], 0b11);
        im = _mm512_extractf32x4_ps(C_im[i], 0b11);
        _mm_store_ps(pC, _mm_unpacklo_ps(re, im));
        _mm_store_ps(pC + SSE_LENGTH_float, _mm_unpackhi_ps(re, im));
    }
#endif
}



static inline void avx512_cgenmv(const int N, const float *A, int lda, const float *B, float *C)
{
    __m512i idxe = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30); // addr of bytes
    __m512i idxo = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);

    __m512 A_re;
    __m512 A_im;
    __m512 B_re;
    __m512 B_im;
    __m512 C_re[lda / AVX_LENGTH_float];
    __m512 C_im[lda / AVX_LENGTH_float];

    // deinterleaved load
    for (int i = 0; i < lda; i += AVX_LENGTH_float) {
        C_re[i / AVX_LENGTH_float] = _mm512_i32gather_ps(idxe, &C[2 * i], 4); //idxe * 4 bytes
        C_im[i / AVX_LENGTH_float] = _mm512_i32gather_ps(idxo, &C[2 * i], 4);
    }

    for (int j = 0; j < N; j++) {
        B_re = _mm512_set1_ps(B[2 * j]);
        B_im = _mm512_set1_ps(B[2 * j + 1]);

        for (int i = 0; i < lda; i += AVX_LENGTH_float) {
            A_re = _mm512_loadu_ps(A + 2 * j * lda + i);
            A_im = _mm512_loadu_ps(A + (2 * j + 1) * lda + i);

            // C -= A*B
            C_re[i / AVX_LENGTH_float] = _mm512_fnmadd_ps(A_re, B_re, _mm512_fmadd_ps(A_im, B_im, C_re[i / AVX_LENGTH_float]));
            C_im[i / AVX_LENGTH_float] = _mm512_fnmadd_ps(A_re, B_im, _mm512_fnmadd_ps(A_im, B_re, C_im[i / AVX_LENGTH_float]));
        }
    }

    // store to float *C
    for (int i = 0; i < lda; i += AVX_LENGTH_float) {
        _mm512_i32scatter_ps(&C[2 * i], idxe, C_re[i / AVX_LENGTH_float], 4);
        _mm512_i32scatter_ps(&C[2 * i], idxo, C_im[i / AVX_LENGTH_float], 4);
    }
}

#endif // RELEASE

#endif // end define file