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

#if !defined(AVX_LENGTH_float)
#define AVX_LENGTH_float  16
#define AVX_LENGTH_double 8
#elif !(AVX_LENGTH_float == 16)
#error(AVX512 needs AVX_LENGTH_float == 16)
#endif

#define RELEASE

#if defined(RELEASE)

// static inline void avx512_cgem_inverse(const int N, float *A_inverse, float *A, int lda);

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