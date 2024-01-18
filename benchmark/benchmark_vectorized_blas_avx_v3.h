/**
 * @file benchmark_vectorized_blas_avx_v2.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-01-16
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
#ifndef SSE_LENGTH_float
#define SSE_LENGTH_float 4
#endif



static inline void simd_cgemv_v3(const int N, const float *A, int lda, const float *B, float *C)
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


static inline void simd_cgenmv_v3(const int N, const float *A, int lda, const float *B, float *C)
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



#if 0
struct AvxComplex {
    __m256 real;
    __m256 imag;
};

inline void simd_complexload_complex2simd(__m256 *Y_re, __m256 *Y_im, const float *addr, const int N)
{
    __m256i idxe = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14); // addr of bytes
    __m256i idxo = _mm256_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15);
    for (int i = 0; i < N; i++) {
        Y_re[i] = _mm256_i32gather_ps(addr + i * 2 * AVX_LENGTH_float, idxe, 4); //idxe * 4 bytes
        Y_im[i] = _mm256_i32gather_ps(addr + i * 2 * AVX_LENGTH_float, idxo, 4);
    }
}

inline void simd_complexload_real2simd(__m256 *Y_re, __m256 *Y_im, const float *addre, const float *addim, const int N)
{
    for (int i = 0; i < N; i++) {
        Y_re[i] = _mm256_loadu_ps(addre + i * AVX_LENGTH_float); //idxe * 4 bytes
        Y_im[i] = _mm256_loadu_ps(addim + i * AVX_LENGTH_float);
    }
}

inline void simd_complexstore_simd2complex(float *addr, __m256 *Y_re, __m256 *Y_im, const int N)
{
    __m256i idxA = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    for (int i = 0; i < N; i++) {
        __m256 A_re = _mm256_permutevar8x32_ps(Y_re[i], idxA);
        __m256 A_im = _mm256_permutevar8x32_ps(Y_im[i], idxA);
        __m256 B_re = _mm256_unpacklo_ps(A_re, A_im);
        __m256 B_im = _mm256_unpackhi_ps(A_re, A_im);
        _mm256_storeu_ps(addr + 2 * i * AVX_LENGTH_float, B_re);
        _mm256_storeu_ps(addr + 2 * i * AVX_LENGTH_float + AVX_LENGTH_float, B_im);
    }
}

// Y += a * X;
inline void simd_kernal_caX(__m256 *Yre, __m256 *Yim, __m256 are, __m256 aim, __m256 *Xre, __m256 *Xim, const int N)
{
    for (int i = 0; i < N; i++) {
        Yre[i] = _mm256_fmsub_ps(are, Xre[i], _mm256_fmsub_ps(aim, Xim[i], Yre[i]));
        Yim[i] = _mm256_fmadd_ps(are, Xim[i], _mm256_fmadd_ps(aim, Xre[i], Yim[i]));
    }
}


static inline void simd_cgemv_v3(const int N, const float *A, int lda, const float *B, float *C)
{
    const int Num = lda / AVX_LENGTH_float; // column length counted by AVX SIMD
    __m256 A_re[Num];
    __m256 A_im[Num];
    __m256 B_re;
    __m256 B_im;
    __m256 C_re[Num];
    __m256 C_im[Num];


    // load Cre Cim
    simd_complexload_complex2simd(C_re, C_im, C, Num);

    // apply cgemv with out-product method;
    for (int j = 0; j < N; j++) {
        // load the j-th complex number in B
        B_re = _mm256_set1_ps(B[2 * j]);
        B_im = _mm256_set1_ps(B[2 * j + 1]);
        simd_complexload_real2simd(A_re, A_im, A + 2 * j * lda, A + (2 * j + 1) * lda, Num);
        // for (int i = 0; i < Num; i++) {
        //     C_re[i] = _mm256_fmsub_ps(B_re, A_re[i], _mm256_fmsub_ps(B_im, A_im[i], C_re[i]));
        //     C_im[i] = _mm256_fmadd_ps(B_im, A_re[i], _mm256_fmadd_ps(B_re, A_im[i], C_im[i]));
        // }
        simd_kernal_caX(C_re, C_im, B_re, B_im, A_re, A_im, Num);
    }


    // store to float *C; it has to do some unpack and permute operation; be carefull about A's data ordering.
    simd_complexstore_simd2complex(C, C_re, C_im, Num);
}
#endif