/**
 * @file vectorized_blas_avx.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-01-25
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef VECTORIZED_BLAS_AVX_H
#define VECTORIZED_BLAS_AVX_H

#include <immintrin.h>

#if !defined(AVX_LENGTH_float)
#define AVX_LENGTH_float  8
#define AVX_LENGTH_double 4
#elif !(AVX_LENGTH_float == 8)
#error(AVX/AVX2 needs AVX_LENGTH_float==8)
#endif

#define RELEASE

#if defined(RELEASE)

// static inline void avx_cgem_inverse(const int N, float *A_inverse, float *A, int lda);

/**
 * @brief 
 * 
 * @param N 
 * @param A A[i][j]: A[i][j].re = A[0:lda][j], A[i][j].im = A[lda:2*lda][j]; matrix is stored in specific way: column-major, Are[]
 * @param lda number of rows;
 * @param B B[j], N complex elems 
 * @param C 
 */
static inline void avx_cgemv(const int N, const float *A, int lda, const float *B, float *C)
{
    __m256i idxe = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14); // addr of bytes
    __m256i idxo = _mm256_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15);


    __m256 C_re[lda / AVX_LENGTH_float];
    __m256 C_im[lda / AVX_LENGTH_float];


    // load Cre Cim
    for (int i = 0; i < lda; i += AVX_LENGTH_float) {
        C_re[i / AVX_LENGTH_float] = _mm256_i32gather_ps(&C[2 * i], idxe, 4); //idxe * 4 bytes
        C_im[i / AVX_LENGTH_float] = _mm256_i32gather_ps(&C[2 * i], idxo, 4);
    }

    // apply cgemv with out-product method;
    for (int j = 0; j < N; j++) {
        __m256 A_re, A_im;
        __m256 B_re, B_im;
        // load the j-th complex number in B
        B_re = _mm256_set1_ps(B[2 * j]);
        B_im = _mm256_set1_ps(B[2 * j + 1]);
        for (int i = 0; i < lda; i += AVX_LENGTH_float) {
            A_re = _mm256_loadu_ps(A + 2 * j * lda + i);
            A_im = _mm256_loadu_ps(A + (2 * j + 1) * lda + i);

            // C += A*B
            C_re[i / AVX_LENGTH_float] = _mm256_fmsub_ps(A_re, B_re, _mm256_fmsub_ps(A_im, B_im, C_re[i / AVX_LENGTH_float]));
            C_im[i / AVX_LENGTH_float] = _mm256_fmadd_ps(A_im, B_re, _mm256_fmadd_ps(A_re, B_im, C_im[i / AVX_LENGTH_float]));
        }
    }

    // store to float *C; it has to do some unpack and permute operation; be carefull about A's data ordering.
    __m256i idxA = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    for (int i = 0; i < lda; i += AVX_LENGTH_float) {
        __m256 A_re, A_im;
        __m256 B_re, B_im;
        A_re = _mm256_permutevar8x32_ps(C_re[i / AVX_LENGTH_float], idxA);
        A_im = _mm256_permutevar8x32_ps(C_im[i / AVX_LENGTH_float], idxA);
        B_re = _mm256_unpacklo_ps(A_re, A_im);
        B_im = _mm256_unpackhi_ps(A_re, A_im);
        _mm256_storeu_ps(C + 2 * i, B_re);
        _mm256_storeu_ps(C + 2 * i + AVX_LENGTH_float, B_im);
    }
}


/**
 * @brief 
 * 
 * @param N 
 * @param A 
 * @param lda 
 * @param B 
 * @param C 
 */
static inline void avx_cgenmv(const int N, const float *A, int lda, const float *B, float *C)
{
    __m256i idxe = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14); // addr of bytes
    __m256i idxo = _mm256_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15);

    __m256 A_re;
    __m256 A_im;
    __m256 B_re;
    __m256 B_im;
    __m256 C_re[lda / AVX_LENGTH_float];
    __m256 C_im[lda / AVX_LENGTH_float];

    for (int i = 0; i < lda; i += AVX_LENGTH_float) {
        C_re[i / AVX_LENGTH_float] = _mm256_i32gather_ps(&C[2 * i], idxe, 4);
        C_im[i / AVX_LENGTH_float] = _mm256_i32gather_ps(&C[2 * i], idxo, 4);
    }

    for (int j = 0; j < N; j++) {
        B_re = _mm256_set1_ps(B[2 * j]);
        B_im = _mm256_set1_ps(B[2 * j + 1]);

        for (int i = 0; i < lda; i += AVX_LENGTH_float) {
            A_re = _mm256_loadu_ps(A + 2 * j * lda + i);
            A_im = _mm256_loadu_ps(A + (2 * j + 1) * lda + i);

            // C -= A*B
            C_re[i / AVX_LENGTH_float] = _mm256_fnmadd_ps(A_re, B_re, _mm256_fmadd_ps(A_im, B_im, C_re[i / AVX_LENGTH_float]));
            C_im[i / AVX_LENGTH_float] = _mm256_fnmadd_ps(A_re, B_im, _mm256_fnmadd_ps(A_im, B_re, C_im[i / AVX_LENGTH_float]));
        }
    }

    __m256i idxA = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    for (int i = 0; i < lda; i += AVX_LENGTH_float) {
        A_re = _mm256_permutevar8x32_ps(C_re[i / AVX_LENGTH_float], idxA);
        A_im = _mm256_permutevar8x32_ps(C_im[i / AVX_LENGTH_float], idxA);
        B_re = _mm256_unpacklo_ps(A_re, A_im);
        B_im = _mm256_unpackhi_ps(A_re, A_im);
        _mm256_storeu_ps(C + 2 * i, B_re);
        _mm256_storeu_ps(C + 2 * i + AVX_LENGTH_float, B_im);
    }
}




#endif // RELEASE

// #if defined(DEVEL) || !defined(RELEASE)
// #endif // defined(DEVEL) || !defined(RELEASE)

#endif // VECTORIZED_BLAS_AVX_H