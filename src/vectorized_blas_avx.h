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

#if defined(AVX) || defined(AVX2)
#define AVX_LENGTH_float  8
#define AVX_LENGTH_double 4
#endif

#define RELEASE

static inline void avx_cgem_inverse(const int N, float *A_inverse, float *A, int lda)
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


#if defined(RELEASE)

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