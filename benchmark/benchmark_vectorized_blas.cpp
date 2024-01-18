/**
 * @file benchmark_vectorized_blas.c
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-01-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "timer.h"
extern "C"
{
#include "benchmark_vectorized_blas_sse.h"
#include "benchmark_vectorized_blas_avx.h"
#include "benchmark_vectorized_blas_avx_v2.h"
#include "benchmark_vectorized_blas_avx_v3.h"
}

#ifndef LOOP
#define LOOP 1000
#endif


void (*avx_cgemv)(int, const float *, int, const float *, float *);
void (*avx_cgenmv)(int, const float *, int, const float *, float *);


int main(int argc, char const *argv[])
{
#if defined(VERSION_02)
    avx_cgemv  = simd_cgemv_v2;
    avx_cgenmv = simd_cgenmv_v2;
#elif defined(VERSION_03)
    avx_cgemv  = simd_cgemv_v3;
    avx_cgenmv = simd_cgenmv_v3;
#else
    avx_cgemv  = simd_cgemv;
    avx_cgenmv = simd_cgenmv;
#endif

    timer watch;

    watch.reset();
    const int Loop = LOOP;
    // A: float A[lda][N]
    int lda        = 48;
    int N          = 48;

    float sum_diff_norm2 = 0.0;
    float diff_re        = 0.0;
    float diff_im        = 0.0;

    double time_sse = 0.0;
    double time_avx = 0.0;

    float *A    = (float *) malloc(lda * N * 2 * sizeof(float));
    float *B    = (float *) malloc(N * 2 * sizeof(float));
    float *Csse = (float *) malloc(lda * 2 * sizeof(float));
    float *Cavx = (float *) malloc(lda * 2 * sizeof(float));

    for (size_t i = 0; i < lda * N * 2; i++) { A[i] = (float) rand() / (float) RAND_MAX; }
    for (size_t i = 0; i < N * 2; i++) { B[i] = (float) rand() / (float) RAND_MAX; }
    for (size_t i = 0; i < lda * 2; i++) { Csse[i] = (float) rand() / (float) RAND_MAX; }
    for (size_t i = 0; i < lda * 2; i++) { Cavx[i] = Csse[i]; }

    for (size_t i = 0; i < N; i++) {
        diff_re = Csse[2 * i] - Cavx[2 * i];
        diff_im = Csse[2 * i + 1] - Cavx[2 * i + 1];
        sum_diff_norm2 += diff_re * diff_re + diff_im * diff_im;
    }
    printf(" Csse vs Cavx init: sum_diff_norm2: %12.4g\n", sum_diff_norm2);



#if 1
    {
        printf("===========blas: cgemv===================\n");


        watch.reset();
        for (size_t i = 0; i < Loop; i++) { sse_cgemv(N, A, lda, B, Csse); }
        time_sse = watch.use_usec();

        watch.reset();
        for (size_t i = 0; i < Loop; i++) { avx_cgemv(N, A, lda, B, Cavx); }
        time_avx = watch.use_usec();

        sum_diff_norm2 = 0.0;
        for (size_t i = 0; i < lda; i++) {
            diff_re = Csse[2 * i] - Cavx[2 * i];
            diff_im = Csse[2 * i + 1] - Cavx[2 * i + 1];
            // diff_re = diff_re < 1.0e-6 ? 0.0 : diff_re;
            // diff_im = diff_im < 1.0e-6 ? 0.0 : diff_im;

            sum_diff_norm2 += diff_re * diff_re + diff_im * diff_im;
#ifdef DEBUG
            if (i % 8 == 7) { printf("\n"); }
            printf("(%10.4f,%12.4f) | (%10.4f,%12.4f) | (%10.4f,%12.4f)\n", Csse[2 * i], Csse[2 * i + 1], Cavx[2 * i],
                   Cavx[2 * i + 1], diff_re, diff_im);
#endif
        }
        printf("-----------------------------\n");
        printf(" sum_diff_norm2: %12.6f\n", sum_diff_norm2);
        printf(" time : sse%12.6f    avx%12.6f    sse/avx%12.6f\n", time_sse, time_avx, time_sse / time_avx);
    }
#endif

#if 1
    {
        printf("===========blas: cgenmv===================\n");

        watch.reset();
        for (int l = 0; l < Loop; l++) { sse_cgenmv(N, A, lda, B, Csse); }
        time_sse = watch.use_usec();

        watch.reset();
        for (int l = 0; l < Loop; l++) { avx_cgenmv(N, A, lda, B, Cavx); }
        time_avx = watch.use_usec();

        sum_diff_norm2 = 0.0;
        for (size_t i = 0; i < lda; i++) {
            diff_re = Csse[2 * i] - Cavx[2 * i];
            diff_im = Csse[2 * i + 1] - Cavx[2 * i + 1];
            diff_re = diff_re < 1.0e-6 ? 0.0 : diff_re;
            diff_im = diff_im < 1.0e-6 ? 0.0 : diff_im;

            sum_diff_norm2 += diff_re * diff_re + diff_im * diff_im;
#ifdef DEBUG
            if (i % 8 == 7) { printf("\n"); }
            printf("%12.4g%12.6f |%12.4g%12.6f |%12.4g%12.4g\n", Csse[2 * i], Csse[2 * i + 1], Cavx[2 * i], Cavx[2 * i + 1],
                   diff_re, diff_im);
#endif
        }
        printf("-----------------------------\n");
        printf(" sum_diff_norm2: %12.4g\n", sum_diff_norm2);
        printf(" time : sse%12.6f    avx%12.6f    sse/avx%12.4g\n", time_sse, time_avx, time_sse / time_avx);
    }
#endif

    free(A);
    free(B);
    free(Csse);
    free(Cavx);

    return 0;
}
