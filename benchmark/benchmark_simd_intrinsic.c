#include <stdio.h>

#include <immintrin.h>

// here's a check for _mm256_permutevar8x32_ps(__m256, __m256i);

int main(int argc, char const *argv[])
{

    {
        float VA[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        __m256i idx = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
        __m256 va   = _mm256_loadu_ps(VA);
        va          = _mm256_permutevar8x32_ps(va, idx);
        _mm256_storeu_ps(VA, va);
        for (size_t i = 0; i < 8; i++) { printf("%8.0f", VA[i]); }
        printf("\n");
    }

    {
        float VB[4] = {0, 1, 2, 3};
        __m128 vb   = _mm_load_ps(VB);
        vb          = _mm_permute_ps(vb, 0b11011000);
        _mm_store_ps(VB, vb);
        for (size_t i = 0; i < 4; i++) { printf("%8.0f", VB[i]); }
        printf("\n");
    }

    {
        printf("\n---------_mm256_extractf128_ps(va, 0b01) \n");
        float VA[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        for (size_t i = 0; i < 8; i++) { printf("%8.0f", VA[i]); }
        printf("\n");
        __m256 va = _mm256_loadu_ps(VA);
        __m128 vb = _mm256_extractf128_ps(va, 0b0);
        _mm_store_ps(VA, vb);
        for (size_t i = 0; i < 8; i++) { printf("%8.0f", VA[i]); }
        printf("\n");

        printf("\n---------_mm256_castps256_ps128(va) \n");
        __m128 vc = _mm256_castps256_ps128(va);
        _mm_store_ps(VA, vc);
        for (size_t i = 0; i < 8; i++) { printf("%8.0f", VA[i]); }
        printf("\n");
    }


    return 0;
}