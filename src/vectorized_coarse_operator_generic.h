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

#ifndef VECTORIZED_COARSE_OPERATOR_PRECISION_HEADER
#define VECTORIZED_COARSE_OPERATOR_PRECISION_HEADER

#include "vectorized_blas.h"
#include <assert.h>
static int coarse_self_couplings_PRECISION_count = 0;

/**
 * @brief vectorized_coarse_self_couplings_PRECISION(); AVX2 defibed by vectorized_cgemv() in vectorized_blas.h/vectorized_blas_avx.h
 * 
 * @param eta 
 * @param phi 
 * @param clover 
 * @param start 
 * @param end 
 * @param l 
 */
static inline void vectorized_coarse_self_couplings_PRECISION(vector_PRECISION eta, vector_PRECISION phi,
                                                              OPERATOR_TYPE_PRECISION *clover, int start, int end,
                                                              level_struct *l)
{
#if defined DEBUG
    if (coarse_self_couplings_PRECISION_count == 0) {
        printf("========== vectorized_coarse_self_couplings_PRECISION() called\n");
        coarse_self_couplings_PRECISION_count++;
    }
    assert(eta);
    assert(phi);
    assert(clover);
#endif

    int site_size = l->num_lattice_site_var;
    int lda       = AVX_LENGTH_float * ((site_size + AVX_LENGTH_float - 1) / AVX_LENGTH_float);

    for (int i = start; i < end; i++) {
        for (int j = 0; j < site_size; j++) { eta[i * site_size + j] = 0.0; }
        vectorized_cgemv(site_size, clover + i * 2 * site_size * lda, lda, (float *) (phi + i * site_size),
                         (float *) (eta + i * site_size));
    }
}


// void vectorized_coarse_operator_PRECISION_set_couplings(operator_PRECISION_struct *op, level_struct *l, struct Thread *threading);

/**
 * @brief 
 * 
 * @param eta 
 * @param phi 
 * @param D 
 * @param l 
 */
static inline void vectorized_coarse_hopp_PRECISION(vector_PRECISION eta, vector_PRECISION phi, OPERATOR_TYPE_PRECISION *D,
                                                    level_struct *l)
{
#ifdef VECTORIZE_COARSE_OPERATOR_PRECISION
    int lda = SIMD_LENGTH_PRECISION * ((l->num_lattice_site_var + SIMD_LENGTH_PRECISION - 1) / SIMD_LENGTH_PRECISION);
    vectorized_cgenmv(l->num_lattice_site_var, D, lda, (float *) phi, (float *) eta);
#endif
}

/**
 * @brief 
 * 
 * @param eta 
 * @param phi 
 * @param D 
 * @param l 
 */
static inline void vectorized_coarse_n_hopp_PRECISION(vector_PRECISION eta, vector_PRECISION phi, OPERATOR_TYPE_PRECISION *D,
                                                          level_struct *l)
{

    int lda = AVX_LENGTH_float * ((l->num_lattice_site_var + AVX_LENGTH_float - 1) / AVX_LENGTH_float);
    vectorized_cgemv(l->num_lattice_site_var, D, lda, (float *) phi, (float *) eta);
}

#endif