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

#include <assert.h>
#include "main.h"
#include "vectorized_blas.h"
#include "vectorization_control.h"


/**
 * @brief vectorized_coarse_self_couplings_PRECISION(); vectorized_cgemv() in vectorized_blas.h/vectorized_blas_avx.h
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
    int lda = SIMD_LENGTH_PRECISION * ((l->num_lattice_site_var + SIMD_LENGTH_PRECISION - 1) / SIMD_LENGTH_PRECISION);
    vectorized_cgenmv(l->num_lattice_site_var, D, lda, (float *) phi, (float *) eta);
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

    int lda = SIMD_LENGTH_float * ((l->num_lattice_site_var + SIMD_LENGTH_float - 1) / SIMD_LENGTH_float);
    vectorized_cgemv(l->num_lattice_site_var, D, lda, (float *) phi, (float *) eta);
}


static inline void vectorised_coarse_hopp_PRECISION(vector_PRECISION eta, vector_PRECISION phi, OPERATOR_TYPE_PRECISION *D,
                                                    level_struct *l)
{
    int lda = SIMD_LENGTH_PRECISION * ((l->num_lattice_site_var + SIMD_LENGTH_PRECISION - 1) / SIMD_LENGTH_PRECISION);
    vectorized_cgenmv(l->num_lattice_site_var, D, lda, (float *) phi, (float *) eta);
}


#endif