/**
 * @file vectorized_coarse_operator_generic.c
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-12-29
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "main.h"
#include "vectorized_coarse_operator_PRECISION.h"

#if 0
void vectorized_coarse_operator_PRECISION_set_couplings(operator_PRECISION_struct *op, level_struct *l, struct Thread *threading)
{

    int n       = l->num_inner_lattice_sites;
    int sc_size = (l->num_lattice_site_var / 2) * (l->num_lattice_site_var + 1);
    int nc_size = SQUARE(l->num_lattice_site_var);
    int n1, n2;
    if (l->depth > 0) {
        n1 = l->num_lattice_sites;
        n2 = 2 * l->num_lattice_sites - l->num_inner_lattice_sites;
    } else {
        n1 = l->num_inner_lattice_sites;
        n2 = l->num_inner_lattice_sites;
    }

    START_LOCKED_MASTER(threading)
    if (op->D_vectorized == NULL) {
        int column_offset =
            SIMD_LENGTH_PRECISION * ((l->num_lattice_site_var + SIMD_LENGTH_PRECISION - 1) / SIMD_LENGTH_PRECISION);
        // 2 is for complex, 4 is for 4 directions
        MALLOC_HUGEPAGES(op->D_vectorized, OPERATOR_TYPE_PRECISION, 2 * 4 * l->num_lattice_site_var * column_offset * n2, 64);
        MALLOC_HUGEPAGES(op->D_transformed_vectorized, OPERATOR_TYPE_PRECISION,
                         2 * 4 * l->num_lattice_site_var * column_offset * n2, 64);
        MALLOC_HUGEPAGES(op->clover_vectorized, OPERATOR_TYPE_PRECISION, 2 * l->num_lattice_site_var * column_offset * n, 64);
    }
    END_LOCKED_MASTER(threading)

    int start, end;
    compute_core_start_end_custom(0, n, &start, &end, l, threading, 1);
    int n_per_core    = end - start;
    int column_offset = SIMD_LENGTH_PRECISION * ((l->num_lattice_site_var + SIMD_LENGTH_PRECISION - 1) / SIMD_LENGTH_PRECISION);
    int offset_v      = 2 * l->num_lattice_site_var * column_offset;
    copy_coarse_operator_to_vectorized_layout_PRECISION(op->D + 4 * start * nc_size, op->D_vectorized + 4 * start * offset_v,
                                                        n_per_core, l->num_lattice_site_var / 2);
    copy_coarse_operator_to_transformed_vectorized_layout_PRECISION(op->D + 4 * start * nc_size,
                                                                    op->D_transformed_vectorized + 4 * start * offset_v,
                                                                    n_per_core, l->num_lattice_site_var / 2);
    copy_coarse_operator_clover_to_vectorized_layout_PRECISION(
        op->clover + start * sc_size, op->clover_vectorized + start * offset_v, n_per_core, l->num_lattice_site_var / 2);
    SYNC_CORES(threading)

    // vectorize negative boundary
    if (l->depth > 0) {
        compute_core_start_end_custom(n1, n2, &start, &end, l, threading, 1);
        n_per_core = end - start;
        copy_coarse_operator_to_vectorized_layout_PRECISION(op->D + 4 * start * nc_size, op->D_vectorized + 4 * start * offset_v,
                                                            n_per_core, l->num_lattice_site_var / 2);
        copy_coarse_operator_to_transformed_vectorized_layout_PRECISION(op->D + 4 * start * nc_size,
                                                                        op->D_transformed_vectorized + 4 * start * offset_v,
                                                                        n_per_core, l->num_lattice_site_var / 2);
        SYNC_CORES(threading)
    }
}

#endif