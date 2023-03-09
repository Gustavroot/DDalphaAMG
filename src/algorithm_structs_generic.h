/**
 * \file algorithm_structs_generic.h
 * 
 * \brief Contains structs related to various methods applied throughout the application.
 */

#ifndef ALGORITHM_STRUCTS_PRECISION_H
#define ALGORITHM_STRUCTS_PRECISION_H

#ifdef CUDA_OPT
#include <cuda_runtime.h>
#include "gpu/cuda_vectors_PRECISION.h"
#include "gpu/cuda_algorithm_structs_PRECISION.h"
#endif

#include "block_struct.h"
#include "complex_types_PRECISION.h"
#include "communication_PRECISION.h"
#include "vectorization_control.h"

typedef struct
{
    config_PRECISION D, clover, oe_clover;
#ifdef CUDA_OPT
    // TODO remove
    vector_PRECISION w_test;

    cuda_config_PRECISION clover_gpu, D_gpu;

    // Local vectors to apply the operator w = w + Dx on the GPU.
    cuda_vector_PRECISION x_gpu, w_gpu, pbuf_gpu;
    cuda_vector_PRECISION prpT_gpu, prpZ_gpu, prpY_gpu, prpX_gpu;
    cuda_vector_PRECISION prnT_gpu, prnZ_gpu, prnY_gpu, prnX_gpu;

    /** \see neighbor_table */
    int * neighbor_table_gpu;
#endif
    int oe_offset, self_coupling, num_even_sites, num_odd_sites,
        *index_table,
        /**
         * \brief Gives the indices of neighboring lattice sites in each dimension.
         * 
         * *Disclaimer*: This documentation is from Tilmann Matthaei and added way after this was
         * initially introduced. So this might be slightly incorrect or not in the spirit of the
         * original author.
         * 
         * The neighbor_table looks something like this: 512, 64, 8, 1, 513, 65, 9, 2...
         * It gives the indices of the neighboring lattice sites in groups of 4. E.g. in the above
         * example the lattice site with index 0 has neighbors:
         * 
         * T: 512, Z: 64, Y: 8, X: 1
         * 
         * That is followed by the neighbors of the lattice site with index 1 in all 4 directions.
         */
        *neighbor_table, *translation_table, table_dim[4],
        *backward_neighbor_table,
        table_mod_dim[4], *config_boundary_table[4];
    complex_PRECISION shift;
    vector_PRECISION *buffer, prnT, prnZ, prnY, prnX, prpT, prpZ, prpY, prpX;
    comm_PRECISION_struct c;
    OPERATOR_TYPE_PRECISION *D_vectorized;
    OPERATOR_TYPE_PRECISION *D_transformed_vectorized;
    OPERATOR_TYPE_PRECISION *clover_vectorized;
    OPERATOR_TYPE_PRECISION *oe_clover_vectorized;
} operator_PRECISION_struct;

struct level_struct;
struct Thread;

typedef struct
{
    vector_PRECISION x, b, r, w, *V, *Z;
#ifdef CUDA_OPT
    // <streams> are objects that live on the CPU, and help the CPU to
    // control the GPU kernels ordering
    cudaStream_t *streams;

    vector_PRECISION xtmp;
#endif
    complex_PRECISION **H, *y, *gamma, *c, *s, shift;
    config_PRECISION *D, *clover;
    operator_PRECISION_struct *op;
    PRECISION tol;
    int num_restart, restart_length, timing, print, kind,
        initial_guess_zero, layout, v_start, v_end, total_storage;
    void (*preconditioner)();
    void (*eval_operator)(vector_PRECISION eta, vector_PRECISION phi, operator_PRECISION_struct *op,
                          struct level_struct *l, struct Thread *threading);
} gmres_PRECISION_struct;

typedef struct
{
    operator_PRECISION_struct op;
    vector_PRECISION buf1, buf2, buf3, buf4, buf5, bbuf1, bbuf2, bbuf3, oe_bbuf[6];
    vector_PRECISION oe_buf[4];
    vector_PRECISION local_minres_buffer[3];
    int block_oe_offset, *index[4], dir_length[4], num_blocks, num_colors,
        dir_length_even[4], dir_length_odd[4], *oe_index[4],
        num_block_even_sites, num_block_odd_sites, num_aggregates,
        block_vector_size, num_block_sites, block_boundary_length[9],
        **block_list, *block_list_length;
    block_struct *block;
#ifdef CUDA_OPT
    // <streams> are objects that live on the CPU, and help the CPU to
    // control the GPU kernels ordering
    cudaStream_t *streams;
    int nr_streams;
    // the elements of this struct will be accessed from the CPU, but their content
    // are pointers pointing to GPU-data
    cuda_schwarz_PRECISION_struct cu_s;
    // there's a good reason for having two of these:
    //		s_on_gpu_cpubuff: this one lives (always) on the CPU, and it's created
    //				  to then be copied to the GPU
    //		s_on_gpu:         this one will point to data on the GPU, corresponding
    //				  to a copy of s_on_gpu_cpubuff
    schwarz_PRECISION_struct_on_gpu s_on_gpu_cpubuff;
    schwarz_PRECISION_struct_on_gpu *s_on_gpu;
    int tot_num_boundary_work;
    int num_boundary_sites[8];
    int *nr_DD_blocks_in_comms, *nr_DD_blocks_notin_comms;
    int **DD_blocks_in_comms, **DD_blocks_notin_comms;
    int *nr_DD_blocks;
    int **DD_blocks;
    int nr_thrDD_blocks_notin_comms_[2], nr_thrDD_blocks_in_comms_[2], DD_thr_offset_notin_comms_[2], DD_thr_offset_in_comms_[2];
#endif
} schwarz_PRECISION_struct;

typedef struct
{
    int num_agg, *agg_index[4], agg_length[4], *agg_boundary_index[4],
        *agg_boundary_neighbor[4], agg_boundary_length[4], num_bootstrap_vect;
    vector_PRECISION *test_vector, *interpolation, *bootstrap_vector, tmp;
    complex_PRECISION *op, *eigenvalues, *bootstrap_eigenvalues;
} interpolation_PRECISION_struct;

#endif // ALGORITHM_STRUCTS_PRECISION_H