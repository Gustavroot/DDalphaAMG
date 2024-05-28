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
#include "gpu/cuda_communication_PRECISION.h"
#endif

#include "block_struct.h"
#include "complex_types_PRECISION.h"
#include "communication_PRECISION.h"
#include "vectorization_control.h"

typedef struct
{
    config_PRECISION D, clover, oe_clover;
#ifdef CUDA_OPT
    cuda_config_PRECISION
        /** Self coupling coefficients on the GPU. */
        clover_gpu,
        /** Self coupling coefficients in componentwise ordering on the GPU.*/
        clover_componentwise_gpu,
        /** Neighbor coupling coefficients on the GPU. */
        D_gpu;
    /** Neighbor coupling coefficients in componentwise ordering on the GPU.
     * 
     *  This is an array of four pointers to the coefficients in each direction.
     *  Indices are T, Z, Y and X.
     */
    cu_cmplx_PRECISION * Ds_componentwise_gpu[4];

    // Local vectors to apply the operator w = w + Dx on the GPU.
    cuda_vector_PRECISION
        /** x in D x = w for the iterative solution. */
        x_gpu,
        /** x in D x = w for the iterative solution.
         * 
         *  Vector is used for a copy of x in componentwise ordering.
         */
        x_componentwise_gpu,
        /** w in D x = w for the iterative solution. */
        w_gpu,
        /** w in D x = w for the iterative solution.
         * 
         *  Vector is used for a copy of w in componentwise ordering.
         */
        w_componentwise_gpu,
        /** Temporary buffer used to store intermediate results. */
        pbuf_gpu;
    cuda_vector_PRECISION prpT_gpu, prpZ_gpu, prpY_gpu, prpX_gpu;
    cuda_vector_PRECISION prnT_gpu, prnZ_gpu, prnY_gpu, prnX_gpu;

    cuda_vector_PRECISION buffer_gpu[2];

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
        table_mod_dim[4];
    complex_PRECISION shift;
    vector_PRECISION *buffer, prnT, prnZ, prnY, prnX, prpT, prpZ, prpY, prpX;
    comm_PRECISION_struct c;
#ifdef CUDA_OPT
    cuda_comm_PRECISION_struct cuda_c;
#endif
    OPERATOR_TYPE_PRECISION *D_vectorized;
    OPERATOR_TYPE_PRECISION *D_transformed_vectorized;
    OPERATOR_TYPE_PRECISION *clover_vectorized;
    OPERATOR_TYPE_PRECISION *oe_clover_vectorized;
} operator_PRECISION_struct;

struct level_struct;
struct Thread;

#if defined(POLYPREC) || defined(GCRODR)
  typedef struct
  {
    int N, nrhs, lda, ldb, info;

    int *ipiv;
    vector_PRECISION x, b;
    complex_PRECISION *Hcc;  

    void (*dirctslvr_PRECISION)();

  } dirctslvr_PRECISION_struct;
#endif

#if defined(GCRODR) || defined(POLYPREC)
  // this is both eigensolver and generalized eigensolver
  typedef struct {
    char jobvl, jobvr;

    int N, lda, ldb, ldvl, ldvr, info, qr_m, qr_n, qr_lda, qr_k;

    int *ordr_idxs;

    complex_PRECISION *ordr_keyscpy, *qr_tau;
    vector_PRECISION vl, vr, w, beta, A, B;

    complex_PRECISION **qr_QR, **qr_Q, **qr_R, **qr_Rinv;
    complex_PRECISION **Hc;

    void (*eigslvr_PRECISION)();
    void (*gen_eigslvr_PRECISION)();
  } eigslvr_PRECISION_struct;
#endif

#ifdef GCRODR
  typedef struct {
    int i, k, CU_usable, syst_size, finish, orth_against_Ck, update_CU, recompute_DPCk_poly, recompute_DPCk_plain, upd_ctr;

    PRECISION b_norm, norm_r0;

    vector_PRECISION *Pk, *C, *Cc, *U, *Yk, *hatZ, *hatW;
//#ifdef BLOCK_JACOBI
#if 0
    vector_PRECISION r_aux;
#endif
    // Gc is used to copy G
    complex_PRECISION *lsp_x, *lsp_diag_G, **lsp_H;
    complex_PRECISION **gev_A, **gev_B, **Bbuff, **QR, **Q, **R, **Rinv, **ort_B, **G, **Gc;

    eigslvr_PRECISION_struct eigslvr;

#if defined(SINGLE_ALLREDUCE_ARNOLDI) && defined(PIPELINED_ARNOLDI)
    vector_PRECISION *PC, *DPC;
#endif
  } gcrodr_PRECISION_struct;
#endif

#ifdef POLYPREC
  typedef struct
  {
    int update_lejas;
    int d_poly;
    int syst_size;
      
    complex_PRECISION **Hc;
    complex_PRECISION *Hcc;
    complex_PRECISION **L;
    complex_PRECISION *col_prods;
    vector_PRECISION h_ritz;
    vector_PRECISION lejas;
    vector_PRECISION random_rhs;
    vector_PRECISION accum_prod, product, temp, xtmp;

    void (*preconditioner)();
    void (*preconditioner_bare)();

    eigslvr_PRECISION_struct eigslvr;
    dirctslvr_PRECISION_struct dirctslvr;
  } polyprec_PRECISION_struct;
#endif

//#ifdef BLOCK_JACOBI
#if 0
  typedef struct {
    vector_PRECISION x, b, r, w, *V, *Z;
    complex_PRECISION **H, *y, *gamma, *c, *s;
    config_PRECISION *D, *clover;
    operator_PRECISION_struct *op;
    PRECISION tol;
    int num_restart, restart_length, timing, print, kind,
      initial_guess_zero, layout, v_start, v_end;
    long int total_storage;
    void (*eval_operator)();

    polyprec_PRECISION_struct polyprec_PRECISION;
  } local_gmres_PRECISION_struct;

  typedef struct {
    int BJ_usable, syst_size;
    vector_PRECISION b_backup;
    vector_PRECISION xtmp;
    local_gmres_PRECISION_struct local_p;

    // for direct solves
    OPERATOR_TYPE_PRECISION* bj_op_inv_vectorized;
    OPERATOR_TYPE_PRECISION* bj_op_vectorized;
    OPERATOR_TYPE_PRECISION* bj_doublet_op_inv_vectorized;
    OPERATOR_TYPE_PRECISION* bj_doublet_op_vectorized;

    //vector_PRECISION xxxtmp[4];

  } block_jacobi_PRECISION_struct;
#endif

typedef struct
{
    vector_PRECISION x, b, r, w, *V, *Z;
#ifdef CUDA_OPT
    int gpu_syst_size;

    // <streams> are objects that live on the CPU, and help the CPU to
    // control the GPU kernels ordering
    cudaStream_t *streams;

    vector_PRECISION xtmp;

    cuda_vector_PRECISION b_gpu, b_componentwise_gpu, x_gpu,
                          x_componentwise_gpu, w_componentwise_gpu,
                          r_componentwise_gpu;
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
#ifdef GCR_SMOOTHER
    int use_gcr;
#endif

    complex_PRECISION *gcr_buffer_dotprods;
    complex_PRECISION *gcr_betas_dotprods;

#ifdef RICHARDSON_SMOOTHER
    int use_richardson,richardson_update_omega,richardson_sub_degree;
    PRECISION *omega;
    double richardson_factor;
#endif

#ifdef GCRODR
    gcrodr_PRECISION_struct gcrodr_PRECISION;
    int was_there_stagnation;
    vector_PRECISION rhs_bk;
#endif
#ifdef POLYPREC
    polyprec_PRECISION_struct polyprec_PRECISION;
#endif
//#ifdef BLOCK_JACOBI
#if 0
    block_jacobi_PRECISION_struct block_jacobi_PRECISION;
#endif
#if defined(SINGLE_ALLREDUCE_ARNOLDI) && defined(PIPELINED_ARNOLDI)
    int syst_size;
    vector_PRECISION *Va, *Za;
#endif
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
