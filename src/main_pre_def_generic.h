/*
 * Copyright (C) 2016, Matthias Rottmann, Artur Strebel, Gustavo Ramirez, Simon Heybrock, Simone Bacchio, Bjoern Leder, Issaku Kanamori.
 * 
 * This file is part of the DDalphaAMG solver library.
 * 
 * The DDalphaAMG solver library is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * The DDalphaAMG solver library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * 
 * You should have received a copy of the GNU General Public License
 * along with the DDalphaAMG solver library. If not, see http://www.gnu.org/licenses/.
 * 
 */

#ifndef MAIN_PRE_DEF_PRECISION_HEADER
  #define MAIN_PRE_DEF_PRECISION_HEADER

  typedef PRECISION _Complex complex_PRECISION;
  typedef PRECISION _Complex *config_PRECISION;
  typedef PRECISION _Complex *vector_PRECISION;
#ifdef CUDA_OPT
  // CUDA-only typedefs  ---->  cuda vectors
  typedef cu_cmplx_PRECISION* cuda_vector_PRECISION;
  typedef cu_cmplx_PRECISION* cuda_config_PRECISION;
#endif

  struct Thread;
  struct level_struct;

#ifdef CUDA_OPT
  extern __constant__ cu_cmplx_PRECISION gamma_info_vals_PRECISION[16];
  extern __constant__ int gamma_info_coo_PRECISION[16];
#endif

  typedef struct {
    int length[8], *boundary_table[8], max_length[4],
        comm_start[8], in_use[8], offset, comm,
        num_even_boundary_sites[8], num_odd_boundary_sites[8],
        num_boundary_sites[8];
#ifdef CUDA_OPT
    int *boundary_table_gpu[8];
    cuda_vector_PRECISION buffer_gpu[8];
#endif
    vector_PRECISION buffer[8];
    MPI_Request sreqs[8], rreqs[8];
  } comm_PRECISION_struct;
  
  typedef struct {
    int ilde, dist_local_lattice[4], dist_inner_lattice_sites,
        *permutation, *gather_list, gather_list_length;
    vector_PRECISION buffer, transfer_buffer;
    MPI_Request *reqs;
    MPI_Group level_comm_group;
    MPI_Comm level_comm;
  } gathering_PRECISION_struct;
  
  typedef struct {
    config_PRECISION D, clover, oe_clover;
#ifdef CUDA_OPT
    cuda_config_PRECISION clover_gpu;
#endif
    int oe_offset, self_coupling, num_even_sites, num_odd_sites,
        *index_table, *neighbor_table, *translation_table, table_dim[4],
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

#ifdef CUDA_OPT
  typedef struct {
    cu_config_PRECISION *oe_clover_vectorized;
    int *neighbor_table;
    cu_config_PRECISION *D;
    cu_cmplx_PRECISION *Dgpu[16];
    int nr_elems_Dgpu[16];
    cu_cmplx_PRECISION *clover_gpustorg;
    cu_cmplx_PRECISION *oe_clover_gpustorg;
  } operator_PRECISION_struct_on_gpu;
#endif
  
  typedef struct {
    vector_PRECISION x, b, r, w, *V, *Z;
#ifdef CUDA_OPT
    // assuming LEFT preconditioning always, hence 1 element in Z
    cuda_vector_PRECISION x_gpu, w_gpu;
    // <streams> are objects that live on the CPU, and help the CPU to
    // control the GPU kernels ordering
    cudaStream_t *streams;
#endif
    complex_PRECISION **H, *y, *gamma, *c, *s, shift;
    config_PRECISION *D, *clover;
    operator_PRECISION_struct *op;
    PRECISION tol;
    int num_restart, restart_length, timing, print, kind,
        initial_guess_zero, layout, v_start, v_end, total_storage;
    void (*preconditioner)();
    void (*eval_operator)( vector_PRECISION eta, vector_PRECISION phi, operator_PRECISION_struct *op,
                           struct level_struct *l, struct Thread *threading );
  } gmres_PRECISION_struct;

#ifdef CUDA_OPT
  // CUDA structs:
  //	cuda_schwarz_PRECISION_struct:
  //		the elements of this struct will be accessed from the CPU, but their content
  //		are pointers pointing to GPU-data
  //	schwarz_PRECISION_struct_on_gpu:
  //		the elements of this struct will be accessed from within the GPU !
  typedef struct {
    cuda_vector_PRECISION buf1, buf2, buf3, buf4, buf5, buf6;
    int **DD_blocks_in_comms, **DD_blocks_notin_comms, **DD_blocks;
    block_struct* block;
    cuda_vector_PRECISION local_minres_buffer[3];
  } cuda_schwarz_PRECISION_struct;
  typedef struct {
    cu_cmplx_PRECISION* oe_buf[4];
    cu_config_PRECISION *oe_clover_vectorized;
    operator_PRECISION_struct_on_gpu op;
    int num_block_even_sites, num_block_odd_sites;
    int block_vector_size;
    int *oe_index[4];
    int *index[4];
    int dir_length_even[4], dir_length_odd[4];
    int dir_length[4];
    int block_boundary_length[9];
    cu_cmplx_PRECISION gamma_info_vals[16];
    int gamma_info_coo[16];
    cu_cmplx_PRECISION *alphas;
  } schwarz_PRECISION_struct_on_gpu;
#endif

  typedef struct {
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
    //the elements of this struct will be accessed from the CPU, but their content
    //are pointers pointing to GPU-data
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

  typedef struct {
    int num_agg, *agg_index[4], agg_length[4], *agg_boundary_index[4],
        *agg_boundary_neighbor[4], agg_boundary_length[4], num_bootstrap_vect;
    vector_PRECISION *test_vector, *interpolation, *bootstrap_vector, tmp;
    complex_PRECISION *op, *eigenvalues, *bootstrap_eigenvalues;
  } interpolation_PRECISION_struct;
  
  typedef struct {
    double time[_NUM_PROF];
    double flop[_NUM_PROF];
    double count[_NUM_PROF];
    char name[_NUM_PROF][50];
  } profiling_PRECISION_struct;

  #ifdef PROFILING
    #define PROF_PRECISION_START_UNTHREADED( TYPE ) do{ l->prof_PRECISION.time[TYPE] -= MPI_Wtime(); }while(0)
    #define PROF_PRECISION_START_THREADED( TYPE, threading ) do{ if(threading->core + threading->thread == 0) l->prof_PRECISION.time[TYPE] -= MPI_Wtime(); }while(0)
  #else
    #define PROF_PRECISION_START_UNTHREADED( TYPE )
    #define PROF_PRECISION_START_THREADED( TYPE, threading )
  #endif
  
  #ifdef PROFILING
    #define PROF_PRECISION_STOP_UNTHREADED( TYPE, COUNT ) do{ l->prof_PRECISION.time[TYPE] += MPI_Wtime(); \
    l->prof_PRECISION.count[TYPE] += COUNT; }while(0)
    #define PROF_PRECISION_STOP_THREADED( TYPE, COUNT, threading ) do{ if(threading->core + threading->thread == 0) { l->prof_PRECISION.time[TYPE] += MPI_Wtime(); \
    l->prof_PRECISION.count[TYPE] += COUNT; } }while(0)
  #else
    #define PROF_PRECISION_STOP_UNTHREADED( TYPE, COUNT )
    #define PROF_PRECISION_STOP_THREADED( TYPE, COUNT, threading )
  #endif

  #define GET_MACRO2(_1,_2,NAME,...) NAME
  #define GET_MACRO3(_1,_2,_3,NAME,...) NAME
  #define PROF_PRECISION_START(...) GET_MACRO2(__VA_ARGS__, PROF_PRECISION_START_THREADED, PROF_PRECISION_START_UNTHREADED, padding)(__VA_ARGS__)
  #define PROF_PRECISION_STOP(...) GET_MACRO3(__VA_ARGS__, PROF_PRECISION_STOP_THREADED, PROF_PRECISION_STOP_UNTHREADED, padding)(__VA_ARGS__)
  
#endif
