#ifdef CUDA_OPT
#ifndef ODDEVEN_PRECISION_CUDA
  #define ODDEVEN_PRECISION_CUDA

struct Thread;

  // device functions

  __global__ void cuda_site_clover_PRECISION( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                              schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                              int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                              int num_latt_site_var, block_struct* block );

  __global__ void cuda_block_hopping_term_PRECISION_plus_6threads_naive( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                                                         schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                         int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                                         int num_latt_site_var, block_struct* block, int ext_dir, int amount );

  __global__ void cuda_block_hopping_term_PRECISION_minus_6threads_naive( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                                                          schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                          int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                                          int num_latt_site_var, block_struct* block, int ext_dir, int amount );

  // host functions

  extern void cuda_block_solve_oddeven_PRECISION( cuda_vector_PRECISION phi, cuda_vector_PRECISION r, cuda_vector_PRECISION latest_iter,
                                                  int start, int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s, level_struct *l,
                                                  struct Thread *threading, int stream_id, cudaStream_t *streams, int solve_at_cpu, int color,
                                                  int* DD_blocks_to_compute_gpu, int* DD_blocks_to_compute_cpu );

  extern void cuda_block_PRECISION_boundary_op( cuda_vector_PRECISION eta, cuda_vector_PRECISION phi,
                                                int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s, level_struct *l,
                                                struct Thread *threading, int stream_id, cudaStream_t *streams, int color,
                                                int* DD_blocks_to_compute_gpu, int* DD_blocks_to_compute_cpu );

  extern void cuda_n_block_PRECISION_boundary_op( cuda_vector_PRECISION eta, cuda_vector_PRECISION phi,
                                                  int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s, level_struct *l,
                                                  struct Thread *threading, int stream_id, cudaStream_t *streams, int color,
                                                  int* DD_blocks_to_compute_gpu, int* DD_blocks_to_compute_cpu );

  extern void cuda_boundary_comms_copy_PRECISION( cuda_vector_PRECISION out, cuda_vector_PRECISION in, int *bound_table, int num_sites, level_struct *l );

  extern void cuda_apply_block_schur_complement_PRECISION( cuda_vector_PRECISION out, cuda_vector_PRECISION in,
                                                           schwarz_PRECISION_struct *s, level_struct *l, int nr_DD_blocks_to_compute,
                                                           int* DD_blocks_to_compute, cudaStream_t *streams, int stream_id, int sites_to_solve );

#endif
#endif
