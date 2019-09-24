#ifndef ODDEVEN_PRECISION_CUDA
  #define ODDEVEN_PRECISION_CUDA

struct Thread;

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

  extern void cuda_block_d_plus_clover_PRECISION( cuda_vector_PRECISION eta, cuda_vector_PRECISION phi,
                                                  int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s, level_struct *l,
                                                  struct Thread *threading, int stream_id, cudaStream_t *streams, int color,
                                                  int* DD_blocks_to_compute_gpu, int* DD_blocks_to_compute_cpu );

  extern void cuda_block_vector_PRECISION_scale( cuda_vector_PRECISION out, cuda_vector_PRECISION in, cu_cmplx_PRECISION scalar,
                                                 int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s, level_struct *l,
                                                 struct Thread *threading, int stream_id, cudaStream_t *streams, int color,
                                                 int* DD_blocks_to_compute_gpu, int* DD_blocks_to_compute_cpu );

  extern void cuda_block_vector_PRECISION_minus( cuda_vector_PRECISION out, cuda_vector_PRECISION in1, cuda_vector_PRECISION in2,
                                                 int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s, level_struct *l,
                                                 struct Thread *threading, int stream_id, cudaStream_t *streams, int color,
                                                 int* DD_blocks_to_compute_gpu, int* DD_blocks_to_compute_cpu );

  extern void cuda_boundary_comms_copy_PRECISION( cuda_vector_PRECISION out, cuda_vector_PRECISION in, int *bound_table, int num_sites, level_struct *l );

  extern void cuda_vector_PRECISION_define( cuda_vector_PRECISION phi, cu_cmplx_PRECISION scalar, int start,
                                            int length, level_struct *l, int sync_type, int stream_id, cudaStream_t *streams );

  extern void cuda_vector_PRECISION_scale( cuda_vector_PRECISION phi, cu_cmplx_PRECISION scalar, int start,
                                           int length, level_struct *l, int sync_type, int stream_id, cudaStream_t *streams );

#endif
