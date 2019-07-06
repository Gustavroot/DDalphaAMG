#ifndef ODDEVEN_PRECISION_CUDA
  #define ODDEVEN_PRECISION_CUDA

struct Thread;

  extern void cuda_block_solve_oddeven_PRECISION( cuda_vector_PRECISION phi, cuda_vector_PRECISION r, cuda_vector_PRECISION latest_iter,
                                                  int start, int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s, level_struct *l,
                                                  struct Thread *threading, int stream_id, cudaStream_t *streams, int solve_at_cpu );

#endif
