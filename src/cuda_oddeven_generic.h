#ifndef ODDEVEN_PRECISION_CUDA
  #define ODDEVEN_PRECISION_CUDA

struct Thread;

  extern void cuda_block_solve_oddeven_PRECISION( cuda_vector_PRECISION phi, cuda_vector_PRECISION r, cuda_vector_PRECISION latest_iter,
                                                  int start, schwarz_PRECISION_struct *s, level_struct *l, struct Thread *threading );

#endif
