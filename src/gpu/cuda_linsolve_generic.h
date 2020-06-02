#ifndef LINSOLVE_PRECISION_HEADER_CUDA
  #define LINSOLVE_PRECISION_HEADER_CUDA

  extern void local_minres_PRECISION_CUDA( cuda_vector_PRECISION phi, cuda_vector_PRECISION eta, cuda_vector_PRECISION latest_iter,
                                           schwarz_PRECISION_struct *s, level_struct *l, int nr_DD_blocks_to_compute,
                                           int* DD_blocks_to_compute, cudaStream_t *streams, int stream_id, int sites_to_solve );

#endif
