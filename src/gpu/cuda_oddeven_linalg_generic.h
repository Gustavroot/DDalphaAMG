#ifndef ODDEVEN_LINALG_PRECISION_HEADER_CUDA
  #define ODDEVEN_LINALG_PRECISION_HEADER_CUDA

  // device functions

  // 6 and 12 threads, optimized

  __global__ void cuda_block_oe_vector_PRECISION_copy_6threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, schwarz_PRECISION_struct_on_gpu *s, 
                                                                    int thread_id, double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, int num_latt_site_var,
                                                                    block_struct* block, int sites_to_copy );

  __global__ void cuda_block_oe_vector_PRECISION_copy_12threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, schwarz_PRECISION_struct_on_gpu *s,
                                                                     int thread_id, double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, int num_latt_site_var,
                                                                     block_struct* block, int sites_to_copy );

  __global__ void cuda_block_oe_vector_PRECISION_define_6threads_opt( cu_cmplx_PRECISION* spinor, schwarz_PRECISION_struct_on_gpu *s, int thread_id, double csw,
                                                                      int nr_threads_per_DD_block, int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block,
                                                                      int sites_to_define, cu_cmplx_PRECISION val_to_assign );

  __global__ void cuda_block_oe_vector_PRECISION_define_12threads_opt( cu_cmplx_PRECISION* spinor, schwarz_PRECISION_struct_on_gpu *s, int thread_id, double csw,
                                                                       int nr_threads_per_DD_block, int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block,
                                                                       int sites_to_define, cu_cmplx_PRECISION val_to_assign );

  __global__ void cuda_block_oe_vector_PRECISION_plus_6threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in1, cu_cmplx_PRECISION* in2,
                                                                    schwarz_PRECISION_struct_on_gpu *s, int thread_id, double csw, int nr_threads_per_DD_block,
                                                                    int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block, int sites_to_add );

  __global__ void cuda_block_oe_vector_PRECISION_plus_12threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in1, cu_cmplx_PRECISION* in2,
                                                                     schwarz_PRECISION_struct_on_gpu *s, int thread_id, double csw, int nr_threads_per_DD_block,
                                                                     int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block, int sites_to_add );

  __global__ void cuda_block_oe_vector_PRECISION_minus_6threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in1, cu_cmplx_PRECISION* in2, schwarz_PRECISION_struct_on_gpu *s,
                                                                     int thread_id, double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                     int num_latt_site_var, block_struct* block, int sites_to_add );

  __global__ void cuda_local_xy_over_xx_PRECISION( cu_cmplx_PRECISION* vec1, cu_cmplx_PRECISION* vec2, \
                                                   schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                   double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                   int num_latt_site_var, block_struct* block, int sites_to_dot );

  __global__ void cuda_block_oe_vector_PRECISION_saxpy_6threads_opt_onchip( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in1, cu_cmplx_PRECISION* in2, PRECISION prefctr_alpha,
                                                                            cu_cmplx_PRECISION *alpha, schwarz_PRECISION_struct_on_gpu *s, int thread_id, double csw,
                                                                            int nr_threads_per_DD_block, int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block, int sites_to_add );

  __global__ void cuda_block_oe_vector_PRECISION_saxpy_12threads_opt_onchip( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in1, cu_cmplx_PRECISION* in2, PRECISION prefctr_alpha,
                                                                             cu_cmplx_PRECISION *alpha, schwarz_PRECISION_struct_on_gpu *s, int thread_id, double csw,
                                                                             int nr_threads_per_DD_block, int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block, int sites_to_add );


  // host functions

  extern void cuda_block_vector_PRECISION_minus( cuda_vector_PRECISION out, cuda_vector_PRECISION in1, cuda_vector_PRECISION in2,
                                                 int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s, level_struct *l,
                                                 struct Thread *threading, int stream_id, cudaStream_t *streams, int color,
                                                 int* DD_blocks_to_compute_gpu, int* DD_blocks_to_compute_cpu );

  extern void cuda_vector_PRECISION_define( cuda_vector_PRECISION phi, cu_cmplx_PRECISION scalar, int start,
                                            int length, level_struct *l, int sync_type, int stream_id, cudaStream_t *streams );

  extern void cuda_vector_PRECISION_scale( cuda_vector_PRECISION phi1, cuda_vector_PRECISION phi2, cu_cmplx_PRECISION scalar, int start,
                                           int length, level_struct *l, int sync_type, int stream_id, cudaStream_t *streams );

  extern void cuda_block_vector_PRECISION_scale( cuda_vector_PRECISION out, cuda_vector_PRECISION in, cu_cmplx_PRECISION scalar,
                                                 int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s, level_struct *l,
                                                 struct Thread *threading, int stream_id, cudaStream_t *streams, int color,
                                                 int* DD_blocks_to_compute_gpu, int* DD_blocks_to_compute_cpu );

#endif
