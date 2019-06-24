#ifndef LINALG_PRECISION_HEADER_CUDA
  #define LINALG_PRECISION_HEADER_CUDA

  void cuda_vector_PRECISION_copy( void* out, void* in, int start, int end, level_struct *l,
                                   const int memcpy_kind, const int cuda_async_type, const int stream_id, cudaStream_t *streams );

#endif
