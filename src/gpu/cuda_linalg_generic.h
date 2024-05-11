#ifndef LINALG_PRECISION_HEADER_CUDA
#define LINALG_PRECISION_HEADER_CUDA

#include <cuda_runtime.h>

#include "global_enums.h"
#include "level_struct.h"
#ifdef __cplusplus
extern "C" {
#endif

void cuda_vector_PRECISION_copy(void *out, void const *in, int start, int size_of_copy,
                                level_struct *l, const int memcpy_kind, const int cuda_async_type,
                                const int stream_id, cudaStream_t *streams);

void cuda_vector_PRECISION_minus( cuda_vector_PRECISION z, cuda_vector_PRECISION x, cuda_vector_PRECISION y, int start,
                                  int length, level_struct *l, int sync_type, int stream_id, cudaStream_t *streams );

void cuda_vector_PRECISION_saxpy( cuda_vector_PRECISION z, cuda_vector_PRECISION x, cuda_vector_PRECISION y, cu_cmplx_PRECISION alpha, int start,
                                  int length, level_struct *l, int sync_type, int stream_id, cudaStream_t *streams );

#ifdef __cplusplus
}
#endif

#endif
