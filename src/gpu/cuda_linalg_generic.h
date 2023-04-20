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

#ifdef __cplusplus
}
#endif

#endif
