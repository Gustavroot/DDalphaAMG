#ifndef TEST_MACROS_H
#define TEST_MACROS_H

#include <cuda_runtime.h>

#define RC_ASSERT_CUDA_SUCCESS(err) RC_ASSERT(cudaSuccess == err)

#endif // TEST_MACROS_H
