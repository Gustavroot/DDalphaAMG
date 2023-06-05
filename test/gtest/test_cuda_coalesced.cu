#include <cuda.h>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include "gpu/cuda_coalesced.h"
#include "gpu/cuda_miscellaneous.h"
#include "miscellaneous.h"

__global__ void _checkDstKernel(int *dst, int const *src, unsigned int chunkSize,
                                unsigned int gapSize, unsigned int chunkCount) {
  copyChunksToConsecutiveAsBlock<int>(dst, src, chunkSize, gapSize, chunkCount);
}

RC_GTEST_PROP(CopyChunksToConsecutiveAsBlockTest, CheckDst, ()) {
  unsigned int chunkSize = *rc::gen::inRange(1, 10);
  unsigned int gapSize = *rc::gen::inRange(0, 10);
  unsigned int blockSize = *rc::gen::inRange(1, 256);

  unsigned int arraySize = blockSize * (chunkSize + gapSize);
  int *src = (int *)malloc(arraySize * sizeof(int));
  int *dst = (int *)malloc(chunkSize * blockSize * sizeof(int));
  int *srcCuda, *dstCuda;
  cuda_safe_call(cudaMalloc(&srcCuda, arraySize * sizeof(int)));
  cuda_safe_call(cudaMalloc(&dstCuda, chunkSize * blockSize * sizeof(int)));

  // set chunks to 1 and gaps to -1
  for (unsigned int i = 0; i < arraySize; i++) {
    src[i] = (i % (chunkSize + gapSize) < chunkSize) ? 1 : -1;
  }

  cuda_safe_call(cudaMemcpy(srcCuda, src, arraySize * sizeof(int), cudaMemcpyHostToDevice));
  _checkDstKernel<<<1, blockSize>>>(dstCuda, srcCuda, chunkSize, gapSize, blockSize);
  cuda_safe_call(cudaDeviceSynchronize());
  cuda_safe_call(
      cudaMemcpy(dst, dstCuda, chunkSize * blockSize * sizeof(int), cudaMemcpyDeviceToHost));

  for (unsigned int i = 0; i < chunkSize * blockSize; i++) {
    RC_ASSERT(dst[i] == 1);
  }

  free(src);
  free(dst);
  cuda_safe_call(cudaFree(srcCuda));
  cuda_safe_call(cudaFree(dstCuda));
}

RC_GTEST_PROP(CopyChunksToConsecutiveAsBlockTest, CheckDstNonDivisible, ()) {
  unsigned int chunkSize = *rc::gen::inRange(1, 10);
  unsigned int chunkCount = *rc::gen::inRange(1, 1024);
  unsigned int gapSize = *rc::gen::inRange(0, 10);
  unsigned int blockSize = *rc::gen::inRange(1, 256);

  unsigned int arraySize = chunkCount * (chunkSize + gapSize);
  int *src = (int *)malloc(arraySize * sizeof(int));
  int *dst = (int *)malloc(chunkSize * chunkCount * sizeof(int));
  int *srcCuda, *dstCuda;
  cuda_safe_call(cudaMalloc(&srcCuda, arraySize * sizeof(int)));
  cuda_safe_call(cudaMalloc(&dstCuda, chunkSize * chunkCount * sizeof(int)));

  // set chunks to 1 and gaps to -1
  for (unsigned int i = 0; i < arraySize; i++) {
    src[i] = (i % (chunkSize + gapSize) < chunkSize) ? 1 : -1;
  }

  cuda_safe_call(cudaMemcpy(srcCuda, src, arraySize * sizeof(int), cudaMemcpyHostToDevice));
  _checkDstKernel<<<1, blockSize>>>(dstCuda, srcCuda, chunkSize, gapSize, chunkCount);
  cuda_safe_call(cudaDeviceSynchronize());
  cuda_safe_call(
      cudaMemcpy(dst, dstCuda, chunkSize * chunkCount * sizeof(int), cudaMemcpyDeviceToHost));

  for (unsigned int i = 0; i < chunkSize * chunkCount; i++) {
    RC_ASSERT(dst[i] == 1);
  }

  free(src);
  free(dst);
  cuda_safe_call(cudaFree(srcCuda));
  cuda_safe_call(cudaFree(dstCuda));
}

__global__ void _sharedMemoryKernel(int const *src, unsigned int chunkSize, unsigned int gapSize) {
  extern __shared__ int dst[];
  copyChunksToConsecutiveAsBlock<int>(dst, src, chunkSize, gapSize, blockDim.x);
}

RC_GTEST_PROP(CopyChunksToConsecutiveAsBlockTest, SharedMemory, ()) {
  unsigned int chunkSize = *rc::gen::inRange(1, 10);
  unsigned int gapSize = *rc::gen::inRange(0, 10);
  unsigned int blockSize = *rc::gen::inRange(1, 256);

  unsigned int arraySize = blockSize * (chunkSize + gapSize);
  int *src = (int *)malloc(arraySize * sizeof(int));
  int *srcCuda;
  cuda_safe_call(cudaMalloc(&srcCuda, arraySize * sizeof(int)));

  // set chunks to 1 and gaps to -1
  for (unsigned int i = 0; i < arraySize; i++) {
    src[i] = (i % (chunkSize + gapSize) < chunkSize) ? 1 : -1;
  }

  cuda_safe_call(cudaMemcpy(srcCuda, src, arraySize * sizeof(int), cudaMemcpyHostToDevice));
  _sharedMemoryKernel<<<1, blockSize, blockSize * chunkSize * sizeof(int)>>>(srcCuda, chunkSize,
                                                                             gapSize);
  cuda_safe_call(cudaDeviceSynchronize());

  // We don't actually want to assert anything here. Just that dst can
  // be in shared memory.
  RC_SUCCEED();

  free(src);
  cuda_safe_call(cudaFree(srcCuda));
}

RC_GTEST_PROP(ReorderVectorByComponentTest, CheckDst, ()) {
  unsigned int chunkSize = *rc::gen::inRange(1, 10);
  unsigned int chunkCount = *rc::gen::inRange(1, 1024);
  unsigned int blockSize = *rc::gen::inRange(1, 256);
  unsigned int gridSize = minGridSizeForN(chunkCount, blockSize);

  unsigned int arraySize = chunkCount * chunkSize;
  int *src = (int *)malloc(arraySize * sizeof(int));
  int *dst = (int *)malloc(arraySize * sizeof(int));
  int *srcCuda, *dstCuda;
  cuda_safe_call(cudaMalloc(&srcCuda, arraySize * sizeof(int)));
  cuda_safe_call(cudaMalloc(&dstCuda, arraySize * sizeof(int)));

  // set elements in chunks to ascending numbers
  for (unsigned int i = 0; i < arraySize; i++) {
    src[i] = i % chunkSize;
  }

  cuda_safe_call(cudaMemcpy(srcCuda, src, arraySize * sizeof(int), cudaMemcpyHostToDevice));
  reorderVectorByComponent<<<gridSize, blockSize>>>(dstCuda, srcCuda, chunkSize, chunkCount);
  cuda_safe_call(cudaDeviceSynchronize());
  cuda_safe_call(cudaMemcpy(dst, dstCuda, arraySize * sizeof(int), cudaMemcpyDeviceToHost));

  for (unsigned int i = 0; i < arraySize; i++) {
    RC_ASSERT(dst[i] == i / chunkCount);
  }

  free(src);
  free(dst);
  cuda_safe_call(cudaFree(srcCuda));
  cuda_safe_call(cudaFree(dstCuda));
}

RC_GTEST_PROP(ReorderVectorByComponentTest, SrcPreserved, ()) {
  unsigned int chunkSize = *rc::gen::inRange(1, 10);
  unsigned int chunkCount = *rc::gen::inRange(1, 1024);
  unsigned int blockSize = *rc::gen::inRange(1, 256);
  unsigned int gridSize = minGridSizeForN(chunkCount, blockSize);

  unsigned int arraySize = chunkCount * chunkSize;
  int *src = (int *)malloc(arraySize * sizeof(int));
  int *dst = (int *)malloc(arraySize * sizeof(int));
  int *srcCuda, *dstCuda;
  cuda_safe_call(cudaMalloc(&srcCuda, arraySize * sizeof(int)));
  cuda_safe_call(cudaMalloc(&dstCuda, arraySize * sizeof(int)));

  // set elements in chunks to ascending numbers
  for (unsigned int i = 0; i < arraySize; i++) {
    src[i] = i % chunkSize;
  }

  cuda_safe_call(cudaMemcpy(srcCuda, src, arraySize * sizeof(int), cudaMemcpyHostToDevice));
  reorderVectorByComponent<<<gridSize, blockSize>>>(dstCuda, srcCuda, chunkSize, chunkCount);
  cuda_safe_call(cudaDeviceSynchronize());
  cuda_safe_call(cudaMemcpy(dst, srcCuda, arraySize * sizeof(int), cudaMemcpyDeviceToHost));

  for (unsigned int i = 0; i < arraySize; i++) {
    RC_ASSERT(dst[i] == src[i]);
  }

  free(src);
  free(dst);
  cuda_safe_call(cudaFree(srcCuda));
  cuda_safe_call(cudaFree(dstCuda));
}
