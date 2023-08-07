#include <cuda.h>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include "gpu/cuda_componentwise.h"
#include "gpu/cuda_miscellaneous.h"
#include "miscellaneous.h"
#include "test_macros.h"

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
  RC_ASSERT_CUDA_SUCCESS(cudaMalloc(&srcCuda, arraySize * sizeof(int)));
  RC_ASSERT_CUDA_SUCCESS(cudaMalloc(&dstCuda, chunkSize * blockSize * sizeof(int)));

  // set chunks to 1 and gaps to -1
  for (unsigned int i = 0; i < arraySize; i++) {
    src[i] = (i % (chunkSize + gapSize) < chunkSize) ? 1 : -1;
  }

  RC_ASSERT_CUDA_SUCCESS(cudaMemcpy(srcCuda, src, arraySize * sizeof(int), cudaMemcpyHostToDevice));
  _checkDstKernel<<<1, blockSize>>>(dstCuda, srcCuda, chunkSize, gapSize, blockSize);
  RC_ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
  RC_ASSERT_CUDA_SUCCESS(
      cudaMemcpy(dst, dstCuda, chunkSize * blockSize * sizeof(int), cudaMemcpyDeviceToHost));

  for (unsigned int i = 0; i < chunkSize * blockSize; i++) {
    RC_ASSERT(dst[i] == 1);
  }

  free(src);
  free(dst);
  RC_ASSERT_CUDA_SUCCESS(cudaFree(srcCuda));
  RC_ASSERT_CUDA_SUCCESS(cudaFree(dstCuda));
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
  RC_ASSERT_CUDA_SUCCESS(cudaMalloc(&srcCuda, arraySize * sizeof(int)));
  RC_ASSERT_CUDA_SUCCESS(cudaMalloc(&dstCuda, chunkSize * chunkCount * sizeof(int)));

  // set chunks to 1 and gaps to -1
  for (unsigned int i = 0; i < arraySize; i++) {
    src[i] = (i % (chunkSize + gapSize) < chunkSize) ? 1 : -1;
  }

  RC_ASSERT_CUDA_SUCCESS(cudaMemcpy(srcCuda, src, arraySize * sizeof(int), cudaMemcpyHostToDevice));
  _checkDstKernel<<<1, blockSize>>>(dstCuda, srcCuda, chunkSize, gapSize, chunkCount);
  RC_ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
  RC_ASSERT_CUDA_SUCCESS(
      cudaMemcpy(dst, dstCuda, chunkSize * chunkCount * sizeof(int), cudaMemcpyDeviceToHost));

  for (unsigned int i = 0; i < chunkSize * chunkCount; i++) {
    RC_ASSERT(dst[i] == 1);
  }

  free(src);
  free(dst);
  RC_ASSERT_CUDA_SUCCESS(cudaFree(srcCuda));
  RC_ASSERT_CUDA_SUCCESS(cudaFree(dstCuda));
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
  RC_ASSERT_CUDA_SUCCESS(cudaMalloc(&srcCuda, arraySize * sizeof(int)));

  // set chunks to 1 and gaps to -1
  for (unsigned int i = 0; i < arraySize; i++) {
    src[i] = (i % (chunkSize + gapSize) < chunkSize) ? 1 : -1;
  }

  RC_ASSERT_CUDA_SUCCESS(cudaMemcpy(srcCuda, src, arraySize * sizeof(int), cudaMemcpyHostToDevice));
  _sharedMemoryKernel<<<1, blockSize, blockSize * chunkSize * sizeof(int)>>>(srcCuda, chunkSize,
                                                                             gapSize);
  RC_ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());

  // We don't actually want to assert anything here. Just that dst can
  // be in shared memory.
  RC_SUCCEED();

  free(src);
  RC_ASSERT_CUDA_SUCCESS(cudaFree(srcCuda));
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
  RC_ASSERT_CUDA_SUCCESS(cudaMalloc(&srcCuda, arraySize * sizeof(int)));
  RC_ASSERT_CUDA_SUCCESS(cudaMalloc(&dstCuda, arraySize * sizeof(int)));

  // set elements in chunks to ascending numbers
  for (unsigned int i = 0; i < arraySize; i++) {
    src[i] = i % chunkSize;
  }

  RC_ASSERT_CUDA_SUCCESS(cudaMemcpy(srcCuda, src, arraySize * sizeof(int), cudaMemcpyHostToDevice));
  reorderArrayByComponent<<<gridSize, blockSize>>>(dstCuda, srcCuda, chunkSize, chunkCount);
  RC_ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
  RC_ASSERT_CUDA_SUCCESS(cudaMemcpy(dst, dstCuda, arraySize * sizeof(int), cudaMemcpyDeviceToHost));

  for (unsigned int i = 0; i < arraySize; i++) {
    RC_ASSERT(dst[i] == i / chunkCount);
  }

  free(src);
  free(dst);
  RC_ASSERT_CUDA_SUCCESS(cudaFree(srcCuda));
  RC_ASSERT_CUDA_SUCCESS(cudaFree(dstCuda));
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
  RC_ASSERT_CUDA_SUCCESS(cudaMalloc(&srcCuda, arraySize * sizeof(int)));
  RC_ASSERT_CUDA_SUCCESS(cudaMalloc(&dstCuda, arraySize * sizeof(int)));

  // set elements in chunks to ascending numbers
  for (unsigned int i = 0; i < arraySize; i++) {
    src[i] = i % chunkSize;
  }

  RC_ASSERT_CUDA_SUCCESS(cudaMemcpy(srcCuda, src, arraySize * sizeof(int), cudaMemcpyHostToDevice));
  reorderArrayByComponent<<<gridSize, blockSize>>>(dstCuda, srcCuda, chunkSize, chunkCount);
  RC_ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
  RC_ASSERT_CUDA_SUCCESS(cudaMemcpy(dst, srcCuda, arraySize * sizeof(int), cudaMemcpyDeviceToHost));

  for (unsigned int i = 0; i < arraySize; i++) {
    RC_ASSERT(dst[i] == src[i]);
  }

  free(src);
  free(dst);
  RC_ASSERT_CUDA_SUCCESS(cudaFree(srcCuda));
  RC_ASSERT_CUDA_SUCCESS(cudaFree(dstCuda));
}

RC_GTEST_PROP(ReorderVectorWithGapsByComponentTest, CheckDst, ()) {
  unsigned int chunkSize = *rc::gen::inRange(1, 10);
  unsigned int gapSize = *rc::gen::inRange(1, 10);
  unsigned int chunkCount = *rc::gen::inRange(1, 1024);
  unsigned int blockSize = *rc::gen::inRange(1, 256);
  unsigned int gridSize = minGridSizeForN(chunkCount, blockSize);

  unsigned int arraySize = chunkCount * (chunkSize + gapSize);
  auto src = *rc::gen::container<std::vector<int>>(arraySize, rc::gen::arbitrary<int>());
  int *dst = (int *)malloc(chunkCount * chunkSize * sizeof(int));
  int *srcCuda, *dstCuda;
  RC_ASSERT_CUDA_SUCCESS(cudaMalloc(&srcCuda, arraySize * sizeof(int)));
  RC_ASSERT_CUDA_SUCCESS(cudaMalloc(&dstCuda, chunkCount * chunkSize * sizeof(int)));

  RC_ASSERT_CUDA_SUCCESS(cudaMemcpy(srcCuda, src.data(), arraySize * sizeof(int), cudaMemcpyHostToDevice));
  reorderArrayWithGapsByComponent<<<gridSize, blockSize>>>(dstCuda, srcCuda, chunkSize, gapSize,
                                                            chunkCount);
  RC_ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
  RC_ASSERT_CUDA_SUCCESS(
      cudaMemcpy(dst, dstCuda, chunkCount * chunkSize * sizeof(int), cudaMemcpyDeviceToHost));

  for (unsigned int i = 0; i < chunkCount * chunkSize; i++) {
    uint chunkIdx = i % chunkCount;
    uint valueIdx = i / chunkCount;
    RC_ASSERT(dst[i] == src[chunkIdx * (chunkSize + gapSize) + valueIdx]);
  }

  free(dst);
  RC_ASSERT_CUDA_SUCCESS(cudaFree(srcCuda));
  RC_ASSERT_CUDA_SUCCESS(cudaFree(dstCuda));
}

RC_GTEST_PROP(ReorderVectorByChunksTest, SrcRestored, ()) {
  unsigned int chunkSize = *rc::gen::inRange(1, 10);
  unsigned int chunkCount = *rc::gen::inRange(1, 1024);
  unsigned int blockSize = *rc::gen::inRange(1, 256);
  unsigned int gridSize = minGridSizeForN(chunkCount, blockSize);

  unsigned int arraySize = chunkCount * chunkSize;
  auto src = *rc::gen::container<std::vector<int>>(arraySize, rc::gen::arbitrary<int>());
  int *dst = (int *)malloc(arraySize * sizeof(int));
  int *srcCuda, *srcComponentwiseCuda, *dstCuda;
  RC_ASSERT_CUDA_SUCCESS(cudaMalloc(&srcCuda, arraySize * sizeof(int)));
  RC_ASSERT_CUDA_SUCCESS(cudaMalloc(&srcComponentwiseCuda, arraySize * sizeof(int)));
  RC_ASSERT_CUDA_SUCCESS(cudaMalloc(&dstCuda, arraySize * sizeof(int)));

  RC_ASSERT_CUDA_SUCCESS(
      cudaMemcpy(srcCuda, src.data(), arraySize * sizeof(int), cudaMemcpyHostToDevice));
  reorderArrayByComponent<<<gridSize, blockSize>>>(srcComponentwiseCuda, srcCuda, chunkSize,
                                                    chunkCount);
  reorderArrayByChunks<<<gridSize, blockSize>>>(dstCuda, srcComponentwiseCuda, chunkSize,
                                                 chunkCount);
  RC_ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
  RC_ASSERT_CUDA_SUCCESS(cudaMemcpy(dst, dstCuda, arraySize * sizeof(int), cudaMemcpyDeviceToHost));

  for (unsigned int i = 0; i < arraySize; i++) {
    RC_ASSERT(dst[i] == src[i]);
  }

  free(dst);
  RC_ASSERT_CUDA_SUCCESS(cudaFree(srcCuda));
  RC_ASSERT_CUDA_SUCCESS(cudaFree(srcComponentwiseCuda));
  RC_ASSERT_CUDA_SUCCESS(cudaFree(dstCuda));
}

__global__ void _compareOriginalKernel(int *dst, int const *srcComponentwise, uint siteCount) {
  auto caSrc = ComponentAccess(srcComponentwise, siteCount);
  dst[threadIdx.x] = caSrc[threadIdx.x];
}

RC_GTEST_PROP(ComponentAccessTest, CompareOriginal, ()) {
  unsigned int componentCount = *rc::gen::inRange(1, 10);
  unsigned int siteCount = *rc::gen::inRange(1, 1024);
  unsigned int siteIdx = *rc::gen::inRange(0u, siteCount - 1);
  constexpr unsigned int blockSize = 32;
  unsigned int gridSize = minGridSizeForN(siteCount, blockSize);

  unsigned int arraySize = siteCount * componentCount;
  auto src = *rc::gen::container<std::vector<int>>(arraySize, rc::gen::arbitrary<int>());
  int *dst = (int *)malloc(componentCount * sizeof(int));
  int *srcCuda, *srcComponentwiseCuda, *dstCuda;
  RC_ASSERT_CUDA_SUCCESS(cudaMalloc(&srcCuda, arraySize * sizeof(int)));
  RC_ASSERT_CUDA_SUCCESS(cudaMalloc(&srcComponentwiseCuda, arraySize * sizeof(int)));
  RC_ASSERT_CUDA_SUCCESS(cudaMalloc(&dstCuda, componentCount * sizeof(int)));

  RC_ASSERT_CUDA_SUCCESS(
      cudaMemcpy(srcCuda, src.data(), arraySize * sizeof(int), cudaMemcpyHostToDevice));
  reorderArrayByComponent<<<gridSize, blockSize>>>(srcComponentwiseCuda, srcCuda, componentCount,
                                                    siteCount);
  // note `+ siteIdx` -> we access components for site at siteIdx
  _compareOriginalKernel<<<1, componentCount>>>(dstCuda, srcComponentwiseCuda + siteIdx, siteCount);
  RC_ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
  RC_ASSERT_CUDA_SUCCESS(
      cudaMemcpy(dst, dstCuda, componentCount * sizeof(int), cudaMemcpyDeviceToHost));

  for (unsigned int i = 0; i < componentCount; i++) {
    RC_ASSERT(dst[i] == src[siteIdx * componentCount + i]);
  }
  free(dst);
  RC_ASSERT_CUDA_SUCCESS(cudaFree(srcCuda));
  RC_ASSERT_CUDA_SUCCESS(cudaFree(srcComponentwiseCuda));
  RC_ASSERT_CUDA_SUCCESS(cudaFree(dstCuda));
}
