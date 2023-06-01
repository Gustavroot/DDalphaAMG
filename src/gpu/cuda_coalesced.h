/** \file cuda_coalesced.h
 *
 *  \brief Contains functions that support coalesced access to global memory.
 */

#ifndef CUDA_COALESCED_H
#define CUDA_COALESCED_H

/** \brief Copy data from array that has gaps to one that is consecutive.
 *  
 *  Consider an array with chunks of 3 values of interest alternating with
 *  2 values that are not of interest (visually 123XX456XX789XX...).
 *  This function copies the values of interest consecutively into dst
 *  (visually 123456789).
 * 
 *  It does so by letting all threads of a block access values from src in
 *  a coalesced fashion (visually if threads are A, B, C and D src is
 *  accessed like ABCXXDABXXCDAXX...).
 * 
 *  Given that the size of each chunk is sufficiently large this greatly
 *  reduces load on global memory as the consecutive values can be fetched
 *  together.
 * 
 *  \param[out] dst         Array that data will be written to.
 *  \param[in]  src         Array that data will be read from.
 *  \param[in]  chunkSize   The number of consecutive elements within each chunk.
 *  \param[in]  gapSize     The numer of elements in the gap between chunks.
 *  \param[in]  chunkCount  The total number of chunks to copy.
 */
template <typename ElementType>
__device__ void copy_chunks_to_consecutive_as_block(ElementType* dst, ElementType const* src,
                                                    unsigned int chunkSize, unsigned int gapSize,
                                                    unsigned int chunkCount) {
  size_t elementCount = chunkCount * chunkSize;
  size_t iterationCount = elementCount / blockDim.x;
  // need another iteration to handle remaining elements.
  if (elementCount % blockDim.x != 0) {
    iterationCount++;
  }
  for (size_t i = 0; i < iterationCount; i++) {
    size_t consecutiveIdx = (i * blockDim.x) + threadIdx.x;
    // thread does not need to handle another value
    if (consecutiveIdx > chunkSize * chunkCount) {
      break;
    }
    size_t chunkIdx = consecutiveIdx / chunkSize;
    size_t valueIdx = consecutiveIdx % chunkSize;
    dst[consecutiveIdx] = src[chunkIdx * (chunkSize + gapSize) + valueIdx];
  }
  __syncthreads();
}

#endif  // CUDA_COALESCED_H