/** \file cuda_componentwise.h
 *
 *  \brief Contains functions that support componentwise access to global memory.
 *
 * A vector is considered to be in chunkwise format if it is ordered like
 * 123412341234.
 *
 * A vector is considered to be in componentwise format if it ordered like
 * 111222333444.
 *
 * Access to a componentwise vector is more efficient in CUDA code if all
 * threads access the same component as more values fit each cache line and
 * thus the amount of data that needs to be transferred from global memory
 * (or another cache) is reduced.
 */

#ifndef CUDA_COMPONENTWISE_H
#define CUDA_COMPONENTWISE_H

#include <assert.h>

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
__device__ void copyChunksToConsecutiveAsBlock(ElementType* dst, ElementType const* src,
                                               unsigned int chunkSize, unsigned int gapSize,
                                               unsigned int chunkCount) {
  size_t elementCount = chunkCount * chunkSize;
  size_t iterationCount = (elementCount - 1) / blockDim.x + 1;
  for (size_t i = 0; i < iterationCount; i++) {
    size_t consecutiveIdx = (i * blockDim.x) + threadIdx.x;
    // thread does not need to handle another value
    if (consecutiveIdx >= chunkSize * chunkCount) {
      break;
    }
    size_t chunkIdx = consecutiveIdx / chunkSize;
    size_t valueIdx = consecutiveIdx % chunkSize;
    dst[consecutiveIdx] = src[chunkIdx * (chunkSize + gapSize) + valueIdx];
  }
  __syncthreads();
}

/**
 * \brief Simplifies access to components of a componentwise vector.
 *
 * Objects of this class can be used to access data[i * num_sites]
 * as ComponentAccess(data, num_sites)[i]. The created object is reusable to
 * eliminate the excessive writing of the num_sites variable.
 */
template <typename ElementType>
class ComponentAccess {
 public:
  /** Constructs the wrapper object.
   * 
   *  Ownership of data is not assumed. Caller must ensure that operator[]
   *  is never called after data is no longer valid.
   * 
   *  \param[in]  data        A pointer to a vector in componentwise ordering.
   *  \param[in] num_sites   The number of lattice sites which are stored in data.
   */
  __host__ __device__ ComponentAccess(ElementType* data, size_t num_sites) {
    this->data = data;
    this->num_sites = num_sites;
  }

  /** 
   *  \returns data[i * num_sites]
   */
  __device__ ElementType& operator[](size_t i) {
    return data[i * this->num_sites];
  }

 private:
  ElementType* data;
  size_t num_sites;
};

 /** \brief Reorder a full vector from chunkwise to componentwise ordering using
 *          arbitrary block and grid dimensions.
 *
 *  \param[out] dst         The vector that will be set to the componentwise reordering of src.
 *                          Must have at least size chunkSize * chunkCount.
 *  \param[in]  src         The vector in subvector format that will be reordered. Remains
 *                          unchanged.
 *  \param[in]  chunkSize   The count of components in the src vector.
 *  \param[in]  chunkCount  The count of subvectors/chunks/sites in the src vector.
 */
template <typename ElementType>
__global__ void reorderArrayByComponent(ElementType* dst, ElementType const* src, size_t chunkSize,
                                         size_t chunkCount) {
  assert(blockDim.x * gridDim.x >= chunkCount);
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= chunkCount) {
    // there is no more chunk for this index
    return;
  }
  // set src to first element that will be read
  src += chunkSize * idx;
  auto caDst = ComponentAccess(dst + idx, chunkCount);
  for (size_t i = 0; i < chunkSize; i++) {
    caDst[i] = src[i];
  }
}

/** \brief Reorder a vector that has gaps (uninteresting elements in between) by component.
 * 
 *  \param[out] dst         The vector that will be set to the componentwise reordering of src.
 *                          Must have at least size chunkSize * chunkCount.
 *  \param[in]  src         The vector in subvector format that will be reordered. Remains
 *                          unchanged. Is alternating between chunkSize interesting
 *                          and gapSize unintersting elements.
 *  \param[in]  chunkSize   The count of components in the src vector.
 *  \param[in]  gapSize     The count of elements in between chunks.
 *  \param[in]  chunkCount  The count of subvectors/chunks/sites in the src vector.
 */
template <typename ElementType>
__global__ void reorderArrayWithGapsByComponent(ElementType* dst, ElementType const* src,
                                                 unsigned int chunkSize, unsigned int gapSize,
                                                 unsigned int chunkCount) {
  assert(blockDim.x * gridDim.x >= chunkCount);
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= chunkCount) {
    // there is no more chunk for this index
    return;
  }
  // set src to first element that will be read (start of handled chunk)
  src += (chunkSize + gapSize) * idx;
  auto caDst = ComponentAccess(dst + idx, chunkCount);
  for (size_t i = 0; i < chunkSize; i++) {
    caDst[i] = src[i];
  }
}

/** \brief Reorder a full vector from componentwise to chunkwise ordering using
 *         arbitrary block and grid dimensions.
 *
 *  \param[out] dst         The vector that will be set to the componentwise reordering of src.
 *                          Must have at least size chunkSize * chunkCount.
 *  \param[in]  src         The vector in subvector format that will be reordered. Remains
 *                          unchanged.
 *  \param[in]  chunkSize   The count of components in the src vector.
 *  \param[in]  chunkCount  The count of subvectors/chunks/sites in the src vector.
 */
template <typename ElementType>
__global__ void reorderArrayByChunks(ElementType* dst, ElementType const* src, size_t chunkSize,
                                      size_t chunkCount) {
  assert(blockDim.x * gridDim.x >= chunkCount);
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= chunkCount) {
    // there is no more chunk for this index
    return;
  }
  // set dst to first element that will be written
  dst += chunkSize * idx;
  auto caSrc = ComponentAccess(src + idx, chunkCount);
  for (size_t i = 0; i < chunkSize; i++) {
    dst[i] = caSrc[i];
  }
}

#endif  // CUDA_COMPONENTWISE_H
