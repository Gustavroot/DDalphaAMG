/** \file cuda_componentwise.h
 *
 *  \brief Contains functions that support componentwise access to global memory.
 * 
 * A vector is considered to be in subvector format if it is ordered like
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

/** \brief Reorder a full vector by component using arbitrary block and grid dimensions.
 * 
 *  Transforms a vector from subvector format to componentwise format.
 * 
 *  \param[out] dst         The vector that will be set to the componentwise reordering of src.
 *                          Must have at least size chunkSize * chunkCount.
 *  \param[in]  src         The vector in subvector format that will be reordered. Remains unchanged.
 *  \param[in]  chunkSize   The count of components in the src vector.
 *  \param[in]  chunkCount  The count of subvectors/chunks/sites in the src vector. Equivalent to
 *                          the distance between components +1.
 * 
 */
template <typename ElementType>
__global__ void reorderVectorByComponent(ElementType* dst, ElementType const* src,
                                         size_t chunkSize, size_t chunkCount) {
  // integer division rounding up
  size_t chunksPerBlock = (chunkCount - 1) / gridDim.x + 1;
  size_t requiredBlocks = (chunkCount - 1) / chunksPerBlock + 1;
  if (blockIdx.x >= requiredBlocks) {
    return;
  }
  //set src on first element to read
  src += chunkSize * chunksPerBlock * blockIdx.x;
  //set dst to first element that will be written
  dst += blockIdx.x * chunksPerBlock;
  size_t handledChunks = chunksPerBlock;

  // last block might need to account for early end of array
  if (blockIdx.x == requiredBlocks - 1) {
    handledChunks = chunkCount - (blockIdx.x * chunksPerBlock);
  }
  for (uint i = 0; i < chunkSize; i++) {
    copyChunksToConsecutiveAsBlock(dst, src, 1, chunkSize - 1, handledChunks);
    src += 1;
    dst += chunkCount;
  }
}

/**
 * \brief Simplifies access to components of a componentwise vector.
 * 
 * Objects of this class can be used to access src[i * num_sites]
 * as ComponentAccess(src, num_sites)[i]. The created object is reusable to
 * eliminate the excessive writing of the num_sites variable.
 */
template <typename ElementType>
class ComponentAccess{
  public:
    __host__ __device__ ComponentAccess(ElementType * data, size_t num_sites){
      this->data = data;
      this->num_sites = num_sites;
    }

    __host__ __device__ ElementType& operator[](size_t i){
      return data[i * this->num_sites];
    }
  private:
    ElementType * data;
    size_t num_sites;
};

#endif  // CUDA_COMPONENTWISE_H