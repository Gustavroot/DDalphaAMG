/** \file cuda_miscellaneous.h
 *  \brief Miscellaneous support functions for CUDA computing.
 */
#ifdef CUDA_OPT
#ifndef MISCELLANEOUS_HEADER_CUDA
#define MISCELLANEOUS_HEADER_CUDA

void get_device_properties();

#ifdef __cplusplus
/**
 * \brief Calculate the minimum grid size (i.e. amount of blocks) for a given number of elements.
 *
 * Let n=3 and blockSize=2. There is one full block which captures the first two elements.
 *    The additional element is moved to the next block, where one thread will not have an element
 *    to process. Thus minGridSizeForN is 2!
 *
 * Let n=3 and blockSize=3. There is one full block which captures the first three
 *    elements. As all n elements are now captured, no additional (incomplete) block will be needed.
 *    Thus minGridSizeForN is 1!
 * \param n   Number of elements to be distributed over the blocks and grids.
 * \param blockSize   The number of elements within one block.
 * \returns The minimum grid size (i.e. amount of blocks) for a given number of elements.
 */
constexpr size_t minGridSizeForN(size_t n, size_t blockSize) { return (n - 1) / blockSize + 1; }
#endif

#endif
#endif
