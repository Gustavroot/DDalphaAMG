/** \file profiling.h
 *  \brief Functions for profiling.
 * 
 *  This provides a wrapper in order to not pull CUDA/NVTX dependencies into
 *  the code where they are not needed otherwise. However these functions
 *  can be called without needing to link the application to CUDA/NVTX.
 *  In that case these functions do nothing.
 */

#ifndef PROFILING_H
#define PROFILING_H

/** nvtxRangeId_t if compiling with NVTX, a stub otherwise.*/
#ifndef NVTX_DISABLE
#include <nvtx3/nvToolsExt.h>
typedef nvtxRangeId_t RangeHandleType;
#else
typedef void * RangeHandleType;
#endif //CUDA_OPT

/** \brief Starts a range for purposes of profiling if enabled.
 * 
 *  This starts a range in the underlying implementation of NVTX
 *  if CUDA is enabled and does nothing otherwise.
 * 
 *  \param label A label to be given to that range.
 *  \returns A handle which can be used to close the range.
 *      Must be closed using endProfilingRange.
 */
RangeHandleType startProfilingRange(char const * label);

/** \brief Ends a range for purposes of profiling if enabled.
 * 
 * This ends a range in the underlying implementation of NVTX
 * if CUDA is enabled and does nothing otherwise.
 * 
 * \param handle A handle obtained from startProfilingRange.
 */
void endProfilingRange(RangeHandleType handle);


#endif //PROFILING_H
