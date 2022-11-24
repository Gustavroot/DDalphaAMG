#ifndef PROFILING_H
#define PROFILING_H

#ifdef CUDA_OPT
#include <nvtx3/nvToolsExt.h>
typedef nvtxRangeId_t RangeHandleType;
#else
typedef void * RangeHandleType;
#endif //CUDA_OPT

RangeHandleType startProfilingRange(char const * label);
void endProfilingRange(RangeHandleType handle);


#endif //PROFILING_H