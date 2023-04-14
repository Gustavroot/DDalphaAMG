#include "profiling.h"
#include <stddef.h>

#ifndef NVTX_DISABLE
#include <nvtx3/nvToolsExt.h>

RangeHandleType startProfilingRange(char const * label) {
    return nvtxRangeStartA(label);
}

void endProfilingRange(RangeHandleType handle) {
    nvtxRangeEnd(handle);
}

#else
// stub, so we can be agnostic to whether nvtx is available or not

RangeHandleType startProfilingRange(char const * label) {
    return NULL;
}

void endProfilingRange(RangeHandleType handle){
    return;
}
#endif  // NVTX_DISABLE
