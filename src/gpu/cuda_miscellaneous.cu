#include <mpi.h>

extern "C"{

  #define IMPORT_FROM_EXTERN_C
  #include "main.h"
  #undef IMPORT_FROM_EXTERN_C

}

#ifdef CUDA_OPT


extern "C" void get_device_properties(){

  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, g.device_id);
  g.warp_size = devProp.warpSize;

}


#endif
