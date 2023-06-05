#include <mpi.h>
#include "global_struct.h"

extern "C" void get_device_properties() {
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, g.device_id);
  g.warp_size = devProp.warpSize;
}
