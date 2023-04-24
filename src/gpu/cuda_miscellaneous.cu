#include <mpi.h>
#include "global_struct.h"

extern "C" void get_device_properties() {
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, g.device_id);
  g.warp_size = devProp.warpSize;
}

extern "C" size_t minGridSizeForN(size_t n, size_t blockSize) {
  size_t incompleteBlocks = (n % blockSize == 0) ? 0 : 1;
  return (n / blockSize) + incompleteBlocks;
}
