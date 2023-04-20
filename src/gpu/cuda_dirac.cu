#include "cuda_dirac.h"
#include "cuda_linalg_double.h"
extern "C" {
#include "operator.h"
}
#include <cuda.h>

#include "global_struct.h"

void cuda_dirac_setup(config_double hopp, config_double clover, level_struct* l) {
  cudaStream_t stream = CU_STREAM_PER_THREAD;
  cudaStream_t* const streams = &stream;
  const size_t css = clover_site_size(l->num_lattice_site_var, l->depth);
  // Float clover term does not seem to get filled (possibly because on lower levels, there)
  cuda_vector_double_copy(g.op_double.clover_gpu, g.op_double.clover, 0,
                          l->num_inner_lattice_sites * css, l, _H2D, _CUDA_SYNC, 0, streams);
}
