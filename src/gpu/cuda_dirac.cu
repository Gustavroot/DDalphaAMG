#include "cuda_componentwise.h"
#include "cuda_dirac.h"
#include "cuda_linalg_double.h"
#include "cuda_miscellaneous.h"
#include "miscellaneous.h"
extern "C" {
#include "operator.h"
}
#include <cuda.h>

#include "global_struct.h"

void cuda_dirac_setup(config_double hopp, config_double clover, level_struct* l) {
  cudaStream_t stream = CU_STREAM_PER_THREAD;
  cudaStream_t* const streams = &stream;
  const size_t css = clover_site_size(l->num_lattice_site_var, l->depth);
  // Float clover and Dirac data does not seem to get filled
  // (possibly because on lower levels, there are different methods handling that)
  cuda_vector_double_copy(g.op_double.clover_gpu, g.op_double.clover, 0,
                          l->num_inner_lattice_sites * css, l, _H2D, _CUDA_SYNC, 0, streams);
  constexpr uint blockSize = 128;
  uint gridSize = minGridSizeForN(l->num_inner_lattice_sites, blockSize);
  reorderArrayByComponent<<<gridSize, 128>>>(g.op_double.clover_componentwise_gpu,
                                              g.op_double.clover_gpu, css,
                                              l->num_inner_lattice_sites);
  cuda_vector_double_copy(g.op_double.D_gpu, g.op_double.D, 0, 4 * 9 * l->num_inner_lattice_sites,
                          l, _H2D, _CUDA_SYNC, 0, streams);
  gridSize = minGridSizeForN(9 * l->num_inner_lattice_sites, blockSize);
  for (int mu = 0; mu < 4; mu++) {
    reorderArrayWithGapsByComponent<<<gridSize, blockSize>>>(g.op_double.Ds_componentwise_gpu[mu],
                                                              g.op_double.D_gpu + (9 * mu), 9,
                                                              3 * 9, l->num_inner_lattice_sites);
  };
  cuda_safe_call(cudaDeviceSynchronize());
}
