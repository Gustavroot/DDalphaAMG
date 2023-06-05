#include "cuda_complex.h"
#include "cuda_complex_cxx.h"
#include "cuda_complex_operators.h"
#include "cuda_complex_operators_PRECISION.h"
#include "cuda_dirac_kernels_coalesced_PRECISION.h"
#include "cuda_mvm_PRECISION.h"
#include "global_enums.h"

// The clifford header uses a C compiler extension version of I that is not compatible with CUDA.
// CU_OVERWRITE_I replaces that.
#define CU_OVERWRITE_I
#include "clifford.h"
#undef CU_OVERWRITE_I

__global__ void cuda_prp_T_coalesced_PRECISION(cu_cmplx_PRECISION* prpT,
                                               cu_cmplx_PRECISION const* phi, size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  phi += idx;
  prpT += 6 * idx;
  prpT[0] = phi[0 * num_sites] - GAMMA_T_SPIN0_VAL * phi[(3 * GAMMA_T_SPIN0_CO) * num_sites];
  prpT[1] = phi[1 * num_sites] - GAMMA_T_SPIN0_VAL * phi[(3 * GAMMA_T_SPIN0_CO + 1) * num_sites];
  prpT[2] = phi[2 * num_sites] - GAMMA_T_SPIN0_VAL * phi[(3 * GAMMA_T_SPIN0_CO + 2) * num_sites];
  prpT[3] = phi[3 * num_sites] - GAMMA_T_SPIN1_VAL * phi[(3 * GAMMA_T_SPIN1_CO) * num_sites];
  prpT[4] = phi[4 * num_sites] - GAMMA_T_SPIN1_VAL * phi[(3 * GAMMA_T_SPIN1_CO + 1) * num_sites];
  prpT[5] = phi[5 * num_sites] - GAMMA_T_SPIN1_VAL * phi[(3 * GAMMA_T_SPIN1_CO + 2) * num_sites];
}