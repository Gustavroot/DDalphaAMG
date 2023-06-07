#include "cuda_complex.h"
#include "cuda_complex_cxx.h"
#include "cuda_complex_operators.h"
#include "cuda_complex_operators_PRECISION.h"
#include "cuda_dirac_kernels_componentwise_PRECISION.h"
#include "cuda_mvm_PRECISION.h"
#include "global_enums.h"
#include "cuda_componentwise.h"

// The clifford header uses a C compiler extension version of I that is not compatible with CUDA.
// CU_OVERWRITE_I replaces that.
#define CU_OVERWRITE_I
#include "clifford.h"
#undef CU_OVERWRITE_I

__global__ void cuda_site_clover_componentwise_PRECISION(cuda_vector_PRECISION eta,
                                                         cu_cmplx_PRECISION const* phi,
                                                         cu_cmplx_PRECISION const* clover,
                                                         size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  eta += 12 * idx;
  auto caPhi = ComponentAccess(phi + idx, num_sites);
  // clover += idx;
  auto caClover = ComponentAccess(clover + idx, num_sites);

  // diagonal
  eta[0] = caClover[0] * caPhi[0];
  eta[1] = caClover[1] * caPhi[1];
  eta[2] = caClover[2] * caPhi[2];
  eta[3] = caClover[3] * caPhi[3];
  eta[4] = caClover[4] * caPhi[4];
  eta[5] = caClover[5] * caPhi[5];
  eta[6] = caClover[6] * caPhi[6];
  eta[7] = caClover[7] * caPhi[7];
  eta[8] = caClover[8] * caPhi[8];
  eta[9] = caClover[9] * caPhi[9];
  eta[10] = caClover[10] * caPhi[10];
  eta[11] = caClover[11] * caPhi[11];
  // spin 0 and 1, row major
  eta[0] += caClover[12] * caPhi[1];
  eta[0] += caClover[13] * caPhi[2];
  eta[0] += caClover[14] * caPhi[3];
  eta[0] += caClover[15] * caPhi[4];
  eta[0] += caClover[16] * caPhi[5];
  eta[1] += caClover[17] * caPhi[2];
  eta[1] += caClover[18] * caPhi[3];
  eta[1] += caClover[19] * caPhi[4];
  eta[1] += caClover[20] * caPhi[5];
  eta[2] += caClover[21] * caPhi[3];
  eta[2] += caClover[22] * caPhi[4];
  eta[2] += caClover[23] * caPhi[5];
  eta[3] += caClover[24] * caPhi[4];
  eta[3] += caClover[25] * caPhi[5];
  eta[4] += caClover[26] * caPhi[5];
  eta[1] += cu_conj_PRECISION(caClover[12]) * caPhi[0];
  eta[2] += cu_conj_PRECISION(caClover[13]) * caPhi[0];
  eta[3] += cu_conj_PRECISION(caClover[14]) * caPhi[0];
  eta[4] += cu_conj_PRECISION(caClover[15]) * caPhi[0];
  eta[5] += cu_conj_PRECISION(caClover[16]) * caPhi[0];
  eta[2] += cu_conj_PRECISION(caClover[17]) * caPhi[1];
  eta[3] += cu_conj_PRECISION(caClover[18]) * caPhi[1];
  eta[4] += cu_conj_PRECISION(caClover[19]) * caPhi[1];
  eta[5] += cu_conj_PRECISION(caClover[20]) * caPhi[1];
  eta[3] += cu_conj_PRECISION(caClover[21]) * caPhi[2];
  eta[4] += cu_conj_PRECISION(caClover[22]) * caPhi[2];
  eta[5] += cu_conj_PRECISION(caClover[23]) * caPhi[2];
  eta[4] += cu_conj_PRECISION(caClover[24]) * caPhi[3];
  eta[5] += cu_conj_PRECISION(caClover[25]) * caPhi[3];
  eta[5] += cu_conj_PRECISION(caClover[26]) * caPhi[4];
  // spin 2 and 3, row major
  eta[6] += caClover[27] * caPhi[7];
  eta[6] += caClover[28] * caPhi[8];
  eta[6] += caClover[29] * caPhi[9];
  eta[6] += caClover[30] * caPhi[10];
  eta[6] += caClover[31] * caPhi[11];
  eta[7] += caClover[32] * caPhi[8];
  eta[7] += caClover[33] * caPhi[9];
  eta[7] += caClover[34] * caPhi[10];
  eta[7] += caClover[35] * caPhi[11];
  eta[8] += caClover[36] * caPhi[9];
  eta[8] += caClover[37] * caPhi[10];
  eta[8] += caClover[38] * caPhi[11];
  eta[9] += caClover[39] * caPhi[10];
  eta[9] += caClover[40] * caPhi[11];
  eta[10] += caClover[41] * caPhi[11];
  eta[7] += cu_conj_PRECISION(caClover[27]) * caPhi[6];
  eta[8] += cu_conj_PRECISION(caClover[28]) * caPhi[6];
  eta[9] += cu_conj_PRECISION(caClover[29]) * caPhi[6];
  eta[10] += cu_conj_PRECISION(caClover[30]) * caPhi[6];
  eta[11] += cu_conj_PRECISION(caClover[31]) * caPhi[6];
  eta[8] += cu_conj_PRECISION(caClover[32]) * caPhi[7];
  eta[9] += cu_conj_PRECISION(caClover[33]) * caPhi[7];
  eta[10] += cu_conj_PRECISION(caClover[34]) * caPhi[7];
  eta[11] += cu_conj_PRECISION(caClover[35]) * caPhi[7];
  eta[9] += cu_conj_PRECISION(caClover[36]) * caPhi[8];
  eta[10] += cu_conj_PRECISION(caClover[37]) * caPhi[8];
  eta[11] += cu_conj_PRECISION(caClover[38]) * caPhi[8];
  eta[10] += cu_conj_PRECISION(caClover[39]) * caPhi[9];
  eta[11] += cu_conj_PRECISION(caClover[40]) * caPhi[9];
  eta[11] += cu_conj_PRECISION(caClover[41]) * caPhi[10];
}

__global__ void cuda_prp_T_componentwise_PRECISION(cu_cmplx_PRECISION* prpT,
                                                   cu_cmplx_PRECISION const* phi,
                                                   size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caPhi = ComponentAccess(phi + idx, num_sites);
  prpT += 6 * idx;
  prpT[0] = caPhi[0] - GAMMA_T_SPIN0_VAL * caPhi[3 * GAMMA_T_SPIN0_CO + 0];
  prpT[1] = caPhi[1] - GAMMA_T_SPIN0_VAL * caPhi[3 * GAMMA_T_SPIN0_CO + 1];
  prpT[2] = caPhi[2] - GAMMA_T_SPIN0_VAL * caPhi[3 * GAMMA_T_SPIN0_CO + 2];
  prpT[3] = caPhi[3] - GAMMA_T_SPIN1_VAL * caPhi[3 * GAMMA_T_SPIN1_CO + 0];
  prpT[4] = caPhi[4] - GAMMA_T_SPIN1_VAL * caPhi[3 * GAMMA_T_SPIN1_CO + 1];
  prpT[5] = caPhi[5] - GAMMA_T_SPIN1_VAL * caPhi[3 * GAMMA_T_SPIN1_CO + 2];
}

__global__ void cuda_prn_T_componentwise_PRECISION(cu_cmplx_PRECISION* prnT,
                                                   cu_cmplx_PRECISION const* phi,
                                                   size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caPhi = ComponentAccess(phi + idx, num_sites);
  prnT += 6 * idx;
  prnT[0] = caPhi[0] + GAMMA_T_SPIN0_VAL * caPhi[3 * GAMMA_T_SPIN0_CO + 0];
  prnT[1] = caPhi[1] + GAMMA_T_SPIN0_VAL * caPhi[3 * GAMMA_T_SPIN0_CO + 1];
  prnT[2] = caPhi[2] + GAMMA_T_SPIN0_VAL * caPhi[3 * GAMMA_T_SPIN0_CO + 2];
  prnT[3] = caPhi[3] + GAMMA_T_SPIN1_VAL * caPhi[3 * GAMMA_T_SPIN1_CO + 0];
  prnT[4] = caPhi[4] + GAMMA_T_SPIN1_VAL * caPhi[3 * GAMMA_T_SPIN1_CO + 1];
  prnT[5] = caPhi[5] + GAMMA_T_SPIN1_VAL * caPhi[3 * GAMMA_T_SPIN1_CO + 2];
}

__global__ void cuda_prp_Z_componentwise_PRECISION(cu_cmplx_PRECISION* prpZ,
                                                   cu_cmplx_PRECISION const* phi,
                                                   size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caPhi = ComponentAccess(phi + idx, num_sites);
  prpZ += 6 * idx;
  prpZ[0] = caPhi[0] - GAMMA_Z_SPIN0_VAL * caPhi[3 * GAMMA_Z_SPIN0_CO + 0];
  prpZ[1] = caPhi[1] - GAMMA_Z_SPIN0_VAL * caPhi[3 * GAMMA_Z_SPIN0_CO + 1];
  prpZ[2] = caPhi[2] - GAMMA_Z_SPIN0_VAL * caPhi[3 * GAMMA_Z_SPIN0_CO + 2];
  prpZ[3] = caPhi[3] - GAMMA_Z_SPIN1_VAL * caPhi[3 * GAMMA_Z_SPIN1_CO + 0];
  prpZ[4] = caPhi[4] - GAMMA_Z_SPIN1_VAL * caPhi[3 * GAMMA_Z_SPIN1_CO + 1];
  prpZ[5] = caPhi[5] - GAMMA_Z_SPIN1_VAL * caPhi[3 * GAMMA_Z_SPIN1_CO + 2];
}

__global__ void cuda_prn_Z_componentwise_PRECISION(cu_cmplx_PRECISION* prnZ,
                                                   cu_cmplx_PRECISION const* phi,
                                                   size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caPhi = ComponentAccess(phi + idx, num_sites);
  prnZ += 6 * idx;
  prnZ[0] = caPhi[0] + GAMMA_Z_SPIN0_VAL * caPhi[3 * GAMMA_Z_SPIN0_CO + 0];
  prnZ[1] = caPhi[1] + GAMMA_Z_SPIN0_VAL * caPhi[3 * GAMMA_Z_SPIN0_CO + 1];
  prnZ[2] = caPhi[2] + GAMMA_Z_SPIN0_VAL * caPhi[3 * GAMMA_Z_SPIN0_CO + 2];
  prnZ[3] = caPhi[3] + GAMMA_Z_SPIN1_VAL * caPhi[3 * GAMMA_Z_SPIN1_CO + 0];
  prnZ[4] = caPhi[4] + GAMMA_Z_SPIN1_VAL * caPhi[3 * GAMMA_Z_SPIN1_CO + 1];
  prnZ[5] = caPhi[5] + GAMMA_Z_SPIN1_VAL * caPhi[3 * GAMMA_Z_SPIN1_CO + 2];
}

__global__ void cuda_prp_Y_componentwise_PRECISION(cu_cmplx_PRECISION* prpY,
                                                   cu_cmplx_PRECISION const* phi,
                                                   size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caPhi = ComponentAccess(phi + idx, num_sites);
  prpY += 6 * idx;
  prpY[0] = caPhi[0] - GAMMA_Y_SPIN0_VAL * caPhi[3 * GAMMA_Y_SPIN0_CO + 0];
  prpY[1] = caPhi[1] - GAMMA_Y_SPIN0_VAL * caPhi[3 * GAMMA_Y_SPIN0_CO + 1];
  prpY[2] = caPhi[2] - GAMMA_Y_SPIN0_VAL * caPhi[3 * GAMMA_Y_SPIN0_CO + 2];
  prpY[3] = caPhi[3] - GAMMA_Y_SPIN1_VAL * caPhi[3 * GAMMA_Y_SPIN1_CO + 0];
  prpY[4] = caPhi[4] - GAMMA_Y_SPIN1_VAL * caPhi[3 * GAMMA_Y_SPIN1_CO + 1];
  prpY[5] = caPhi[5] - GAMMA_Y_SPIN1_VAL * caPhi[3 * GAMMA_Y_SPIN1_CO + 2];
}

__global__ void cuda_prn_Y_componentwise_PRECISION(cu_cmplx_PRECISION* prnY,
                                                   cu_cmplx_PRECISION const* phi,
                                                   size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caPhi = ComponentAccess(phi + idx, num_sites);
  prnY += 6 * idx;
  prnY[0] = caPhi[0] + GAMMA_Y_SPIN0_VAL * caPhi[3 * GAMMA_Y_SPIN0_CO + 0];
  prnY[1] = caPhi[1] + GAMMA_Y_SPIN0_VAL * caPhi[3 * GAMMA_Y_SPIN0_CO + 1];
  prnY[2] = caPhi[2] + GAMMA_Y_SPIN0_VAL * caPhi[3 * GAMMA_Y_SPIN0_CO + 2];
  prnY[3] = caPhi[3] + GAMMA_Y_SPIN1_VAL * caPhi[3 * GAMMA_Y_SPIN1_CO + 0];
  prnY[4] = caPhi[4] + GAMMA_Y_SPIN1_VAL * caPhi[3 * GAMMA_Y_SPIN1_CO + 1];
  prnY[5] = caPhi[5] + GAMMA_Y_SPIN1_VAL * caPhi[3 * GAMMA_Y_SPIN1_CO + 2];
}

__global__ void cuda_prp_X_componentwise_PRECISION(cu_cmplx_PRECISION* prpX,
                                                   cu_cmplx_PRECISION const* phi,
                                                   size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caPhi = ComponentAccess(phi + idx, num_sites);
  prpX += 6 * idx;
  prpX[0] = caPhi[0] - GAMMA_X_SPIN0_VAL * caPhi[3 * GAMMA_X_SPIN0_CO + 0];
  prpX[1] = caPhi[1] - GAMMA_X_SPIN0_VAL * caPhi[3 * GAMMA_X_SPIN0_CO + 1];
  prpX[2] = caPhi[2] - GAMMA_X_SPIN0_VAL * caPhi[3 * GAMMA_X_SPIN0_CO + 2];
  prpX[3] = caPhi[3] - GAMMA_X_SPIN1_VAL * caPhi[3 * GAMMA_X_SPIN1_CO + 0];
  prpX[4] = caPhi[4] - GAMMA_X_SPIN1_VAL * caPhi[3 * GAMMA_X_SPIN1_CO + 1];
  prpX[5] = caPhi[5] - GAMMA_X_SPIN1_VAL * caPhi[3 * GAMMA_X_SPIN1_CO + 2];
}

__global__ void cuda_prn_X_componentwise_PRECISION(cu_cmplx_PRECISION* prnX,
                                                   cu_cmplx_PRECISION const* phi,
                                                   size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caPhi = ComponentAccess(phi + idx, num_sites);
  prnX += 6 * idx;
  prnX[0] = caPhi[0] + GAMMA_X_SPIN0_VAL * caPhi[3 * GAMMA_X_SPIN0_CO + 0];
  prnX[1] = caPhi[1] + GAMMA_X_SPIN0_VAL * caPhi[3 * GAMMA_X_SPIN0_CO + 1];
  prnX[2] = caPhi[2] + GAMMA_X_SPIN0_VAL * caPhi[3 * GAMMA_X_SPIN0_CO + 2];
  prnX[3] = caPhi[3] + GAMMA_X_SPIN1_VAL * caPhi[3 * GAMMA_X_SPIN1_CO + 0];
  prnX[4] = caPhi[4] + GAMMA_X_SPIN1_VAL * caPhi[3 * GAMMA_X_SPIN1_CO + 1];
  prnX[5] = caPhi[5] + GAMMA_X_SPIN1_VAL * caPhi[3 * GAMMA_X_SPIN1_CO + 2];
}
