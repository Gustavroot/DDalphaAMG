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
  phi += idx;
  // clover += idx;
  auto coClover = ComponentAccess(clover + idx, num_sites);

  // diagonal
  eta[0] = coClover[0] * phi[0 * num_sites];
  eta[1] = coClover[1] * phi[1 * num_sites];
  eta[2] = coClover[2] * phi[2 * num_sites];
  eta[3] = coClover[3] * phi[3 * num_sites];
  eta[4] = coClover[4] * phi[4 * num_sites];
  eta[5] = coClover[5] * phi[5 * num_sites];
  eta[6] = coClover[6] * phi[6 * num_sites];
  eta[7] = coClover[7] * phi[7 * num_sites];
  eta[8] = coClover[8] * phi[8 * num_sites];
  eta[9] = coClover[9] * phi[9 * num_sites];
  eta[10] = coClover[10] * phi[10 * num_sites];
  eta[11] = coClover[11] * phi[11 * num_sites];
  // spin 0 and 1, row major
  eta[0] += coClover[12] * phi[1 * num_sites];
  eta[0] += coClover[13] * phi[2 * num_sites];
  eta[0] += coClover[14] * phi[3 * num_sites];
  eta[0] += coClover[15] * phi[4 * num_sites];
  eta[0] += coClover[16] * phi[5 * num_sites];
  eta[1] += coClover[17] * phi[2 * num_sites];
  eta[1] += coClover[18] * phi[3 * num_sites];
  eta[1] += coClover[19] * phi[4 * num_sites];
  eta[1] += coClover[20] * phi[5 * num_sites];
  eta[2] += coClover[21] * phi[3 * num_sites];
  eta[2] += coClover[22] * phi[4 * num_sites];
  eta[2] += coClover[23] * phi[5 * num_sites];
  eta[3] += coClover[24] * phi[4 * num_sites];
  eta[3] += coClover[25] * phi[5 * num_sites];
  eta[4] += coClover[26] * phi[5 * num_sites];
  eta[1] += cu_conj_PRECISION(coClover[12]) * phi[0 * num_sites];
  eta[2] += cu_conj_PRECISION(coClover[13]) * phi[0 * num_sites];
  eta[3] += cu_conj_PRECISION(coClover[14]) * phi[0 * num_sites];
  eta[4] += cu_conj_PRECISION(coClover[15]) * phi[0 * num_sites];
  eta[5] += cu_conj_PRECISION(coClover[16]) * phi[0 * num_sites];
  eta[2] += cu_conj_PRECISION(coClover[17]) * phi[1 * num_sites];
  eta[3] += cu_conj_PRECISION(coClover[18]) * phi[1 * num_sites];
  eta[4] += cu_conj_PRECISION(coClover[19]) * phi[1 * num_sites];
  eta[5] += cu_conj_PRECISION(coClover[20]) * phi[1 * num_sites];
  eta[3] += cu_conj_PRECISION(coClover[21]) * phi[2 * num_sites];
  eta[4] += cu_conj_PRECISION(coClover[22]) * phi[2 * num_sites];
  eta[5] += cu_conj_PRECISION(coClover[23]) * phi[2 * num_sites];
  eta[4] += cu_conj_PRECISION(coClover[24]) * phi[3 * num_sites];
  eta[5] += cu_conj_PRECISION(coClover[25]) * phi[3 * num_sites];
  eta[5] += cu_conj_PRECISION(coClover[26]) * phi[4 * num_sites];
  // spin 2 and 3, row major
  eta[6] += coClover[27] * phi[7 * num_sites];
  eta[6] += coClover[28] * phi[8 * num_sites];
  eta[6] += coClover[29] * phi[9 * num_sites];
  eta[6] += coClover[30] * phi[10 * num_sites];
  eta[6] += coClover[31] * phi[11 * num_sites];
  eta[7] += coClover[32] * phi[8 * num_sites];
  eta[7] += coClover[33] * phi[9 * num_sites];
  eta[7] += coClover[34] * phi[10 * num_sites];
  eta[7] += coClover[35] * phi[11 * num_sites];
  eta[8] += coClover[36] * phi[9 * num_sites];
  eta[8] += coClover[37] * phi[10 * num_sites];
  eta[8] += coClover[38] * phi[11 * num_sites];
  eta[9] += coClover[39] * phi[10 * num_sites];
  eta[9] += coClover[40] * phi[11 * num_sites];
  eta[10] += coClover[41] * phi[11 * num_sites];
  eta[7] += cu_conj_PRECISION(coClover[27]) * phi[6 * num_sites];
  eta[8] += cu_conj_PRECISION(coClover[28]) * phi[6 * num_sites];
  eta[9] += cu_conj_PRECISION(coClover[29]) * phi[6 * num_sites];
  eta[10] += cu_conj_PRECISION(coClover[30]) * phi[6 * num_sites];
  eta[11] += cu_conj_PRECISION(coClover[31]) * phi[6 * num_sites];
  eta[8] += cu_conj_PRECISION(coClover[32]) * phi[7 * num_sites];
  eta[9] += cu_conj_PRECISION(coClover[33]) * phi[7 * num_sites];
  eta[10] += cu_conj_PRECISION(coClover[34]) * phi[7 * num_sites];
  eta[11] += cu_conj_PRECISION(coClover[35]) * phi[7 * num_sites];
  eta[9] += cu_conj_PRECISION(coClover[36]) * phi[8 * num_sites];
  eta[10] += cu_conj_PRECISION(coClover[37]) * phi[8 * num_sites];
  eta[11] += cu_conj_PRECISION(coClover[38]) * phi[8 * num_sites];
  eta[10] += cu_conj_PRECISION(coClover[39]) * phi[9 * num_sites];
  eta[11] += cu_conj_PRECISION(coClover[40]) * phi[9 * num_sites];
  eta[11] += cu_conj_PRECISION(coClover[41]) * phi[10 * num_sites];
}

__global__ void cuda_prp_T_componentwise_PRECISION(cu_cmplx_PRECISION* prpT,
                                                   cu_cmplx_PRECISION const* phi,
                                                   size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  phi += idx;
  prpT += 6 * idx;
  prpT[0] = phi[0 * num_sites] - GAMMA_T_SPIN0_VAL * phi[(3 * GAMMA_T_SPIN0_CO + 0) * num_sites];
  prpT[1] = phi[1 * num_sites] - GAMMA_T_SPIN0_VAL * phi[(3 * GAMMA_T_SPIN0_CO + 1) * num_sites];
  prpT[2] = phi[2 * num_sites] - GAMMA_T_SPIN0_VAL * phi[(3 * GAMMA_T_SPIN0_CO + 2) * num_sites];
  prpT[3] = phi[3 * num_sites] - GAMMA_T_SPIN1_VAL * phi[(3 * GAMMA_T_SPIN1_CO + 0) * num_sites];
  prpT[4] = phi[4 * num_sites] - GAMMA_T_SPIN1_VAL * phi[(3 * GAMMA_T_SPIN1_CO + 1) * num_sites];
  prpT[5] = phi[5 * num_sites] - GAMMA_T_SPIN1_VAL * phi[(3 * GAMMA_T_SPIN1_CO + 2) * num_sites];
}

__global__ void cuda_prn_T_componentwise_PRECISION(cu_cmplx_PRECISION* prnT,
                                                   cu_cmplx_PRECISION const* phi,
                                                   size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  phi += idx;
  prnT += 6 * idx;
  prnT[0] = phi[0 * num_sites] + GAMMA_T_SPIN0_VAL * phi[(3 * GAMMA_T_SPIN0_CO + 0) * num_sites];
  prnT[1] = phi[1 * num_sites] + GAMMA_T_SPIN0_VAL * phi[(3 * GAMMA_T_SPIN0_CO + 1) * num_sites];
  prnT[2] = phi[2 * num_sites] + GAMMA_T_SPIN0_VAL * phi[(3 * GAMMA_T_SPIN0_CO + 2) * num_sites];
  prnT[3] = phi[3 * num_sites] + GAMMA_T_SPIN1_VAL * phi[(3 * GAMMA_T_SPIN1_CO + 0) * num_sites];
  prnT[4] = phi[4 * num_sites] + GAMMA_T_SPIN1_VAL * phi[(3 * GAMMA_T_SPIN1_CO + 1) * num_sites];
  prnT[5] = phi[5 * num_sites] + GAMMA_T_SPIN1_VAL * phi[(3 * GAMMA_T_SPIN1_CO + 2) * num_sites];
}

__global__ void cuda_prp_Z_componentwise_PRECISION(cu_cmplx_PRECISION* prpZ,
                                                   cu_cmplx_PRECISION const* phi,
                                                   size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  phi += idx;
  prpZ += 6 * idx;
  prpZ[0] = phi[0 * num_sites] - GAMMA_Z_SPIN0_VAL * phi[(3 * GAMMA_Z_SPIN0_CO + 0) * num_sites];
  prpZ[1] = phi[1 * num_sites] - GAMMA_Z_SPIN0_VAL * phi[(3 * GAMMA_Z_SPIN0_CO + 1) * num_sites];
  prpZ[2] = phi[2 * num_sites] - GAMMA_Z_SPIN0_VAL * phi[(3 * GAMMA_Z_SPIN0_CO + 2) * num_sites];
  prpZ[3] = phi[3 * num_sites] - GAMMA_Z_SPIN1_VAL * phi[(3 * GAMMA_Z_SPIN1_CO + 0) * num_sites];
  prpZ[4] = phi[4 * num_sites] - GAMMA_Z_SPIN1_VAL * phi[(3 * GAMMA_Z_SPIN1_CO + 1) * num_sites];
  prpZ[5] = phi[5 * num_sites] - GAMMA_Z_SPIN1_VAL * phi[(3 * GAMMA_Z_SPIN1_CO + 2) * num_sites];
}

__global__ void cuda_prn_Z_componentwise_PRECISION(cu_cmplx_PRECISION* prnZ,
                                                   cu_cmplx_PRECISION const* phi,
                                                   size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  phi += idx;
  prnZ += 6 * idx;
  prnZ[0] = phi[0 * num_sites] + GAMMA_Z_SPIN0_VAL * phi[(3 * GAMMA_Z_SPIN0_CO + 0) *  num_sites];
  prnZ[1] = phi[1 * num_sites] + GAMMA_Z_SPIN0_VAL * phi[(3 * GAMMA_Z_SPIN0_CO + 1) *  num_sites];
  prnZ[2] = phi[2 * num_sites] + GAMMA_Z_SPIN0_VAL * phi[(3 * GAMMA_Z_SPIN0_CO + 2) *  num_sites];
  prnZ[3] = phi[3 * num_sites] + GAMMA_Z_SPIN1_VAL * phi[(3 * GAMMA_Z_SPIN1_CO + 0) *  num_sites];
  prnZ[4] = phi[4 * num_sites] + GAMMA_Z_SPIN1_VAL * phi[(3 * GAMMA_Z_SPIN1_CO + 1) *  num_sites];
  prnZ[5] = phi[5 * num_sites] + GAMMA_Z_SPIN1_VAL * phi[(3 * GAMMA_Z_SPIN1_CO + 2) *  num_sites];
}

__global__ void cuda_prp_Y_componentwise_PRECISION(cu_cmplx_PRECISION* prpY,
                                                   cu_cmplx_PRECISION const* phi,
                                                   size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  phi += idx;
  prpY += 6 * idx;
  prpY[0] = phi[0 * num_sites] - GAMMA_Y_SPIN0_VAL * phi[(3 * GAMMA_Y_SPIN0_CO + 0) * num_sites];
  prpY[1] = phi[1 * num_sites] - GAMMA_Y_SPIN0_VAL * phi[(3 * GAMMA_Y_SPIN0_CO + 1) * num_sites];
  prpY[2] = phi[2 * num_sites] - GAMMA_Y_SPIN0_VAL * phi[(3 * GAMMA_Y_SPIN0_CO + 2) * num_sites];
  prpY[3] = phi[3 * num_sites] - GAMMA_Y_SPIN1_VAL * phi[(3 * GAMMA_Y_SPIN1_CO + 0) * num_sites];
  prpY[4] = phi[4 * num_sites] - GAMMA_Y_SPIN1_VAL * phi[(3 * GAMMA_Y_SPIN1_CO + 1) * num_sites];
  prpY[5] = phi[5 * num_sites] - GAMMA_Y_SPIN1_VAL * phi[(3 * GAMMA_Y_SPIN1_CO + 2) * num_sites];
}

__global__ void cuda_prn_Y_componentwise_PRECISION(cu_cmplx_PRECISION* prnY,
                                                   cu_cmplx_PRECISION const* phi,
                                                   size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  phi += idx;
  prnY += 6 * idx;
  prnY[0] = phi[0 * num_sites] + GAMMA_Y_SPIN0_VAL * phi[(3 * GAMMA_Y_SPIN0_CO + 0) * num_sites];
  prnY[1] = phi[1 * num_sites] + GAMMA_Y_SPIN0_VAL * phi[(3 * GAMMA_Y_SPIN0_CO + 1) * num_sites];
  prnY[2] = phi[2 * num_sites] + GAMMA_Y_SPIN0_VAL * phi[(3 * GAMMA_Y_SPIN0_CO + 2) * num_sites];
  prnY[3] = phi[3 * num_sites] + GAMMA_Y_SPIN1_VAL * phi[(3 * GAMMA_Y_SPIN1_CO + 0) * num_sites];
  prnY[4] = phi[4 * num_sites] + GAMMA_Y_SPIN1_VAL * phi[(3 * GAMMA_Y_SPIN1_CO + 1) * num_sites];
  prnY[5] = phi[5 * num_sites] + GAMMA_Y_SPIN1_VAL * phi[(3 * GAMMA_Y_SPIN1_CO + 2) * num_sites];
}

__global__ void cuda_prp_X_componentwise_PRECISION(cu_cmplx_PRECISION* prpX,
                                                   cu_cmplx_PRECISION const* phi,
                                                   size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  phi += idx;
  prpX += 6 * idx;
  prpX[0] = phi[0 * num_sites] - GAMMA_X_SPIN0_VAL * phi[(3 * GAMMA_X_SPIN0_CO + 0) * num_sites];
  prpX[1] = phi[1 * num_sites] - GAMMA_X_SPIN0_VAL * phi[(3 * GAMMA_X_SPIN0_CO + 1) * num_sites];
  prpX[2] = phi[2 * num_sites] - GAMMA_X_SPIN0_VAL * phi[(3 * GAMMA_X_SPIN0_CO + 2) * num_sites];
  prpX[3] = phi[3 * num_sites] - GAMMA_X_SPIN1_VAL * phi[(3 * GAMMA_X_SPIN1_CO + 0) * num_sites];
  prpX[4] = phi[4 * num_sites] - GAMMA_X_SPIN1_VAL * phi[(3 * GAMMA_X_SPIN1_CO + 1) * num_sites];
  prpX[5] = phi[5 * num_sites] - GAMMA_X_SPIN1_VAL * phi[(3 * GAMMA_X_SPIN1_CO + 2) * num_sites];
}

__global__ void cuda_prn_X_componentwise_PRECISION(cu_cmplx_PRECISION* prnX,
                                                   cu_cmplx_PRECISION const* phi,
                                                   size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  phi += idx;
  prnX += 6 * idx;
  prnX[0] = phi[0 * num_sites] + GAMMA_X_SPIN0_VAL * phi[(3 * GAMMA_X_SPIN0_CO + 0) * num_sites];
  prnX[1] = phi[1 * num_sites] + GAMMA_X_SPIN0_VAL * phi[(3 * GAMMA_X_SPIN0_CO + 1) * num_sites];
  prnX[2] = phi[2 * num_sites] + GAMMA_X_SPIN0_VAL * phi[(3 * GAMMA_X_SPIN0_CO + 2) * num_sites];
  prnX[3] = phi[3 * num_sites] + GAMMA_X_SPIN1_VAL * phi[(3 * GAMMA_X_SPIN1_CO + 0) * num_sites];
  prnX[4] = phi[4 * num_sites] + GAMMA_X_SPIN1_VAL * phi[(3 * GAMMA_X_SPIN1_CO + 1) * num_sites];
  prnX[5] = phi[5 * num_sites] + GAMMA_X_SPIN1_VAL * phi[(3 * GAMMA_X_SPIN1_CO + 2) * num_sites];
}
