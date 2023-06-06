#include "cuda_complex.h"
#include "cuda_complex_cxx.h"
#include "cuda_complex_operators.h"
#include "cuda_complex_operators_PRECISION.h"
#include "cuda_dirac_kernels_componentwise_PRECISION.h"
#include "cuda_mvm_PRECISION.h"
#include "global_enums.h"

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
  clover += 42 * idx;

  // diagonal
  eta[0] = clover[0] * phi[0 * num_sites];
  eta[1] = clover[1] * phi[1 * num_sites];
  eta[2] = clover[2] * phi[2 * num_sites];
  eta[3] = clover[3] * phi[3 * num_sites];
  eta[4] = clover[4] * phi[4 * num_sites];
  eta[5] = clover[5] * phi[5 * num_sites];
  eta[6] = clover[6] * phi[6 * num_sites];
  eta[7] = clover[7] * phi[7 * num_sites];
  eta[8] = clover[8] * phi[8 * num_sites];
  eta[9] = clover[9] * phi[9 * num_sites];
  eta[10] = clover[10] * phi[10 * num_sites];
  eta[11] = clover[11] * phi[11 * num_sites];
  // spin 0 and 1, row major
  eta[0] += clover[12] * phi[1 * num_sites];
  eta[0] += clover[13] * phi[2 * num_sites];
  eta[0] += clover[14] * phi[3 * num_sites];
  eta[0] += clover[15] * phi[4 * num_sites];
  eta[0] += clover[16] * phi[5 * num_sites];
  eta[1] += clover[17] * phi[2 * num_sites];
  eta[1] += clover[18] * phi[3 * num_sites];
  eta[1] += clover[19] * phi[4 * num_sites];
  eta[1] += clover[20] * phi[5 * num_sites];
  eta[2] += clover[21] * phi[3 * num_sites];
  eta[2] += clover[22] * phi[4 * num_sites];
  eta[2] += clover[23] * phi[5 * num_sites];
  eta[3] += clover[24] * phi[4 * num_sites];
  eta[3] += clover[25] * phi[5 * num_sites];
  eta[4] += clover[26] * phi[5 * num_sites];
  eta[1] += cu_conj_PRECISION(clover[12]) * phi[0 * num_sites];
  eta[2] += cu_conj_PRECISION(clover[13]) * phi[0 * num_sites];
  eta[3] += cu_conj_PRECISION(clover[14]) * phi[0 * num_sites];
  eta[4] += cu_conj_PRECISION(clover[15]) * phi[0 * num_sites];
  eta[5] += cu_conj_PRECISION(clover[16]) * phi[0 * num_sites];
  eta[2] += cu_conj_PRECISION(clover[17]) * phi[1 * num_sites];
  eta[3] += cu_conj_PRECISION(clover[18]) * phi[1 * num_sites];
  eta[4] += cu_conj_PRECISION(clover[19]) * phi[1 * num_sites];
  eta[5] += cu_conj_PRECISION(clover[20]) * phi[1 * num_sites];
  eta[3] += cu_conj_PRECISION(clover[21]) * phi[2 * num_sites];
  eta[4] += cu_conj_PRECISION(clover[22]) * phi[2 * num_sites];
  eta[5] += cu_conj_PRECISION(clover[23]) * phi[2 * num_sites];
  eta[4] += cu_conj_PRECISION(clover[24]) * phi[3 * num_sites];
  eta[5] += cu_conj_PRECISION(clover[25]) * phi[3 * num_sites];
  eta[5] += cu_conj_PRECISION(clover[26]) * phi[4 * num_sites];
  // spin 2 and 3, row major
  eta[6] += clover[27] * phi[7 * num_sites];
  eta[6] += clover[28] * phi[8 * num_sites];
  eta[6] += clover[29] * phi[9 * num_sites];
  eta[6] += clover[30] * phi[10 * num_sites];
  eta[6] += clover[31] * phi[11 * num_sites];
  eta[7] += clover[32] * phi[8 * num_sites];
  eta[7] += clover[33] * phi[9 * num_sites];
  eta[7] += clover[34] * phi[10 * num_sites];
  eta[7] += clover[35] * phi[11 * num_sites];
  eta[8] += clover[36] * phi[9 * num_sites];
  eta[8] += clover[37] * phi[10 * num_sites];
  eta[8] += clover[38] * phi[11 * num_sites];
  eta[9] += clover[39] * phi[10 * num_sites];
  eta[9] += clover[40] * phi[11 * num_sites];
  eta[10] += clover[41] * phi[11 * num_sites];
  eta[7] += cu_conj_PRECISION(clover[27]) * phi[6 * num_sites];
  eta[8] += cu_conj_PRECISION(clover[28]) * phi[6 * num_sites];
  eta[9] += cu_conj_PRECISION(clover[29]) * phi[6 * num_sites];
  eta[10] += cu_conj_PRECISION(clover[30]) * phi[6 * num_sites];
  eta[11] += cu_conj_PRECISION(clover[31]) * phi[6 * num_sites];
  eta[8] += cu_conj_PRECISION(clover[32]) * phi[7 * num_sites];
  eta[9] += cu_conj_PRECISION(clover[33]) * phi[7 * num_sites];
  eta[10] += cu_conj_PRECISION(clover[34]) * phi[7 * num_sites];
  eta[11] += cu_conj_PRECISION(clover[35]) * phi[7 * num_sites];
  eta[9] += cu_conj_PRECISION(clover[36]) * phi[8 * num_sites];
  eta[10] += cu_conj_PRECISION(clover[37]) * phi[8 * num_sites];
  eta[11] += cu_conj_PRECISION(clover[38]) * phi[8 * num_sites];
  eta[10] += cu_conj_PRECISION(clover[39]) * phi[9 * num_sites];
  eta[11] += cu_conj_PRECISION(clover[40]) * phi[9 * num_sites];
  eta[11] += cu_conj_PRECISION(clover[41]) * phi[10 * num_sites];
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
