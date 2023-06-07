#include "cuda_complex.h"
#include "cuda_complex_cxx.h"
#include "cuda_complex_operators.h"
#include "cuda_complex_operators_PRECISION.h"
#include "cuda_componentwise.h"
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
  auto caEta = ComponentAccess(eta + idx, num_sites);
  auto caPhi = ComponentAccess(phi + idx, num_sites);
  // clover += idx;
  auto caClover = ComponentAccess(clover + idx, num_sites);

  // diagonal
  caEta[0] = caClover[0] * caPhi[0];
  caEta[1] = caClover[1] * caPhi[1];
  caEta[2] = caClover[2] * caPhi[2];
  caEta[3] = caClover[3] * caPhi[3];
  caEta[4] = caClover[4] * caPhi[4];
  caEta[5] = caClover[5] * caPhi[5];
  caEta[6] = caClover[6] * caPhi[6];
  caEta[7] = caClover[7] * caPhi[7];
  caEta[8] = caClover[8] * caPhi[8];
  caEta[9] = caClover[9] * caPhi[9];
  caEta[10] = caClover[10] * caPhi[10];
  caEta[11] = caClover[11] * caPhi[11];
  // spin 0 and 1, row major
  caEta[0] += caClover[12] * caPhi[1];
  caEta[0] += caClover[13] * caPhi[2];
  caEta[0] += caClover[14] * caPhi[3];
  caEta[0] += caClover[15] * caPhi[4];
  caEta[0] += caClover[16] * caPhi[5];
  caEta[1] += caClover[17] * caPhi[2];
  caEta[1] += caClover[18] * caPhi[3];
  caEta[1] += caClover[19] * caPhi[4];
  caEta[1] += caClover[20] * caPhi[5];
  caEta[2] += caClover[21] * caPhi[3];
  caEta[2] += caClover[22] * caPhi[4];
  caEta[2] += caClover[23] * caPhi[5];
  caEta[3] += caClover[24] * caPhi[4];
  caEta[3] += caClover[25] * caPhi[5];
  caEta[4] += caClover[26] * caPhi[5];
  caEta[1] += cu_conj_PRECISION(caClover[12]) * caPhi[0];
  caEta[2] += cu_conj_PRECISION(caClover[13]) * caPhi[0];
  caEta[3] += cu_conj_PRECISION(caClover[14]) * caPhi[0];
  caEta[4] += cu_conj_PRECISION(caClover[15]) * caPhi[0];
  caEta[5] += cu_conj_PRECISION(caClover[16]) * caPhi[0];
  caEta[2] += cu_conj_PRECISION(caClover[17]) * caPhi[1];
  caEta[3] += cu_conj_PRECISION(caClover[18]) * caPhi[1];
  caEta[4] += cu_conj_PRECISION(caClover[19]) * caPhi[1];
  caEta[5] += cu_conj_PRECISION(caClover[20]) * caPhi[1];
  caEta[3] += cu_conj_PRECISION(caClover[21]) * caPhi[2];
  caEta[4] += cu_conj_PRECISION(caClover[22]) * caPhi[2];
  caEta[5] += cu_conj_PRECISION(caClover[23]) * caPhi[2];
  caEta[4] += cu_conj_PRECISION(caClover[24]) * caPhi[3];
  caEta[5] += cu_conj_PRECISION(caClover[25]) * caPhi[3];
  caEta[5] += cu_conj_PRECISION(caClover[26]) * caPhi[4];
  // spin 2 and 3, row major
  caEta[6] += caClover[27] * caPhi[7];
  caEta[6] += caClover[28] * caPhi[8];
  caEta[6] += caClover[29] * caPhi[9];
  caEta[6] += caClover[30] * caPhi[10];
  caEta[6] += caClover[31] * caPhi[11];
  caEta[7] += caClover[32] * caPhi[8];
  caEta[7] += caClover[33] * caPhi[9];
  caEta[7] += caClover[34] * caPhi[10];
  caEta[7] += caClover[35] * caPhi[11];
  caEta[8] += caClover[36] * caPhi[9];
  caEta[8] += caClover[37] * caPhi[10];
  caEta[8] += caClover[38] * caPhi[11];
  caEta[9] += caClover[39] * caPhi[10];
  caEta[9] += caClover[40] * caPhi[11];
  caEta[10] += caClover[41] * caPhi[11];
  caEta[7] += cu_conj_PRECISION(caClover[27]) * caPhi[6];
  caEta[8] += cu_conj_PRECISION(caClover[28]) * caPhi[6];
  caEta[9] += cu_conj_PRECISION(caClover[29]) * caPhi[6];
  caEta[10] += cu_conj_PRECISION(caClover[30]) * caPhi[6];
  caEta[11] += cu_conj_PRECISION(caClover[31]) * caPhi[6];
  caEta[8] += cu_conj_PRECISION(caClover[32]) * caPhi[7];
  caEta[9] += cu_conj_PRECISION(caClover[33]) * caPhi[7];
  caEta[10] += cu_conj_PRECISION(caClover[34]) * caPhi[7];
  caEta[11] += cu_conj_PRECISION(caClover[35]) * caPhi[7];
  caEta[9] += cu_conj_PRECISION(caClover[36]) * caPhi[8];
  caEta[10] += cu_conj_PRECISION(caClover[37]) * caPhi[8];
  caEta[11] += cu_conj_PRECISION(caClover[38]) * caPhi[8];
  caEta[10] += cu_conj_PRECISION(caClover[39]) * caPhi[9];
  caEta[11] += cu_conj_PRECISION(caClover[40]) * caPhi[9];
  caEta[11] += cu_conj_PRECISION(caClover[41]) * caPhi[10];
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
  auto caPrnT = ComponentAccess(prnT + idx, num_sites);
  caPrnT[0] = caPhi[0] + GAMMA_T_SPIN0_VAL * caPhi[3 * GAMMA_T_SPIN0_CO + 0];
  caPrnT[1] = caPhi[1] + GAMMA_T_SPIN0_VAL * caPhi[3 * GAMMA_T_SPIN0_CO + 1];
  caPrnT[2] = caPhi[2] + GAMMA_T_SPIN0_VAL * caPhi[3 * GAMMA_T_SPIN0_CO + 2];
  caPrnT[3] = caPhi[3] + GAMMA_T_SPIN1_VAL * caPhi[3 * GAMMA_T_SPIN1_CO + 0];
  caPrnT[4] = caPhi[4] + GAMMA_T_SPIN1_VAL * caPhi[3 * GAMMA_T_SPIN1_CO + 1];
  caPrnT[5] = caPhi[5] + GAMMA_T_SPIN1_VAL * caPhi[3 * GAMMA_T_SPIN1_CO + 2];
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
  auto caPrnZ = ComponentAccess(prnZ + idx, num_sites);
  caPrnZ[0] = caPhi[0] + GAMMA_Z_SPIN0_VAL * caPhi[3 * GAMMA_Z_SPIN0_CO + 0];
  caPrnZ[1] = caPhi[1] + GAMMA_Z_SPIN0_VAL * caPhi[3 * GAMMA_Z_SPIN0_CO + 1];
  caPrnZ[2] = caPhi[2] + GAMMA_Z_SPIN0_VAL * caPhi[3 * GAMMA_Z_SPIN0_CO + 2];
  caPrnZ[3] = caPhi[3] + GAMMA_Z_SPIN1_VAL * caPhi[3 * GAMMA_Z_SPIN1_CO + 0];
  caPrnZ[4] = caPhi[4] + GAMMA_Z_SPIN1_VAL * caPhi[3 * GAMMA_Z_SPIN1_CO + 1];
  caPrnZ[5] = caPhi[5] + GAMMA_Z_SPIN1_VAL * caPhi[3 * GAMMA_Z_SPIN1_CO + 2];
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
  auto caPrnY = ComponentAccess(prnY + idx, num_sites);
  caPrnY[0] = caPhi[0] + GAMMA_Y_SPIN0_VAL * caPhi[3 * GAMMA_Y_SPIN0_CO + 0];
  caPrnY[1] = caPhi[1] + GAMMA_Y_SPIN0_VAL * caPhi[3 * GAMMA_Y_SPIN0_CO + 1];
  caPrnY[2] = caPhi[2] + GAMMA_Y_SPIN0_VAL * caPhi[3 * GAMMA_Y_SPIN0_CO + 2];
  caPrnY[3] = caPhi[3] + GAMMA_Y_SPIN1_VAL * caPhi[3 * GAMMA_Y_SPIN1_CO + 0];
  caPrnY[4] = caPhi[4] + GAMMA_Y_SPIN1_VAL * caPhi[3 * GAMMA_Y_SPIN1_CO + 1];
  caPrnY[5] = caPhi[5] + GAMMA_Y_SPIN1_VAL * caPhi[3 * GAMMA_Y_SPIN1_CO + 2];
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
  auto caPrnX = ComponentAccess(prnX + idx, num_sites);
  caPrnX[0] = caPhi[0] + GAMMA_X_SPIN0_VAL * caPhi[3 * GAMMA_X_SPIN0_CO + 0];
  caPrnX[1] = caPhi[1] + GAMMA_X_SPIN0_VAL * caPhi[3 * GAMMA_X_SPIN0_CO + 1];
  caPrnX[2] = caPhi[2] + GAMMA_X_SPIN0_VAL * caPhi[3 * GAMMA_X_SPIN0_CO + 2];
  caPrnX[3] = caPhi[3] + GAMMA_X_SPIN1_VAL * caPhi[3 * GAMMA_X_SPIN1_CO + 0];
  caPrnX[4] = caPhi[4] + GAMMA_X_SPIN1_VAL * caPhi[3 * GAMMA_X_SPIN1_CO + 1];
  caPrnX[5] = caPhi[5] + GAMMA_X_SPIN1_VAL * caPhi[3 * GAMMA_X_SPIN1_CO + 2];
}

__global__ void cuda_prn_mvmh_componentwise_PRECISION(cu_cmplx_PRECISION* prp_buf,
                                                      cu_cmplx_PRECISION const* D,
                                                      cu_cmplx_PRECISION const* pbuf,
                                                      int const* neighbors, LatticeAxis dim,
                                                      size_t num_sites) {
  unsigned int neighbor_offset;
  switch (dim) {
    case LatticeAxis::T:
      neighbor_offset = 0;
      break;
    case LatticeAxis::Z:
      neighbor_offset = 1;
      break;
    case LatticeAxis::Y:
      neighbor_offset = 2;
      break;
    case LatticeAxis::X:
      neighbor_offset = 3;
      break;
  }
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  const size_t lattice_idx = idx / 2;

  if (lattice_idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  D += 9 * (4 * lattice_idx + neighbor_offset);
  neighbors += 4 * lattice_idx + neighbor_offset;
  // We operate in steps of 3 here as the application of D happens as 3x3 matrix vector
  // multiplications. There will be two mvms per lattice site.
  if (idx % 2 == 1) {
    // Even indices access elements 0..2
    // Uneven indices access elements 3..5
    pbuf += 3 * num_sites;
  }
  auto caPbuf = ComponentAccess(pbuf + lattice_idx, num_sites);
  const size_t j = 6 * (*neighbors);
  prp_buf += j + (idx % 2 == 0 ? 0 : 3);
  cuda_mvmh_componentwise_PRECISION(prp_buf, D, caPbuf);
}

__global__ void cuda_pbp_su3_mvm_componentwise_PRECISION(cu_cmplx_PRECISION* pbuf,
                                                         cu_cmplx_PRECISION const* D,
                                                         cu_cmplx_PRECISION const* prn_buf,
                                                         int const* neighbors, LatticeAxis dim,
                                                         size_t num_sites) {
  unsigned int neighbor_offset;
  switch (dim) {
    case LatticeAxis::T:
      neighbor_offset = 0;
      break;
    case LatticeAxis::Z:
      neighbor_offset = 1;
      break;
    case LatticeAxis::Y:
      neighbor_offset = 2;
      break;
    case LatticeAxis::X:
      neighbor_offset = 3;
      break;
  }
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  const size_t lattice_idx = idx / 2;

  if (lattice_idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  D += 9 * (4 * lattice_idx + neighbor_offset);
  neighbors += 4 * lattice_idx + neighbor_offset;
  // We operate in steps of 3 here as the application of D happens as 3x3 matrix vector
  // multiplications. There will be two mvms per lattice site.
  pbuf += 3 * idx;
  const size_t j = 6 * (*neighbors);
  prn_buf += j + (idx % 2 == 0 ? 0 : 3);
  cuda_mvm_PRECISION(pbuf, D, prn_buf);
}

__global__ void cuda_pbp_su3_T_componentwise_PRECISION(cu_cmplx_PRECISION* eta,
                                                       cu_cmplx_PRECISION const* pbuf,
                                                       size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caEta = ComponentAccess(eta + idx, num_sites);
  pbuf += 6 * idx;
  caEta[0] -= pbuf[0];
  caEta[1] -= pbuf[1];
  caEta[2] -= pbuf[2];
  caEta[3] -= pbuf[3];
  caEta[4] -= pbuf[4];
  caEta[5] -= pbuf[5];
  caEta[6] += GAMMA_T_SPIN2_VAL * pbuf[3 * GAMMA_T_SPIN2_CO];
  caEta[7] += GAMMA_T_SPIN2_VAL * pbuf[3 * GAMMA_T_SPIN2_CO + 1];
  caEta[8] += GAMMA_T_SPIN2_VAL * pbuf[3 * GAMMA_T_SPIN2_CO + 2];
  caEta[9] += GAMMA_T_SPIN3_VAL * pbuf[3 * GAMMA_T_SPIN3_CO];
  caEta[10] += GAMMA_T_SPIN3_VAL * pbuf[3 * GAMMA_T_SPIN3_CO + 1];
  caEta[11] += GAMMA_T_SPIN3_VAL * pbuf[3 * GAMMA_T_SPIN3_CO + 2];
}

__global__ void cuda_pbp_su3_Z_componentwise_PRECISION(cu_cmplx_PRECISION* eta,
                                                       cu_cmplx_PRECISION const* pbuf,
                                                       size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caEta = ComponentAccess(eta + idx, num_sites);
  pbuf += 6 * idx;
  caEta[0] -= pbuf[0];
  caEta[1] -= pbuf[1];
  caEta[2] -= pbuf[2];
  caEta[3] -= pbuf[3];
  caEta[4] -= pbuf[4];
  caEta[5] -= pbuf[5];
  caEta[6] += GAMMA_Z_SPIN2_VAL * pbuf[3 * GAMMA_Z_SPIN2_CO];
  caEta[7] += GAMMA_Z_SPIN2_VAL * pbuf[3 * GAMMA_Z_SPIN2_CO + 1];
  caEta[8] += GAMMA_Z_SPIN2_VAL * pbuf[3 * GAMMA_Z_SPIN2_CO + 2];
  caEta[9] += GAMMA_Z_SPIN3_VAL * pbuf[3 * GAMMA_Z_SPIN3_CO];
  caEta[10] += GAMMA_Z_SPIN3_VAL * pbuf[3 * GAMMA_Z_SPIN3_CO + 1];
  caEta[11] += GAMMA_Z_SPIN3_VAL * pbuf[3 * GAMMA_Z_SPIN3_CO + 2];
}

__global__ void cuda_pbp_su3_Y_componentwise_PRECISION(cu_cmplx_PRECISION* eta,
                                                       cu_cmplx_PRECISION const* pbuf,
                                                       size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caEta = ComponentAccess(eta + idx, num_sites);
  pbuf += 6 * idx;
  caEta[0] -= pbuf[0];
  caEta[1] -= pbuf[1];
  caEta[2] -= pbuf[2];
  caEta[3] -= pbuf[3];
  caEta[4] -= pbuf[4];
  caEta[5] -= pbuf[5];
  caEta[6] += GAMMA_Y_SPIN2_VAL * pbuf[3 * GAMMA_Y_SPIN2_CO];
  caEta[7] += GAMMA_Y_SPIN2_VAL * pbuf[3 * GAMMA_Y_SPIN2_CO + 1];
  caEta[8] += GAMMA_Y_SPIN2_VAL * pbuf[3 * GAMMA_Y_SPIN2_CO + 2];
  caEta[9] += GAMMA_Y_SPIN3_VAL * pbuf[3 * GAMMA_Y_SPIN3_CO];
  caEta[10] += GAMMA_Y_SPIN3_VAL * pbuf[3 * GAMMA_Y_SPIN3_CO + 1];
  caEta[11] += GAMMA_Y_SPIN3_VAL * pbuf[3 * GAMMA_Y_SPIN3_CO + 2];
}

__global__ void cuda_pbp_su3_X_componentwise_PRECISION(cu_cmplx_PRECISION* eta,
                                                       cu_cmplx_PRECISION const* pbuf,
                                                       size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caEta = ComponentAccess(eta + idx, num_sites);
  pbuf += 6 * idx;
  caEta[0] -= pbuf[0];
  caEta[1] -= pbuf[1];
  caEta[2] -= pbuf[2];
  caEta[3] -= pbuf[3];
  caEta[4] -= pbuf[4];
  caEta[5] -= pbuf[5];
  caEta[6] += GAMMA_X_SPIN2_VAL * pbuf[3 * GAMMA_X_SPIN2_CO];
  caEta[7] += GAMMA_X_SPIN2_VAL * pbuf[3 * GAMMA_X_SPIN2_CO + 1];
  caEta[8] += GAMMA_X_SPIN2_VAL * pbuf[3 * GAMMA_X_SPIN2_CO + 2];
  caEta[9] += GAMMA_X_SPIN3_VAL * pbuf[3 * GAMMA_X_SPIN3_CO];
  caEta[10] += GAMMA_X_SPIN3_VAL * pbuf[3 * GAMMA_X_SPIN3_CO + 1];
  caEta[11] += GAMMA_X_SPIN3_VAL * pbuf[3 * GAMMA_X_SPIN3_CO + 2];
}

__global__ void cuda_pbn_su3_T_componentwise_PRECISION(cu_cmplx_PRECISION* eta,
                                                       cu_cmplx_PRECISION const* prpT,
                                                       size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caEta = ComponentAccess(eta + idx, num_sites);
  prpT += 6 * idx;
  caEta[0] -= prpT[0];
  caEta[1] -= prpT[1];
  caEta[2] -= prpT[2];
  caEta[3] -= prpT[3];
  caEta[4] -= prpT[4];
  caEta[5] -= prpT[5];
  caEta[6] -= GAMMA_T_SPIN2_VAL * prpT[3 * GAMMA_T_SPIN2_CO];
  caEta[7] -= GAMMA_T_SPIN2_VAL * prpT[3 * GAMMA_T_SPIN2_CO + 1];
  caEta[8] -= GAMMA_T_SPIN2_VAL * prpT[3 * GAMMA_T_SPIN2_CO + 2];
  caEta[9] -= GAMMA_T_SPIN3_VAL * prpT[3 * GAMMA_T_SPIN3_CO];
  caEta[10] -= GAMMA_T_SPIN3_VAL * prpT[3 * GAMMA_T_SPIN3_CO + 1];
  caEta[11] -= GAMMA_T_SPIN3_VAL * prpT[3 * GAMMA_T_SPIN3_CO + 2];
}

__global__ void cuda_pbn_su3_Z_componentwise_PRECISION(cu_cmplx_PRECISION* eta,
                                                       cu_cmplx_PRECISION const* prpZ,
                                                       size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caEta = ComponentAccess(eta + idx, num_sites);
  prpZ += 6 * idx;
  caEta[0] -= prpZ[0];
  caEta[1] -= prpZ[1];
  caEta[2] -= prpZ[2];
  caEta[3] -= prpZ[3];
  caEta[4] -= prpZ[4];
  caEta[5] -= prpZ[5];
  caEta[6] -= GAMMA_Z_SPIN2_VAL * prpZ[3 * GAMMA_Z_SPIN2_CO];
  caEta[7] -= GAMMA_Z_SPIN2_VAL * prpZ[3 * GAMMA_Z_SPIN2_CO + 1];
  caEta[8] -= GAMMA_Z_SPIN2_VAL * prpZ[3 * GAMMA_Z_SPIN2_CO + 2];
  caEta[9] -= GAMMA_Z_SPIN3_VAL * prpZ[3 * GAMMA_Z_SPIN3_CO];
  caEta[10] -= GAMMA_Z_SPIN3_VAL * prpZ[3 * GAMMA_Z_SPIN3_CO + 1];
  caEta[11] -= GAMMA_Z_SPIN3_VAL * prpZ[3 * GAMMA_Z_SPIN3_CO + 2];
}

__global__ void cuda_pbn_su3_Y_componentwise_PRECISION(cu_cmplx_PRECISION* eta,
                                                       cu_cmplx_PRECISION const* prpY,
                                                       size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caEta = ComponentAccess(eta + idx, num_sites);
  prpY += 6 * idx;
  caEta[0] -= prpY[0];
  caEta[1] -= prpY[1];
  caEta[2] -= prpY[2];
  caEta[3] -= prpY[3];
  caEta[4] -= prpY[4];
  caEta[5] -= prpY[5];
  caEta[6] -= GAMMA_Y_SPIN2_VAL * prpY[3 * GAMMA_Y_SPIN2_CO];
  caEta[7] -= GAMMA_Y_SPIN2_VAL * prpY[3 * GAMMA_Y_SPIN2_CO + 1];
  caEta[8] -= GAMMA_Y_SPIN2_VAL * prpY[3 * GAMMA_Y_SPIN2_CO + 2];
  caEta[9] -= GAMMA_Y_SPIN3_VAL * prpY[3 * GAMMA_Y_SPIN3_CO];
  caEta[10] -= GAMMA_Y_SPIN3_VAL * prpY[3 * GAMMA_Y_SPIN3_CO + 1];
  caEta[11] -= GAMMA_Y_SPIN3_VAL * prpY[3 * GAMMA_Y_SPIN3_CO + 2];
}

__global__ void cuda_pbn_su3_X_componentwise_PRECISION(cu_cmplx_PRECISION* eta,
                                                       cu_cmplx_PRECISION const* prpX,
                                                       size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caEta = ComponentAccess(eta + idx, num_sites);
  prpX += 6 * idx;
  caEta[0] -= prpX[0];
  caEta[1] -= prpX[1];
  caEta[2] -= prpX[2];
  caEta[3] -= prpX[3];
  caEta[4] -= prpX[4];
  caEta[5] -= prpX[5];
  caEta[6] -= GAMMA_X_SPIN2_VAL * prpX[3 * GAMMA_X_SPIN2_CO];
  caEta[7] -= GAMMA_X_SPIN2_VAL * prpX[3 * GAMMA_X_SPIN2_CO + 1];
  caEta[8] -= GAMMA_X_SPIN2_VAL * prpX[3 * GAMMA_X_SPIN2_CO + 2];
  caEta[9] -= GAMMA_X_SPIN3_VAL * prpX[3 * GAMMA_X_SPIN3_CO];
  caEta[10] -= GAMMA_X_SPIN3_VAL * prpX[3 * GAMMA_X_SPIN3_CO + 1];
  caEta[11] -= GAMMA_X_SPIN3_VAL * prpX[3 * GAMMA_X_SPIN3_CO + 2];
}
