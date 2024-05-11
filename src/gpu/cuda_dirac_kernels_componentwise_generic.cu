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

__global__ void cuda_site_diag_ee_componentwise_PRECISION(cuda_vector_PRECISION eta,
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

  int i, j, k=0, n;
  cu_cmplx_PRECISION z[6];

  for ( n=0; n<2; n++ ) {
    // z = L^H x
    for ( j=0; j<6; j++ ) { // columns
      for ( i=0; i<j; i++ ) { // rows
        //z[i] += conj_PRECISION(*L)*x[j]; L++;
        z[i] += cu_conj_PRECISION(caClover[k])*caPhi[n*6+j]; k++;
      }

      //z[j] = conj_PRECISION(*L)*x[j]; L++;
      z[j] = cu_conj_PRECISION(caClover[k])*caPhi[n*6+j]; k++;
    }
    //L-=21;
    k -= 21;
    // y = L*z;
    for ( i=0; i<6; i++ ) { // rows
      //y[i] = *L * z[0]; L++;
      caEta[n*6+i] = caClover[k]*z[0]; k++;
      for ( j=1; j<=i; j++ ) { // columns
        //y[i] += *L * z[j]; L++;
        caEta[n*6+i] += caClover[k]*z[j]; k++;
      }
    }
    //x+=6;
    //y+=6;
  }
}

__global__ void cuda_site_diag_oo_inv_componentwise_PRECISION(cuda_vector_PRECISION eta,
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

  int i, j, k=0, n;

  for ( n=0; n<2; n++ ) {
    // forward substitution with L
    for ( i=0; i<6; i++ ) {
      //x[i] = b[i];
      caEta[n*6+i] = caPhi[n*6+i];
      for ( j=0; j<i; j++ ) {
        //x[i] = x[i] - *L * x[j]; L++;
        caEta[n*6+i] = caEta[n*6+i] - caClover[k]*caEta[n*6+j]; k++;
      }
      //x[i] = x[i] / *L; L++;
      caEta[n*6+i] = cu_cdiv_PRECISION(caEta[n*6+i],caClover[k]); k++;
    }
    //L -= 21;
    k -= 21;
    // backward substitution with L^H
    for ( i=5; i>=0; i-- ) {
      for ( j=i+1; j<6; j++ ) {
        //x[i] = x[i] - conj_PRECISION(L[(j*(j+1))/2 + i]) * x[j];
        caEta[n*6+i] = caEta[n*6+i] - cu_conj_PRECISION(caClover[k+(j*(j+1))/2+i])*caEta[n*6+j];
      }
      //x[i] = x[i] / conj_PRECISION(L[(i*(i+1))/2 + i]);
      caEta[n*6+i] = cu_cdiv_PRECISION(caEta[n*6+i],cu_conj_PRECISION(caClover[k+(i*(i+1))/2+i]));
    }
    //x+=6;
    //b+=6;
    //L+=21;
    k += 21;
  }
}

__global__ void cuda_prp_T_componentwise_PRECISION(cu_cmplx_PRECISION* prpT,
                                                   cu_cmplx_PRECISION const* phi,
                                                   size_t num_sites) {
  __shared__ cu_cmplx_PRECISION sharedPrpT[6 * diracCommonBlockSize];
  auto localPrpT = sharedPrpT + 6 * threadIdx.x;
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  auto caPhi = ComponentAccess(phi + idx, num_sites);
  // there is no more site for this index
  if (idx >= num_sites) goto copymem;
  localPrpT[0] = caPhi[0] - GAMMA_T_SPIN0_VAL * caPhi[3 * GAMMA_T_SPIN0_CO + 0];
  localPrpT[1] = caPhi[1] - GAMMA_T_SPIN0_VAL * caPhi[3 * GAMMA_T_SPIN0_CO + 1];
  localPrpT[2] = caPhi[2] - GAMMA_T_SPIN0_VAL * caPhi[3 * GAMMA_T_SPIN0_CO + 2];
  localPrpT[3] = caPhi[3] - GAMMA_T_SPIN1_VAL * caPhi[3 * GAMMA_T_SPIN1_CO + 0];
  localPrpT[4] = caPhi[4] - GAMMA_T_SPIN1_VAL * caPhi[3 * GAMMA_T_SPIN1_CO + 1];
  localPrpT[5] = caPhi[5] - GAMMA_T_SPIN1_VAL * caPhi[3 * GAMMA_T_SPIN1_CO + 2];
copymem:
  __syncthreads();
  // advance prpT to first element of block
  prpT += 6 * blockDim.x * blockIdx.x;
  copyChunksToConsecutiveAsBlock(prpT, sharedPrpT, 6, 0,
                                 min(diracCommonBlockSize, num_sites - blockDim.x * blockIdx.x));
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
  __shared__ cu_cmplx_PRECISION sharedPrpZ[6 * diracCommonBlockSize];
  auto localPrpZ = sharedPrpZ + 6 * threadIdx.x;
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  auto caPhi = ComponentAccess(phi + idx, num_sites);
  // there is no more site for this index
  if (idx >= num_sites) goto copymem;
  localPrpZ[0] = caPhi[0] - GAMMA_Z_SPIN0_VAL * caPhi[3 * GAMMA_Z_SPIN0_CO + 0];
  localPrpZ[1] = caPhi[1] - GAMMA_Z_SPIN0_VAL * caPhi[3 * GAMMA_Z_SPIN0_CO + 1];
  localPrpZ[2] = caPhi[2] - GAMMA_Z_SPIN0_VAL * caPhi[3 * GAMMA_Z_SPIN0_CO + 2];
  localPrpZ[3] = caPhi[3] - GAMMA_Z_SPIN1_VAL * caPhi[3 * GAMMA_Z_SPIN1_CO + 0];
  localPrpZ[4] = caPhi[4] - GAMMA_Z_SPIN1_VAL * caPhi[3 * GAMMA_Z_SPIN1_CO + 1];
  localPrpZ[5] = caPhi[5] - GAMMA_Z_SPIN1_VAL * caPhi[3 * GAMMA_Z_SPIN1_CO + 2];
copymem:
  __syncthreads();
  // advance prpZ to first element of block
  prpZ += 6 * blockDim.x * blockIdx.x;
  copyChunksToConsecutiveAsBlock(prpZ, sharedPrpZ, 6, 0,
                                 min(diracCommonBlockSize, num_sites - blockDim.x * blockIdx.x));
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
  __shared__ cu_cmplx_PRECISION sharedPrpY[6 * diracCommonBlockSize];
  auto localPrpY = sharedPrpY + 6 * threadIdx.x;
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  auto caPhi = ComponentAccess(phi + idx, num_sites);
  // there is no more site for this index
  if (idx >= num_sites) goto copymem;
  localPrpY[0] = caPhi[0] - GAMMA_Y_SPIN0_VAL * caPhi[3 * GAMMA_Y_SPIN0_CO + 0];
  localPrpY[1] = caPhi[1] - GAMMA_Y_SPIN0_VAL * caPhi[3 * GAMMA_Y_SPIN0_CO + 1];
  localPrpY[2] = caPhi[2] - GAMMA_Y_SPIN0_VAL * caPhi[3 * GAMMA_Y_SPIN0_CO + 2];
  localPrpY[3] = caPhi[3] - GAMMA_Y_SPIN1_VAL * caPhi[3 * GAMMA_Y_SPIN1_CO + 0];
  localPrpY[4] = caPhi[4] - GAMMA_Y_SPIN1_VAL * caPhi[3 * GAMMA_Y_SPIN1_CO + 1];
  localPrpY[5] = caPhi[5] - GAMMA_Y_SPIN1_VAL * caPhi[3 * GAMMA_Y_SPIN1_CO + 2];
copymem:
  __syncthreads();
  // advance prpY to first element of block
  prpY += 6 * blockDim.x * blockIdx.x;
  copyChunksToConsecutiveAsBlock(prpY, sharedPrpY, 6, 0,
                                 min(diracCommonBlockSize, num_sites - blockDim.x * blockIdx.x));
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
  __shared__ cu_cmplx_PRECISION sharedPrpX[6 * diracCommonBlockSize];
  auto localPrpX = sharedPrpX + 6 * threadIdx.x;
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  auto caPhi = ComponentAccess(phi + idx, num_sites);
  // there is no more site for this index
  if (idx >= num_sites) goto copymem;
  localPrpX[0] = caPhi[0] - GAMMA_X_SPIN0_VAL * caPhi[3 * GAMMA_X_SPIN0_CO + 0];
  localPrpX[1] = caPhi[1] - GAMMA_X_SPIN0_VAL * caPhi[3 * GAMMA_X_SPIN0_CO + 1];
  localPrpX[2] = caPhi[2] - GAMMA_X_SPIN0_VAL * caPhi[3 * GAMMA_X_SPIN0_CO + 2];
  localPrpX[3] = caPhi[3] - GAMMA_X_SPIN1_VAL * caPhi[3 * GAMMA_X_SPIN1_CO + 0];
  localPrpX[4] = caPhi[4] - GAMMA_X_SPIN1_VAL * caPhi[3 * GAMMA_X_SPIN1_CO + 1];
  localPrpX[5] = caPhi[5] - GAMMA_X_SPIN1_VAL * caPhi[3 * GAMMA_X_SPIN1_CO + 2];
copymem:
  __syncthreads();
  // advance prpY to first element of block
  prpX += 6 * blockDim.x * blockIdx.x;
  copyChunksToConsecutiveAsBlock(prpX, sharedPrpX, 6, 0,
                                 min(diracCommonBlockSize, num_sites - blockDim.x * blockIdx.x));
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

__device__ void _store_neighbors_precision(cu_cmplx_PRECISION* dst, cu_cmplx_PRECISION const* src,
                                          int const* neighbors) {

  constexpr uint chunkSize = 6;
  for (size_t i = 0; i < chunkSize / 2; i++) {
    size_t consecutiveIdx = (i * blockDim.x) + threadIdx.x;
    size_t chunkIdx = neighbors[(consecutiveIdx / chunkSize) * 4];

    size_t valueIdx = consecutiveIdx % chunkSize;
    dst[chunkIdx * chunkSize + valueIdx] = src[consecutiveIdx];
  }
}

__global__ void cuda_prn_mvmh_componentwise_PRECISION(cu_cmplx_PRECISION* prp_buf,
                                                      cu_cmplx_PRECISION const* D,
                                                      cu_cmplx_PRECISION const* pbuf,
                                                      int const* neighbors, LatticeAxis dim,
                                                      size_t num_sites) {
  // WARNING! Probably can only be launched with blockSize multiple of 2.
  __shared__ cu_cmplx_PRECISION sharedPrpBuf[3 * diracCommonBlockSize];
  auto localPrpBuf = sharedPrpBuf + 3 * threadIdx.x;
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

  // first neighbor in block
  // careful with the integer division here:  4 * (3 / 2) == 4 != 4 * 3 / 2 == 6
  neighbors += 4 * ((blockDim.x * blockIdx.x) / 2) + neighbor_offset;
  // We operate in steps of 3 here as the application of D happens as 3x3 matrix vector
  // multiplications. There will be two mvms per lattice site.
  if (idx % 2 == 1) {
    // Even indices access elements 0..2
    // Uneven indices access elements 3..5
    pbuf += 3 * num_sites;
  }
  auto caD = ComponentAccess(D + lattice_idx, num_sites);

  auto caPbuf = ComponentAccess(pbuf + lattice_idx, num_sites);
  cuda_mvmh_componentwise_PRECISION(localPrpBuf, caD, caPbuf);
  __syncthreads();
  _store_neighbors_precision(prp_buf, sharedPrpBuf, neighbors);
}

__device__ void _load_neighbors_precision(cu_cmplx_PRECISION* dst, cu_cmplx_PRECISION const* src,
                                          int const* neighbors) {
  constexpr uint chunkSize = 6;
  for (size_t i = 0; i < chunkSize / 2; i++) {
    size_t consecutiveIdx = (i * blockDim.x) + threadIdx.x;
    size_t chunkIdx = neighbors[(consecutiveIdx / chunkSize) * 4];
    size_t valueIdx = consecutiveIdx % chunkSize;
    dst[consecutiveIdx] = src[chunkIdx * chunkSize + valueIdx];
  }
}

__global__ void cuda_pbp_su3_mvm_componentwise_PRECISION(cu_cmplx_PRECISION* pbuf,
                                                         cu_cmplx_PRECISION const* D,
                                                         cu_cmplx_PRECISION const* prn_buf,
                                                         int const* neighbors, LatticeAxis dim,
                                                         size_t num_sites) {
  // WARNING! Probably can only be launched with blockSize multiple of 2.
  __shared__ cu_cmplx_PRECISION sharedPrnBuf[3 * diracCommonBlockSize];
  auto localPrnBuf = sharedPrnBuf + 3 * threadIdx.x;
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

  // first neighbor in block
  // careful with the integer division here:  4 * (3 / 2) == 4 != 4 * 3 / 2 == 6
  neighbors += 4 * ((blockDim.x * blockIdx.x) / 2) + neighbor_offset;
  _load_neighbors_precision(sharedPrnBuf, prn_buf, neighbors);
  __syncthreads();
  // We operate in steps of 3 here as the application of D happens as 3x3 matrix vector
  // multiplications. There will be two mvms per lattice site.
  auto caD = ComponentAccess(D + lattice_idx, num_sites);
  auto caPbuf = ComponentAccess(pbuf + lattice_idx + (idx % 2 == 0 ? 0 : 3) * num_sites, num_sites);
  cuda_mvm_componentwise_PRECISION(caPbuf, caD, localPrnBuf);
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
  auto caPbuf = ComponentAccess(pbuf + idx, num_sites);
  caEta[0] -= caPbuf[0];
  caEta[1] -= caPbuf[1];
  caEta[2] -= caPbuf[2];
  caEta[3] -= caPbuf[3];
  caEta[4] -= caPbuf[4];
  caEta[5] -= caPbuf[5];
  caEta[6] += GAMMA_T_SPIN2_VAL * caPbuf[3 * GAMMA_T_SPIN2_CO];
  caEta[7] += GAMMA_T_SPIN2_VAL * caPbuf[3 * GAMMA_T_SPIN2_CO + 1];
  caEta[8] += GAMMA_T_SPIN2_VAL * caPbuf[3 * GAMMA_T_SPIN2_CO + 2];
  caEta[9] += GAMMA_T_SPIN3_VAL * caPbuf[3 * GAMMA_T_SPIN3_CO];
  caEta[10] += GAMMA_T_SPIN3_VAL * caPbuf[3 * GAMMA_T_SPIN3_CO + 1];
  caEta[11] += GAMMA_T_SPIN3_VAL * caPbuf[3 * GAMMA_T_SPIN3_CO + 2];
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
  auto caPbuf = ComponentAccess(pbuf + idx, num_sites);
  caEta[0] -= caPbuf[0];
  caEta[1] -= caPbuf[1];
  caEta[2] -= caPbuf[2];
  caEta[3] -= caPbuf[3];
  caEta[4] -= caPbuf[4];
  caEta[5] -= caPbuf[5];
  caEta[6] += GAMMA_Z_SPIN2_VAL * caPbuf[3 * GAMMA_Z_SPIN2_CO];
  caEta[7] += GAMMA_Z_SPIN2_VAL * caPbuf[3 * GAMMA_Z_SPIN2_CO + 1];
  caEta[8] += GAMMA_Z_SPIN2_VAL * caPbuf[3 * GAMMA_Z_SPIN2_CO + 2];
  caEta[9] += GAMMA_Z_SPIN3_VAL * caPbuf[3 * GAMMA_Z_SPIN3_CO];
  caEta[10] += GAMMA_Z_SPIN3_VAL * caPbuf[3 * GAMMA_Z_SPIN3_CO + 1];
  caEta[11] += GAMMA_Z_SPIN3_VAL * caPbuf[3 * GAMMA_Z_SPIN3_CO + 2];
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
  auto caPbuf = ComponentAccess(pbuf + idx, num_sites);
  caEta[0] -= caPbuf[0];
  caEta[1] -= caPbuf[1];
  caEta[2] -= caPbuf[2];
  caEta[3] -= caPbuf[3];
  caEta[4] -= caPbuf[4];
  caEta[5] -= caPbuf[5];
  caEta[6] += GAMMA_Y_SPIN2_VAL * caPbuf[3 * GAMMA_Y_SPIN2_CO];
  caEta[7] += GAMMA_Y_SPIN2_VAL * caPbuf[3 * GAMMA_Y_SPIN2_CO + 1];
  caEta[8] += GAMMA_Y_SPIN2_VAL * caPbuf[3 * GAMMA_Y_SPIN2_CO + 2];
  caEta[9] += GAMMA_Y_SPIN3_VAL * caPbuf[3 * GAMMA_Y_SPIN3_CO];
  caEta[10] += GAMMA_Y_SPIN3_VAL * caPbuf[3 * GAMMA_Y_SPIN3_CO + 1];
  caEta[11] += GAMMA_Y_SPIN3_VAL * caPbuf[3 * GAMMA_Y_SPIN3_CO + 2];
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
  auto caPbuf = ComponentAccess(pbuf + idx, num_sites);
  caEta[0] -= caPbuf[0];
  caEta[1] -= caPbuf[1];
  caEta[2] -= caPbuf[2];
  caEta[3] -= caPbuf[3];
  caEta[4] -= caPbuf[4];
  caEta[5] -= caPbuf[5];
  caEta[6] += GAMMA_X_SPIN2_VAL * caPbuf[3 * GAMMA_X_SPIN2_CO];
  caEta[7] += GAMMA_X_SPIN2_VAL * caPbuf[3 * GAMMA_X_SPIN2_CO + 1];
  caEta[8] += GAMMA_X_SPIN2_VAL * caPbuf[3 * GAMMA_X_SPIN2_CO + 2];
  caEta[9] += GAMMA_X_SPIN3_VAL * caPbuf[3 * GAMMA_X_SPIN3_CO];
  caEta[10] += GAMMA_X_SPIN3_VAL * caPbuf[3 * GAMMA_X_SPIN3_CO + 1];
  caEta[11] += GAMMA_X_SPIN3_VAL * caPbuf[3 * GAMMA_X_SPIN3_CO + 2];
}

__global__ void cuda_pbn_su3_T_componentwise_PRECISION(cu_cmplx_PRECISION* eta,
                                                       cu_cmplx_PRECISION const* prpT,
                                                       size_t num_sites) {
  __shared__ cu_cmplx_PRECISION sharedPrpT[6 * diracCommonBlockSize];
  auto localPrpT = sharedPrpT + 6 * threadIdx.x;
  // advance prpT to first element of block
  prpT += 6 * blockDim.x * blockIdx.x;
  copyChunksToConsecutiveAsBlock(sharedPrpT, prpT, 6, 0,
                                 min(diracCommonBlockSize, num_sites - blockDim.x * blockIdx.x));
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caEta = ComponentAccess(eta + idx, num_sites);
  caEta[0] -= localPrpT[0];
  caEta[1] -= localPrpT[1];
  caEta[2] -= localPrpT[2];
  caEta[3] -= localPrpT[3];
  caEta[4] -= localPrpT[4];
  caEta[5] -= localPrpT[5];
  caEta[6] -= GAMMA_T_SPIN2_VAL * localPrpT[3 * GAMMA_T_SPIN2_CO];
  caEta[7] -= GAMMA_T_SPIN2_VAL * localPrpT[3 * GAMMA_T_SPIN2_CO + 1];
  caEta[8] -= GAMMA_T_SPIN2_VAL * localPrpT[3 * GAMMA_T_SPIN2_CO + 2];
  caEta[9] -= GAMMA_T_SPIN3_VAL * localPrpT[3 * GAMMA_T_SPIN3_CO];
  caEta[10] -= GAMMA_T_SPIN3_VAL * localPrpT[3 * GAMMA_T_SPIN3_CO + 1];
  caEta[11] -= GAMMA_T_SPIN3_VAL * localPrpT[3 * GAMMA_T_SPIN3_CO + 2];
}

__global__ void cuda_pbn_su3_Z_componentwise_PRECISION(cu_cmplx_PRECISION* eta,
                                                       cu_cmplx_PRECISION const* prpZ,
                                                       size_t num_sites) {
  __shared__ cu_cmplx_PRECISION sharedPrpZ[6 * diracCommonBlockSize];
  auto localPrpZ = sharedPrpZ + 6 * threadIdx.x;
  // advance prpZ to first element of block
  prpZ += 6 * blockDim.x * blockIdx.x;
  copyChunksToConsecutiveAsBlock(sharedPrpZ, prpZ, 6, 0,
                                 min(diracCommonBlockSize, num_sites - blockDim.x * blockIdx.x));
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caEta = ComponentAccess(eta + idx, num_sites);
  caEta[0] -= localPrpZ[0];
  caEta[1] -= localPrpZ[1];
  caEta[2] -= localPrpZ[2];
  caEta[3] -= localPrpZ[3];
  caEta[4] -= localPrpZ[4];
  caEta[5] -= localPrpZ[5];
  caEta[6] -= GAMMA_Z_SPIN2_VAL * localPrpZ[3 * GAMMA_Z_SPIN2_CO];
  caEta[7] -= GAMMA_Z_SPIN2_VAL * localPrpZ[3 * GAMMA_Z_SPIN2_CO + 1];
  caEta[8] -= GAMMA_Z_SPIN2_VAL * localPrpZ[3 * GAMMA_Z_SPIN2_CO + 2];
  caEta[9] -= GAMMA_Z_SPIN3_VAL * localPrpZ[3 * GAMMA_Z_SPIN3_CO];
  caEta[10] -= GAMMA_Z_SPIN3_VAL * localPrpZ[3 * GAMMA_Z_SPIN3_CO + 1];
  caEta[11] -= GAMMA_Z_SPIN3_VAL * localPrpZ[3 * GAMMA_Z_SPIN3_CO + 2];
}

__global__ void cuda_pbn_su3_Y_componentwise_PRECISION(cu_cmplx_PRECISION* eta,
                                                       cu_cmplx_PRECISION const* prpY,
                                                       size_t num_sites) {
  __shared__ cu_cmplx_PRECISION sharedPrpY[6 * diracCommonBlockSize];
  auto localPrpY = sharedPrpY + 6 * threadIdx.x;
  // advance prpY to first element of block
  prpY += 6 * blockDim.x * blockIdx.x;
  copyChunksToConsecutiveAsBlock(sharedPrpY, prpY, 6, 0,
                                 min(diracCommonBlockSize, num_sites - blockDim.x * blockIdx.x));
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caEta = ComponentAccess(eta + idx, num_sites);
  caEta[0] -= localPrpY[0];
  caEta[1] -= localPrpY[1];
  caEta[2] -= localPrpY[2];
  caEta[3] -= localPrpY[3];
  caEta[4] -= localPrpY[4];
  caEta[5] -= localPrpY[5];
  caEta[6] -= GAMMA_Y_SPIN2_VAL * localPrpY[3 * GAMMA_Y_SPIN2_CO];
  caEta[7] -= GAMMA_Y_SPIN2_VAL * localPrpY[3 * GAMMA_Y_SPIN2_CO + 1];
  caEta[8] -= GAMMA_Y_SPIN2_VAL * localPrpY[3 * GAMMA_Y_SPIN2_CO + 2];
  caEta[9] -= GAMMA_Y_SPIN3_VAL * localPrpY[3 * GAMMA_Y_SPIN3_CO];
  caEta[10] -= GAMMA_Y_SPIN3_VAL * localPrpY[3 * GAMMA_Y_SPIN3_CO + 1];
  caEta[11] -= GAMMA_Y_SPIN3_VAL * localPrpY[3 * GAMMA_Y_SPIN3_CO + 2];
}

__global__ void cuda_pbn_su3_X_componentwise_PRECISION(cu_cmplx_PRECISION* eta,
                                                       cu_cmplx_PRECISION const* prpX,
                                                       size_t num_sites) {
  __shared__ cu_cmplx_PRECISION sharedPrpX[6 * diracCommonBlockSize];
  auto localPrpX = sharedPrpX + 6 * threadIdx.x;
  // advance prpX to first element of block
  prpX += 6 * blockDim.x * blockIdx.x;
  copyChunksToConsecutiveAsBlock(sharedPrpX, prpX, 6, 0,
                                 min(diracCommonBlockSize, num_sites - blockDim.x * blockIdx.x));
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites) {
    // there is no more site for this index
    return;
  }
  auto caEta = ComponentAccess(eta + idx, num_sites);
  prpX += 6 * idx;
  caEta[0] -= localPrpX[0];
  caEta[1] -= localPrpX[1];
  caEta[2] -= localPrpX[2];
  caEta[3] -= localPrpX[3];
  caEta[4] -= localPrpX[4];
  caEta[5] -= localPrpX[5];
  caEta[6] -= GAMMA_X_SPIN2_VAL * localPrpX[3 * GAMMA_X_SPIN2_CO];
  caEta[7] -= GAMMA_X_SPIN2_VAL * localPrpX[3 * GAMMA_X_SPIN2_CO + 1];
  caEta[8] -= GAMMA_X_SPIN2_VAL * localPrpX[3 * GAMMA_X_SPIN2_CO + 2];
  caEta[9] -= GAMMA_X_SPIN3_VAL * localPrpX[3 * GAMMA_X_SPIN3_CO];
  caEta[10] -= GAMMA_X_SPIN3_VAL * localPrpX[3 * GAMMA_X_SPIN3_CO + 1];
  caEta[11] -= GAMMA_X_SPIN3_VAL * localPrpX[3 * GAMMA_X_SPIN3_CO + 2];
}
