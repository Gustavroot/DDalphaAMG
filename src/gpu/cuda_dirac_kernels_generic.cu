#include "cuda_dirac_kernels_PRECISION.h"
#include "global_enums.h"
#include "cuda_complex.h"
#include "cuda_complex_cxx.h"
#include "cuda_complex_operators_PRECISION.h"
#include "cuda_complex_operators.h"
#include "cuda_mvm_PRECISION.h"

// The clifford header uses a C compiler extension version of I that is not compatible with CUDA.
// CU_OVERWRITE_I replaces that.
#define CU_OVERWRITE_I
#include "clifford.h"
#undef CU_OVERWRITE_I

__global__ void cuda_site_clover_PRECISION(cuda_vector_PRECISION eta, cu_cmplx_PRECISION const* phi,
                                           cu_cmplx_PRECISION const* clover, size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites){
    // there is no more site for this index
    return;
  }
  eta += 12*idx;
  phi += 12*idx;
  clover += 42*idx;

    // diagonal
    eta[ 0] = clover[ 0]*phi[ 0];
    eta[ 1] = clover[ 1]*phi[ 1];
    eta[ 2] = clover[ 2]*phi[ 2];
    eta[ 3] = clover[ 3]*phi[ 3];
    eta[ 4] = clover[ 4]*phi[ 4];
    eta[ 5] = clover[ 5]*phi[ 5];
    eta[ 6] = clover[ 6]*phi[ 6];
    eta[ 7] = clover[ 7]*phi[ 7];
    eta[ 8] = clover[ 8]*phi[ 8];
    eta[ 9] = clover[ 9]*phi[ 9];
    eta[10] = clover[10]*phi[10];
    eta[11] = clover[11]*phi[11];
    // spin 0 and 1, row major
    eta[0] += clover[12]*phi[1];
    eta[0] += clover[13]*phi[2];
    eta[0] += clover[14]*phi[3];
    eta[0] += clover[15]*phi[4];
    eta[0] += clover[16]*phi[5];
    eta[1] += clover[17]*phi[2];
    eta[1] += clover[18]*phi[3];
    eta[1] += clover[19]*phi[4];
    eta[1] += clover[20]*phi[5];
    eta[2] += clover[21]*phi[3];
    eta[2] += clover[22]*phi[4];
    eta[2] += clover[23]*phi[5];
    eta[3] += clover[24]*phi[4];
    eta[3] += clover[25]*phi[5];
    eta[4] += clover[26]*phi[5];
    eta[1] += cu_conj_PRECISION(clover[12])*phi[0];
    eta[2] += cu_conj_PRECISION(clover[13])*phi[0];
    eta[3] += cu_conj_PRECISION(clover[14])*phi[0];
    eta[4] += cu_conj_PRECISION(clover[15])*phi[0];
    eta[5] += cu_conj_PRECISION(clover[16])*phi[0];
    eta[2] += cu_conj_PRECISION(clover[17])*phi[1];
    eta[3] += cu_conj_PRECISION(clover[18])*phi[1];
    eta[4] += cu_conj_PRECISION(clover[19])*phi[1];
    eta[5] += cu_conj_PRECISION(clover[20])*phi[1];
    eta[3] += cu_conj_PRECISION(clover[21])*phi[2];
    eta[4] += cu_conj_PRECISION(clover[22])*phi[2];
    eta[5] += cu_conj_PRECISION(clover[23])*phi[2];
    eta[4] += cu_conj_PRECISION(clover[24])*phi[3];
    eta[5] += cu_conj_PRECISION(clover[25])*phi[3];
    eta[5] += cu_conj_PRECISION(clover[26])*phi[4];
    // spin 2 and 3, row major
    eta[ 6] += clover[27]*phi[ 7];
    eta[ 6] += clover[28]*phi[ 8];
    eta[ 6] += clover[29]*phi[ 9];
    eta[ 6] += clover[30]*phi[10];
    eta[ 6] += clover[31]*phi[11];
    eta[ 7] += clover[32]*phi[ 8];
    eta[ 7] += clover[33]*phi[ 9];
    eta[ 7] += clover[34]*phi[10];
    eta[ 7] += clover[35]*phi[11];
    eta[ 8] += clover[36]*phi[ 9];
    eta[ 8] += clover[37]*phi[10];
    eta[ 8] += clover[38]*phi[11];
    eta[ 9] += clover[39]*phi[10];
    eta[ 9] += clover[40]*phi[11];
    eta[10] += clover[41]*phi[11];
    eta[ 7] += cu_conj_PRECISION(clover[27])*phi[ 6];
    eta[ 8] += cu_conj_PRECISION(clover[28])*phi[ 6];
    eta[ 9] += cu_conj_PRECISION(clover[29])*phi[ 6];
    eta[10] += cu_conj_PRECISION(clover[30])*phi[ 6];
    eta[11] += cu_conj_PRECISION(clover[31])*phi[ 6];
    eta[ 8] += cu_conj_PRECISION(clover[32])*phi[ 7];
    eta[ 9] += cu_conj_PRECISION(clover[33])*phi[ 7];
    eta[10] += cu_conj_PRECISION(clover[34])*phi[ 7];
    eta[11] += cu_conj_PRECISION(clover[35])*phi[ 7];
    eta[ 9] += cu_conj_PRECISION(clover[36])*phi[ 8];
    eta[10] += cu_conj_PRECISION(clover[37])*phi[ 8];
    eta[11] += cu_conj_PRECISION(clover[38])*phi[ 8];
    eta[10] += cu_conj_PRECISION(clover[39])*phi[ 9];
    eta[11] += cu_conj_PRECISION(clover[40])*phi[ 9];
    eta[11] += cu_conj_PRECISION(clover[41])*phi[10];
}

__global__ void cuda_prp_T_PRECISION(cu_cmplx_PRECISION * prpT, cu_cmplx_PRECISION const * phi,
                                size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites){
    // there is no more site for this index
    return;
  }
  phi += 12*idx;
  prpT += 6*idx;
  prpT[0] = phi[0] -GAMMA_T_SPIN0_VAL*phi[3*GAMMA_T_SPIN0_CO];
  prpT[1] = phi[1] -GAMMA_T_SPIN0_VAL*phi[3*GAMMA_T_SPIN0_CO+1];
  prpT[2] = phi[2] -GAMMA_T_SPIN0_VAL*phi[3*GAMMA_T_SPIN0_CO+2];
  prpT[3] = phi[3] -GAMMA_T_SPIN1_VAL*phi[3*GAMMA_T_SPIN1_CO];
  prpT[4] = phi[4] -GAMMA_T_SPIN1_VAL*phi[3*GAMMA_T_SPIN1_CO+1];
  prpT[5] = phi[5] -GAMMA_T_SPIN1_VAL*phi[3*GAMMA_T_SPIN1_CO+2];
}

__global__ void cuda_prn_T_PRECISION(cu_cmplx_PRECISION* prnT, cu_cmplx_PRECISION const* phi,
                                     size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites){
    // there is no more site for this index
    return;
  }
  phi += 12*idx;
  prnT += 6*idx;
  prnT[0] = phi[0] +GAMMA_T_SPIN0_VAL*phi[3*GAMMA_T_SPIN0_CO];
  prnT[1] = phi[1] +GAMMA_T_SPIN0_VAL*phi[3*GAMMA_T_SPIN0_CO+1];
  prnT[2] = phi[2] +GAMMA_T_SPIN0_VAL*phi[3*GAMMA_T_SPIN0_CO+2];
  prnT[3] = phi[3] +GAMMA_T_SPIN1_VAL*phi[3*GAMMA_T_SPIN1_CO];
  prnT[4] = phi[4] +GAMMA_T_SPIN1_VAL*phi[3*GAMMA_T_SPIN1_CO+1];
  prnT[5] = phi[5] +GAMMA_T_SPIN1_VAL*phi[3*GAMMA_T_SPIN1_CO+2];
}

__global__ void cuda_prp_Z_PRECISION(cu_cmplx_PRECISION * prpZ, cu_cmplx_PRECISION const * phi,
                                size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites){
    // there is no more site for this index
    return;
  }
  phi += 12*idx;
  prpZ += 6*idx;
  prpZ[0] = phi[0] - GAMMA_Z_SPIN0_VAL*phi[3*GAMMA_Z_SPIN0_CO];
  prpZ[1] = phi[1] - GAMMA_Z_SPIN0_VAL*phi[3*GAMMA_Z_SPIN0_CO+1];
  prpZ[2] = phi[2] - GAMMA_Z_SPIN0_VAL*phi[3*GAMMA_Z_SPIN0_CO+2];
  prpZ[3] = phi[3] - GAMMA_Z_SPIN1_VAL*phi[3*GAMMA_Z_SPIN1_CO];
  prpZ[4] = phi[4] - GAMMA_Z_SPIN1_VAL*phi[3*GAMMA_Z_SPIN1_CO+1];
  prpZ[5] = phi[5] - GAMMA_Z_SPIN1_VAL*phi[3*GAMMA_Z_SPIN1_CO+2];
}

__global__ void cuda_prn_Z_PRECISION(cu_cmplx_PRECISION* prnZ, cu_cmplx_PRECISION const* phi,
                                     size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites){
    // there is no more site for this index
    return;
  }
  phi += 12*idx;
  prnZ += 6*idx;
  prnZ[0] = phi[0] +GAMMA_Z_SPIN0_VAL*phi[3*GAMMA_Z_SPIN0_CO];
  prnZ[1] = phi[1] +GAMMA_Z_SPIN0_VAL*phi[3*GAMMA_Z_SPIN0_CO+1];
  prnZ[2] = phi[2] +GAMMA_Z_SPIN0_VAL*phi[3*GAMMA_Z_SPIN0_CO+2];
  prnZ[3] = phi[3] +GAMMA_Z_SPIN1_VAL*phi[3*GAMMA_Z_SPIN1_CO];
  prnZ[4] = phi[4] +GAMMA_Z_SPIN1_VAL*phi[3*GAMMA_Z_SPIN1_CO+1];
  prnZ[5] = phi[5] +GAMMA_Z_SPIN1_VAL*phi[3*GAMMA_Z_SPIN1_CO+2];
}

__global__ void cuda_prp_Y_PRECISION(cu_cmplx_PRECISION* prpY, cu_cmplx_PRECISION const* phi,
                                     size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites){
    // there is no more site for this index
    return;
  }
  phi += 12*idx;
  prpY += 6*idx;
  prpY[0] = phi[0] -GAMMA_Y_SPIN0_VAL*phi[3*GAMMA_Y_SPIN0_CO];
  prpY[1] = phi[1] -GAMMA_Y_SPIN0_VAL*phi[3*GAMMA_Y_SPIN0_CO+1];
  prpY[2] = phi[2] -GAMMA_Y_SPIN0_VAL*phi[3*GAMMA_Y_SPIN0_CO+2];
  prpY[3] = phi[3] -GAMMA_Y_SPIN1_VAL*phi[3*GAMMA_Y_SPIN1_CO];
  prpY[4] = phi[4] -GAMMA_Y_SPIN1_VAL*phi[3*GAMMA_Y_SPIN1_CO+1];
  prpY[5] = phi[5] -GAMMA_Y_SPIN1_VAL*phi[3*GAMMA_Y_SPIN1_CO+2];
}

__global__ void cuda_prn_Y_PRECISION(cu_cmplx_PRECISION* prnY, cu_cmplx_PRECISION const* phi,
                                     size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites){
    // there is no more site for this index
    return;
  }
  phi += 12*idx;
  prnY += 6*idx;
  prnY[0] = phi[0] +GAMMA_Y_SPIN0_VAL*phi[3*GAMMA_Y_SPIN0_CO];
  prnY[1] = phi[1] +GAMMA_Y_SPIN0_VAL*phi[3*GAMMA_Y_SPIN0_CO+1];
  prnY[2] = phi[2] +GAMMA_Y_SPIN0_VAL*phi[3*GAMMA_Y_SPIN0_CO+2];
  prnY[3] = phi[3] +GAMMA_Y_SPIN1_VAL*phi[3*GAMMA_Y_SPIN1_CO];
  prnY[4] = phi[4] +GAMMA_Y_SPIN1_VAL*phi[3*GAMMA_Y_SPIN1_CO+1];
  prnY[5] = phi[5] +GAMMA_Y_SPIN1_VAL*phi[3*GAMMA_Y_SPIN1_CO+2];
}

__global__ void cuda_prp_X_PRECISION(cu_cmplx_PRECISION* prpX, cu_cmplx_PRECISION const* phi,
                                     size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites){
    // there is no more site for this index
    return;
  }
  phi += 12*idx;
  prpX += 6*idx;
  prpX[0] = phi[0] -GAMMA_X_SPIN0_VAL*phi[3*GAMMA_X_SPIN0_CO];
  prpX[1] = phi[1] -GAMMA_X_SPIN0_VAL*phi[3*GAMMA_X_SPIN0_CO+1];
  prpX[2] = phi[2] -GAMMA_X_SPIN0_VAL*phi[3*GAMMA_X_SPIN0_CO+2];
  prpX[3] = phi[3] -GAMMA_X_SPIN1_VAL*phi[3*GAMMA_X_SPIN1_CO];
  prpX[4] = phi[4] -GAMMA_X_SPIN1_VAL*phi[3*GAMMA_X_SPIN1_CO+1];
  prpX[5] = phi[5] -GAMMA_X_SPIN1_VAL*phi[3*GAMMA_X_SPIN1_CO+2];
}

__global__ void cuda_prn_X_PRECISION(cu_cmplx_PRECISION* prnX, cu_cmplx_PRECISION const* phi,
                                     size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites){
    // there is no more site for this index
    return;
  }
  phi += 12*idx;
  prnX += 6*idx;
  prnX[0] = phi[0] +GAMMA_X_SPIN0_VAL*phi[3*GAMMA_X_SPIN0_CO];
  prnX[1] = phi[1] +GAMMA_X_SPIN0_VAL*phi[3*GAMMA_X_SPIN0_CO+1];
  prnX[2] = phi[2] +GAMMA_X_SPIN0_VAL*phi[3*GAMMA_X_SPIN0_CO+2];
  prnX[3] = phi[3] +GAMMA_X_SPIN1_VAL*phi[3*GAMMA_X_SPIN1_CO];
  prnX[4] = phi[4] +GAMMA_X_SPIN1_VAL*phi[3*GAMMA_X_SPIN1_CO+1];
  prnX[5] = phi[5] +GAMMA_X_SPIN1_VAL*phi[3*GAMMA_X_SPIN1_CO+2];
}

__global__ void cuda_prn_mvmh_PRECISION(cu_cmplx_PRECISION* prp_buf, cu_cmplx_PRECISION const* D,
                                        cu_cmplx_PRECISION const* pbuf, int const* neighbors,
                                        LatticeAxis dim, size_t num_sites) {
  unsigned int neighbor_offset;
  switch (dim)
  {
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
  const size_t lattice_idx = idx/2;
  
  if (lattice_idx >= num_sites){
    // there is no more site for this index
    return;
  }
  D += 9*(4*lattice_idx+neighbor_offset);
  neighbors += 4*lattice_idx+neighbor_offset;
  // We operate in steps of 3 here as the application of D happens as 3x3 matrix vector
  // multiplications. There will be two mvms per lattice site.
  pbuf += 3*idx;
  const size_t j = 6*(*neighbors);
  prp_buf += j+(idx%2==0?0:3);
  cuda_mvmh_PRECISION(prp_buf, D, pbuf);
}

__global__ void cuda_pbp_su3_mvm_PRECISION(cu_cmplx_PRECISION* pbuf, cu_cmplx_PRECISION const* D,
                                           cu_cmplx_PRECISION const* prn_buf, int const * neighbors,
                                           LatticeAxis dim, size_t num_sites) {
  unsigned int neighbor_offset;
  switch (dim)
  {
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
  const size_t lattice_idx = idx/2;
  
  if (lattice_idx >= num_sites){
    // there is no more site for this index
    return;
  }
  D += 9*(4*lattice_idx+neighbor_offset);
  neighbors += 4*lattice_idx+neighbor_offset;
  // We operate in steps of 3 here as the application of D happens as 3x3 matrix vector
  // multiplications. There will be two mvms per lattice site.
  pbuf += 3*idx;
  const size_t j = 6*(*neighbors);
  prn_buf += j+(idx%2==0?0:3);
  cuda_mvm_PRECISION(pbuf, D, prn_buf);
}

__global__ void cuda_pbp_su3_T_PRECISION(cu_cmplx_PRECISION* eta, cu_cmplx_PRECISION const * pbuf,
                                    size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites){
    // there is no more site for this index
    return;
  }
  eta += 12*idx;
  pbuf += 6*idx;
  eta[ 0] -= pbuf[0];
  eta[ 1] -= pbuf[1];
  eta[ 2] -= pbuf[2];
  eta[ 3] -= pbuf[3];
  eta[ 4] -= pbuf[4];
  eta[ 5] -= pbuf[5];
  eta[ 6] += GAMMA_T_SPIN2_VAL*pbuf[3*GAMMA_T_SPIN2_CO];
  eta[ 7] += GAMMA_T_SPIN2_VAL*pbuf[3*GAMMA_T_SPIN2_CO+1];
  eta[ 8] += GAMMA_T_SPIN2_VAL*pbuf[3*GAMMA_T_SPIN2_CO+2];
  eta[ 9] += GAMMA_T_SPIN3_VAL*pbuf[3*GAMMA_T_SPIN3_CO];
  eta[10] += GAMMA_T_SPIN3_VAL*pbuf[3*GAMMA_T_SPIN3_CO+1];
  eta[11] += GAMMA_T_SPIN3_VAL*pbuf[3*GAMMA_T_SPIN3_CO+2];
}

__global__ void cuda_pbp_su3_Z_PRECISION(cu_cmplx_PRECISION* eta, cu_cmplx_PRECISION const * pbuf,
                                    size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites){
    // there is no more site for this index
    return;
  }
  eta += 12*idx;
  pbuf += 6*idx;
  eta[ 0] -= pbuf[0];
  eta[ 1] -= pbuf[1];
  eta[ 2] -= pbuf[2];
  eta[ 3] -= pbuf[3];
  eta[ 4] -= pbuf[4];
  eta[ 5] -= pbuf[5];
  eta[ 6] += GAMMA_Z_SPIN2_VAL*pbuf[3*GAMMA_Z_SPIN2_CO];
  eta[ 7] += GAMMA_Z_SPIN2_VAL*pbuf[3*GAMMA_Z_SPIN2_CO+1];
  eta[ 8] += GAMMA_Z_SPIN2_VAL*pbuf[3*GAMMA_Z_SPIN2_CO+2];
  eta[ 9] += GAMMA_Z_SPIN3_VAL*pbuf[3*GAMMA_Z_SPIN3_CO];
  eta[10] += GAMMA_Z_SPIN3_VAL*pbuf[3*GAMMA_Z_SPIN3_CO+1];
  eta[11] += GAMMA_Z_SPIN3_VAL*pbuf[3*GAMMA_Z_SPIN3_CO+2];
}

__global__ void cuda_pbp_su3_Y_PRECISION(cu_cmplx_PRECISION* eta, cu_cmplx_PRECISION const * pbuf,
                                    size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites){
    // there is no more site for this index
    return;
  }
  eta += 12*idx;
  pbuf += 6*idx;
  eta[ 0] -= pbuf[0];
  eta[ 1] -= pbuf[1];
  eta[ 2] -= pbuf[2];
  eta[ 3] -= pbuf[3];
  eta[ 4] -= pbuf[4];
  eta[ 5] -= pbuf[5];
  eta[ 6] += GAMMA_Y_SPIN2_VAL*pbuf[3*GAMMA_Y_SPIN2_CO];
  eta[ 7] += GAMMA_Y_SPIN2_VAL*pbuf[3*GAMMA_Y_SPIN2_CO+1];
  eta[ 8] += GAMMA_Y_SPIN2_VAL*pbuf[3*GAMMA_Y_SPIN2_CO+2];
  eta[ 9] += GAMMA_Y_SPIN3_VAL*pbuf[3*GAMMA_Y_SPIN3_CO];
  eta[10] += GAMMA_Y_SPIN3_VAL*pbuf[3*GAMMA_Y_SPIN3_CO+1];
  eta[11] += GAMMA_Y_SPIN3_VAL*pbuf[3*GAMMA_Y_SPIN3_CO+2];
}

__global__ void cuda_pbp_su3_X_PRECISION(cu_cmplx_PRECISION* eta, cu_cmplx_PRECISION const * pbuf,
                                    size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites){
    // there is no more site for this index
    return;
  }
  eta += 12*idx;
  pbuf += 6*idx;
  eta[ 0] -= pbuf[0];
  eta[ 1] -= pbuf[1];
  eta[ 2] -= pbuf[2];
  eta[ 3] -= pbuf[3];
  eta[ 4] -= pbuf[4];
  eta[ 5] -= pbuf[5];
  eta[ 6] += GAMMA_X_SPIN2_VAL*pbuf[3*GAMMA_X_SPIN2_CO];
  eta[ 7] += GAMMA_X_SPIN2_VAL*pbuf[3*GAMMA_X_SPIN2_CO+1];
  eta[ 8] += GAMMA_X_SPIN2_VAL*pbuf[3*GAMMA_X_SPIN2_CO+2];
  eta[ 9] += GAMMA_X_SPIN3_VAL*pbuf[3*GAMMA_X_SPIN3_CO];
  eta[10] += GAMMA_X_SPIN3_VAL*pbuf[3*GAMMA_X_SPIN3_CO+1];
  eta[11] += GAMMA_X_SPIN3_VAL*pbuf[3*GAMMA_X_SPIN3_CO+2];
}

__global__ void cuda_pbn_su3_T_PRECISION(cu_cmplx_PRECISION* eta, cu_cmplx_PRECISION const * prpT,
                                         size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites){
    // there is no more site for this index
    return;
  }
  eta += 12*idx;
  prpT += 6*idx;
  eta[ 0] -= prpT[0];
  eta[ 1] -= prpT[1];
  eta[ 2] -= prpT[2];
  eta[ 3] -= prpT[3];
  eta[ 4] -= prpT[4];
  eta[ 5] -= prpT[5];
  eta[ 6] -= GAMMA_T_SPIN2_VAL*prpT[3*GAMMA_T_SPIN2_CO];
  eta[ 7] -= GAMMA_T_SPIN2_VAL*prpT[3*GAMMA_T_SPIN2_CO+1];
  eta[ 8] -= GAMMA_T_SPIN2_VAL*prpT[3*GAMMA_T_SPIN2_CO+2];
  eta[ 9] -= GAMMA_T_SPIN3_VAL*prpT[3*GAMMA_T_SPIN3_CO];
  eta[10] -= GAMMA_T_SPIN3_VAL*prpT[3*GAMMA_T_SPIN3_CO+1];
  eta[11] -= GAMMA_T_SPIN3_VAL*prpT[3*GAMMA_T_SPIN3_CO+2];
}

__global__ void cuda_pbn_su3_Z_PRECISION(cu_cmplx_PRECISION* eta, cu_cmplx_PRECISION const * prpZ,
                                         size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites){
    // there is no more site for this index
    return;
  }
  eta += 12*idx;
  prpZ += 6*idx;
  eta[ 0] -= prpZ[0];
  eta[ 1] -= prpZ[1];
  eta[ 2] -= prpZ[2];
  eta[ 3] -= prpZ[3];
  eta[ 4] -= prpZ[4];
  eta[ 5] -= prpZ[5];
  eta[ 6] -= GAMMA_Z_SPIN2_VAL*prpZ[3*GAMMA_Z_SPIN2_CO];
  eta[ 7] -= GAMMA_Z_SPIN2_VAL*prpZ[3*GAMMA_Z_SPIN2_CO+1];
  eta[ 8] -= GAMMA_Z_SPIN2_VAL*prpZ[3*GAMMA_Z_SPIN2_CO+2];
  eta[ 9] -= GAMMA_Z_SPIN3_VAL*prpZ[3*GAMMA_Z_SPIN3_CO];
  eta[10] -= GAMMA_Z_SPIN3_VAL*prpZ[3*GAMMA_Z_SPIN3_CO+1];
  eta[11] -= GAMMA_Z_SPIN3_VAL*prpZ[3*GAMMA_Z_SPIN3_CO+2];
}

__global__ void cuda_pbn_su3_Y_PRECISION(cu_cmplx_PRECISION* eta, cu_cmplx_PRECISION const * prpY,
                                         size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites){
    // there is no more site for this index
    return;
  }
  eta += 12*idx;
  prpY += 6*idx;
  eta[ 0] -= prpY[0];
  eta[ 1] -= prpY[1];
  eta[ 2] -= prpY[2];
  eta[ 3] -= prpY[3];
  eta[ 4] -= prpY[4];
  eta[ 5] -= prpY[5];
  eta[ 6] -= GAMMA_Y_SPIN2_VAL*prpY[3*GAMMA_Y_SPIN2_CO];
  eta[ 7] -= GAMMA_Y_SPIN2_VAL*prpY[3*GAMMA_Y_SPIN2_CO+1];
  eta[ 8] -= GAMMA_Y_SPIN2_VAL*prpY[3*GAMMA_Y_SPIN2_CO+2];
  eta[ 9] -= GAMMA_Y_SPIN3_VAL*prpY[3*GAMMA_Y_SPIN3_CO];
  eta[10] -= GAMMA_Y_SPIN3_VAL*prpY[3*GAMMA_Y_SPIN3_CO+1];
  eta[11] -= GAMMA_Y_SPIN3_VAL*prpY[3*GAMMA_Y_SPIN3_CO+2];
}

__global__ void cuda_pbn_su3_X_PRECISION(cu_cmplx_PRECISION* eta, cu_cmplx_PRECISION const * prpX,
                                         size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= num_sites){
    // there is no more site for this index
    return;
  }
  eta += 12*idx;
  prpX += 6*idx;
  eta[ 0] -= prpX[0];
  eta[ 1] -= prpX[1];
  eta[ 2] -= prpX[2];
  eta[ 3] -= prpX[3];
  eta[ 4] -= prpX[4];
  eta[ 5] -= prpX[5];
  eta[ 6] -= GAMMA_X_SPIN2_VAL*prpX[3*GAMMA_X_SPIN2_CO];
  eta[ 7] -= GAMMA_X_SPIN2_VAL*prpX[3*GAMMA_X_SPIN2_CO+1];
  eta[ 8] -= GAMMA_X_SPIN2_VAL*prpX[3*GAMMA_X_SPIN2_CO+2];
  eta[ 9] -= GAMMA_X_SPIN3_VAL*prpX[3*GAMMA_X_SPIN3_CO];
  eta[10] -= GAMMA_X_SPIN3_VAL*prpX[3*GAMMA_X_SPIN3_CO+1];
  eta[11] -= GAMMA_X_SPIN3_VAL*prpX[3*GAMMA_X_SPIN3_CO+2];
}
