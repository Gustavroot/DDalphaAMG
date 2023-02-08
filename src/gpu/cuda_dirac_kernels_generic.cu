#include "cuda_complex.h"
#include "cuda_complex_cxx.h"
#include "cuda_complex_operators_PRECISION.h"
#include "cuda_complex_operators.h"
#include "cuda_vectors_PRECISION.h"

// The clifford header uses a C compiler extension version of I that is not compatible with CUDA.
// CU_OVERWRITE_I replaces that.
#define CU_OVERWRITE_I
#include "clifford.h"
#undef CU_OVERWRITE_I

__global__ void cuda_site_clover_PRECISION(cuda_vector_PRECISION eta, cuda_vector_PRECISION phi,
                                           cuda_config_PRECISION clover, size_t num_sites) {
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