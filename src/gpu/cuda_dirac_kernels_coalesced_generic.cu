#include "global_enums.h"
#include "cuda_complex.h"
#include "cuda_complex_cxx.h"
#include "cuda_complex_operators_PRECISION.h"
#include "cuda_complex_operators.h"
#include "cuda_vectors_PRECISION.h"
#include "cuda_mvm_PRECISION.h"
#include "cuda_coalesced.h"
#include "cuda_dirac_kernels_coalesced_PRECISION.h"

// The clifford header uses a C compiler extension version of I that is not compatible with CUDA.
// CU_OVERWRITE_I replaces that.
#define CU_OVERWRITE_I
#include "clifford.h"
#undef CU_OVERWRITE_I

__global__ void cuda_site_clover_coalesced_PRECISION(cuda_vector_PRECISION eta, cu_cmplx_PRECISION const* phi,
                                           cu_cmplx_PRECISION const* clover, size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  constexpr uint phiChunkSize = 12;
  // set phi address to first element handled by block
  phi += phiChunkSize * blockDim.x * blockIdx.x;

  // copy the array elements handled in this block to shared memory
  __shared__ cu_cmplx_PRECISION blockPhi[phiChunkSize * diracCommonBlockSize];
  copy_chunks_to_consecutive_as_block(blockPhi, phi, 12, 0, blockDim.x);

  if (idx >= num_sites){
    // there is no more site for this index
    return;
  }
  eta += 12*idx;
  cu_cmplx_PRECISION * const threadPhi = blockPhi + phiChunkSize * threadIdx.x;
  clover += 42*idx;
  // diagonal
  eta[ 0] = clover[ 0]*threadPhi[ 0];
  eta[ 1] = clover[ 1]*threadPhi[ 1];
  eta[ 2] = clover[ 2]*threadPhi[ 2];
  eta[ 3] = clover[ 3]*threadPhi[ 3];
  eta[ 4] = clover[ 4]*threadPhi[ 4];
  eta[ 5] = clover[ 5]*threadPhi[ 5];
  eta[ 6] = clover[ 6]*threadPhi[ 6];
  eta[ 7] = clover[ 7]*threadPhi[ 7];
  eta[ 8] = clover[ 8]*threadPhi[ 8];
  eta[ 9] = clover[ 9]*threadPhi[ 9];
  eta[10] = clover[10]*threadPhi[10];
  eta[11] = clover[11]*threadPhi[11];
  // spin 0 and 1, row major
  eta[0] += clover[12]*threadPhi[1];
  eta[0] += clover[13]*threadPhi[2];
  eta[0] += clover[14]*threadPhi[3];
  eta[0] += clover[15]*threadPhi[4];
  eta[0] += clover[16]*threadPhi[5];
  eta[1] += clover[17]*threadPhi[2];
  eta[1] += clover[18]*threadPhi[3];
  eta[1] += clover[19]*threadPhi[4];
  eta[1] += clover[20]*threadPhi[5];
  eta[2] += clover[21]*threadPhi[3];
  eta[2] += clover[22]*threadPhi[4];
  eta[2] += clover[23]*threadPhi[5];
  eta[3] += clover[24]*threadPhi[4];
  eta[3] += clover[25]*threadPhi[5];
  eta[4] += clover[26]*threadPhi[5];
  eta[1] += cu_conj_PRECISION(clover[12])*threadPhi[0];
  eta[2] += cu_conj_PRECISION(clover[13])*threadPhi[0];
  eta[3] += cu_conj_PRECISION(clover[14])*threadPhi[0];
  eta[4] += cu_conj_PRECISION(clover[15])*threadPhi[0];
  eta[5] += cu_conj_PRECISION(clover[16])*threadPhi[0];
  eta[2] += cu_conj_PRECISION(clover[17])*threadPhi[1];
  eta[3] += cu_conj_PRECISION(clover[18])*threadPhi[1];
  eta[4] += cu_conj_PRECISION(clover[19])*threadPhi[1];
  eta[5] += cu_conj_PRECISION(clover[20])*threadPhi[1];
  eta[3] += cu_conj_PRECISION(clover[21])*threadPhi[2];
  eta[4] += cu_conj_PRECISION(clover[22])*threadPhi[2];
  eta[5] += cu_conj_PRECISION(clover[23])*threadPhi[2];
  eta[4] += cu_conj_PRECISION(clover[24])*threadPhi[3];
  eta[5] += cu_conj_PRECISION(clover[25])*threadPhi[3];
  eta[5] += cu_conj_PRECISION(clover[26])*threadPhi[4];
  // spin 2 and 3, row major
  eta[ 6] += clover[27]*threadPhi[ 7];
  eta[ 6] += clover[28]*threadPhi[ 8];
  eta[ 6] += clover[29]*threadPhi[ 9];
  eta[ 6] += clover[30]*threadPhi[10];
  eta[ 6] += clover[31]*threadPhi[11];
  eta[ 7] += clover[32]*threadPhi[ 8];
  eta[ 7] += clover[33]*threadPhi[ 9];
  eta[ 7] += clover[34]*threadPhi[10];
  eta[ 7] += clover[35]*threadPhi[11];
  eta[ 8] += clover[36]*threadPhi[ 9];
  eta[ 8] += clover[37]*threadPhi[10];
  eta[ 8] += clover[38]*threadPhi[11];
  eta[ 9] += clover[39]*threadPhi[10];
  eta[ 9] += clover[40]*threadPhi[11];
  eta[10] += clover[41]*threadPhi[11];
  eta[ 7] += cu_conj_PRECISION(clover[27])*threadPhi[ 6];
  eta[ 8] += cu_conj_PRECISION(clover[28])*threadPhi[ 6];
  eta[ 9] += cu_conj_PRECISION(clover[29])*threadPhi[ 6];
  eta[10] += cu_conj_PRECISION(clover[30])*threadPhi[ 6];
  eta[11] += cu_conj_PRECISION(clover[31])*threadPhi[ 6];
  eta[ 8] += cu_conj_PRECISION(clover[32])*threadPhi[ 7];
  eta[ 9] += cu_conj_PRECISION(clover[33])*threadPhi[ 7];
  eta[10] += cu_conj_PRECISION(clover[34])*threadPhi[ 7];
  eta[11] += cu_conj_PRECISION(clover[35])*threadPhi[ 7];
  eta[ 9] += cu_conj_PRECISION(clover[36])*threadPhi[ 8];
  eta[10] += cu_conj_PRECISION(clover[37])*threadPhi[ 8];
  eta[11] += cu_conj_PRECISION(clover[38])*threadPhi[ 8];
  eta[10] += cu_conj_PRECISION(clover[39])*threadPhi[ 9];
  eta[11] += cu_conj_PRECISION(clover[40])*threadPhi[ 9];
  eta[11] += cu_conj_PRECISION(clover[41])*threadPhi[10];
}

__global__ void cuda_prp_T_coalesced_PRECISION(cu_cmplx_PRECISION * prpT, cu_cmplx_PRECISION const * phi,
                                size_t num_sites) {
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  constexpr uint phiChunkSize = 12;
  // set phi address to first element handled by block
  phi += phiChunkSize * blockDim.x * blockIdx.x;

  // copy the array elements handled in this block to shared memory
  __shared__ cu_cmplx_PRECISION blockPhi[phiChunkSize * diracCommonBlockSize];
  copy_chunks_to_consecutive_as_block(blockPhi, phi, 12, 0, blockDim.x);

  if (idx >= num_sites){
    // there is no more site for this index
    return;
  }

  // set threadPhi address to first element handled by thread
  cu_cmplx_PRECISION * const threadPhi = blockPhi + phiChunkSize * threadIdx.x;
  prpT += 6*idx;
  prpT[0] = threadPhi[0] -GAMMA_T_SPIN0_VAL*threadPhi[3*GAMMA_T_SPIN0_CO];
  prpT[1] = threadPhi[1] -GAMMA_T_SPIN0_VAL*threadPhi[3*GAMMA_T_SPIN0_CO+1];
  prpT[2] = threadPhi[2] -GAMMA_T_SPIN0_VAL*threadPhi[3*GAMMA_T_SPIN0_CO+2];
  prpT[3] = threadPhi[3] -GAMMA_T_SPIN1_VAL*threadPhi[3*GAMMA_T_SPIN1_CO];
  prpT[4] = threadPhi[4] -GAMMA_T_SPIN1_VAL*threadPhi[3*GAMMA_T_SPIN1_CO+1];
  prpT[5] = threadPhi[5] -GAMMA_T_SPIN1_VAL*threadPhi[3*GAMMA_T_SPIN1_CO+2];
}
