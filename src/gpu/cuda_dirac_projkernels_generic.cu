#include "cuda_complex.h"
#include "cuda_complex_operators_PRECISION.h"
#include "cuda_vectors_PRECISION.h"
#include "clifford.h"


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

  eta[ 0] = CU_CMPLX_PRECISION_ZERO;
  eta[ 1] = CU_CMPLX_PRECISION_ZERO;
  eta[ 2] = CU_CMPLX_PRECISION_ZERO;
  eta[ 3] = CU_CMPLX_PRECISION_ZERO;
  eta[ 4] = CU_CMPLX_PRECISION_ZERO;
  eta[ 5] = CU_CMPLX_PRECISION_ZERO;
  eta[ 6] = cu_cmul_PRECISION(clover[ 6], phi[ 6]);
  eta[ 7] = cu_cmul_PRECISION(clover[ 7], phi[ 7]);
  eta[ 8] = cu_cmul_PRECISION(clover[ 8], phi[ 8]);
  eta[ 9] = cu_cmul_PRECISION(clover[ 9], phi[ 9]);
  eta[10] = cu_cmul_PRECISION(clover[10], phi[10]);
  eta[11] = cu_cmul_PRECISION(clover[11], phi[11]);
  // spin 2 and 3
  eta[ 6] = cu_cadd_PRECISION(eta[ 6], cu_cmul_PRECISION(clover[28], phi[ 8]));
  eta[ 6] = cu_cadd_PRECISION(eta[ 6], cu_cmul_PRECISION(clover[27], phi[ 7]));
  eta[ 6] = cu_cadd_PRECISION(eta[ 6], cu_cmul_PRECISION(clover[29], phi[ 9]));
  eta[ 6] = cu_cadd_PRECISION(eta[ 6], cu_cmul_PRECISION(clover[30], phi[10]));
  eta[ 6] = cu_cadd_PRECISION(eta[ 6], cu_cmul_PRECISION(clover[31], phi[11]));
  eta[ 7] = cu_cadd_PRECISION(eta[ 7], cu_cmul_PRECISION(clover[32], phi[ 8]));
  eta[ 7] = cu_cadd_PRECISION(eta[ 7], cu_cmul_PRECISION(clover[33], phi[ 9]));
  eta[ 7] = cu_cadd_PRECISION(eta[ 7], cu_cmul_PRECISION(clover[34], phi[10]));
  eta[ 7] = cu_cadd_PRECISION(eta[ 7], cu_cmul_PRECISION(clover[35], phi[11]));
  eta[ 8] = cu_cadd_PRECISION(eta[ 8], cu_cmul_PRECISION(clover[36], phi[ 9]));
  eta[ 8] = cu_cadd_PRECISION(eta[ 8], cu_cmul_PRECISION(clover[37], phi[10]));
  eta[ 8] = cu_cadd_PRECISION(eta[ 8], cu_cmul_PRECISION(clover[38], phi[11]));
  eta[ 9] = cu_cadd_PRECISION(eta[ 9], cu_cmul_PRECISION(clover[39], phi[10]));
  eta[ 9] = cu_cadd_PRECISION(eta[ 9], cu_cmul_PRECISION(clover[40], phi[11]));
  eta[10] = cu_cadd_PRECISION(eta[10], cu_cmul_PRECISION(clover[41], phi[11]));
  eta[ 7] = cu_cadd_PRECISION(eta[ 7], cu_cmul_PRECISION(cu_conj_PRECISION(clover[27]), phi[ 6]));
  eta[ 8] = cu_cadd_PRECISION(eta[ 8], cu_cmul_PRECISION(cu_conj_PRECISION(clover[28]), phi[ 6]));
  eta[ 9] = cu_cadd_PRECISION(eta[ 9], cu_cmul_PRECISION(cu_conj_PRECISION(clover[29]), phi[ 6]));
  eta[10] = cu_cadd_PRECISION(eta[10], cu_cmul_PRECISION(cu_conj_PRECISION(clover[30]), phi[ 6]));
  eta[11] = cu_cadd_PRECISION(eta[11], cu_cmul_PRECISION(cu_conj_PRECISION(clover[31]), phi[ 6]));
  eta[ 8] = cu_cadd_PRECISION(eta[ 8], cu_cmul_PRECISION(cu_conj_PRECISION(clover[32]), phi[ 7]));
  eta[ 9] = cu_cadd_PRECISION(eta[ 9], cu_cmul_PRECISION(cu_conj_PRECISION(clover[33]), phi[ 7]));
  eta[10] = cu_cadd_PRECISION(eta[10], cu_cmul_PRECISION(cu_conj_PRECISION(clover[34]), phi[ 7]));
  eta[11] = cu_cadd_PRECISION(eta[11], cu_cmul_PRECISION(cu_conj_PRECISION(clover[35]), phi[ 7]));
  eta[ 9] = cu_cadd_PRECISION(eta[ 9], cu_cmul_PRECISION(cu_conj_PRECISION(clover[36]), phi[ 8]));
  eta[10] = cu_cadd_PRECISION(eta[10], cu_cmul_PRECISION(cu_conj_PRECISION(clover[37]), phi[ 8]));
  eta[11] = cu_cadd_PRECISION(eta[11], cu_cmul_PRECISION(cu_conj_PRECISION(clover[38]), phi[ 8]));
  eta[10] = cu_cadd_PRECISION(eta[10], cu_cmul_PRECISION(cu_conj_PRECISION(clover[39]), phi[ 9]));
  eta[11] = cu_cadd_PRECISION(eta[11], cu_cmul_PRECISION(cu_conj_PRECISION(clover[40]), phi[ 9]));
  eta[11] = cu_cadd_PRECISION(eta[11], cu_cmul_PRECISION(cu_conj_PRECISION(clover[41]), phi[10]));
}

__global__ void prp_T_PRECISION(cu_cmplx_PRECISION * prpT, cu_cmplx_PRECISION const * phi,
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