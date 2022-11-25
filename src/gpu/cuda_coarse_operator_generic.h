#ifdef CUDA_OPT
#ifndef COARSE_OPERATOR_PRECISION_HEADER_CUDA
  #define COARSE_OPERATOR_PRECISION_HEADER_CUDA
#include "cuda_complex.h"

  __global__ void coarse_self_couplings_PRECISION_CUDA_kernel( cuda_vector_PRECISION eta, cuda_vector_PRECISION phi, cuda_config_PRECISION clover, int num_lattice_site_var );

  extern void coarse_self_couplings_PRECISION_CUDA( cuda_vector_PRECISION eta, cuda_vector_PRECISION phi, cuda_config_PRECISION clover,
                                                      int length, level_struct *l, struct Thread *threading );

  extern void apply_coarse_operator_PRECISION_CUDA( cuda_vector_PRECISION eta_gpu, cuda_vector_PRECISION phi_gpu,
                                                    operator_PRECISION_struct *op, level_struct *l, struct Thread *threading );

  __device__ static __forceinline__ void mvp_PRECISION_CUDA( cuda_vector_PRECISION eta, cuda_config_PRECISION clover, cuda_vector_PRECISION phi, int length, int i )
  {
    int j, m, mag_nr;

    mag_nr = (i+1)*(i+2)/2-1;
    for (j=0; j<length; j++) {
      if (i>j) {
        // conjugation
        m = mag_nr-(i-j);
        eta[i] = cu_cadd_PRECISION( eta[i], cu_cmul_PRECISION( cu_conj_PRECISION(clover[m]),phi[j] ) );
      } else {
        // no conjugation
        m = mag_nr-(j-i);
        eta[i] = cu_cadd_PRECISION( eta[i], cu_cmul_PRECISION( clover[m],phi[j] ) );
      }
    }
  }

  __device__ static __forceinline__ void nmvh_PRECISION_CUDA( cuda_vector_PRECISION eta, cuda_config_PRECISION clover, cuda_vector_PRECISION phi, int length, int i )
  {
    int j,k;

    for ( j=0; j<length; j++) {
      // accessing the matrix columnwise
      k = i*length + j;
      eta[i] = cu_csub_PRECISION( eta[i], cu_cmul_PRECISION( cu_conj_PRECISION(clover[k]),phi[j] ) );
    }
  }

  __device__ static __forceinline__ void mv_PRECISION_CUDA( cuda_vector_PRECISION eta, cuda_config_PRECISION clover, cuda_vector_PRECISION phi, int length, int i )
  {
    int j,k;

    for ( j=0; j<length; j++) {
      // accessing the matrix columnwise
      k = j*length + i;
      eta[i] = cu_cadd_PRECISION( eta[i], cu_cmul_PRECISION( clover[k],phi[j] ) );
    }
  }


#endif
#endif
