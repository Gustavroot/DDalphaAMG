#ifdef CUDA_OPT
#ifndef COARSE_OPERATOR_PRECISION_HEADER_CUDA
  #define COARSE_OPERATOR_PRECISION_HEADER_CUDA

  __global__ void coarse_self_couplings_PRECISION_CUDA_kernel( cuda_vector_PRECISION eta, cuda_vector_PRECISION phi, cuda_config_PRECISION clover, int num_lattice_site_var );

  extern void coarse_self_couplings_PRECISION_CUDA( cuda_vector_PRECISION eta, cuda_vector_PRECISION phi, cuda_config_PRECISION clover,
                                                      int length, level_struct *l, struct Thread *threading );

  extern void apply_coarse_operator_PRECISION_CUDA( cuda_vector_PRECISION eta_gpu, cuda_vector_PRECISION phi_gpu,
                                                    operator_PRECISION_struct *op, level_struct *l, struct Thread *threading );

#endif
#endif
