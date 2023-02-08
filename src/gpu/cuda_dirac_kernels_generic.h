#ifndef CUDA_DIRAC_KERNELS_PRECISION_H
#define CUDA_DIRAC_KERNELS_PRECISION_H

__global__ void cuda_site_clover_PRECISION(cuda_vector_PRECISION eta, cuda_vector_PRECISION phi,
                                           cuda_config_PRECISION clover, size_t num_sites);

// 1 - gamma_T
__global__ void cuda_prp_T_PRECISION(cu_cmplx_PRECISION* prpT, cu_cmplx_PRECISION const* phi,
                                     size_t num_sites);

__global__ void cuda_prp_Z_PRECISION(cu_cmplx_PRECISION* prpZ, cu_cmplx_PRECISION const* phi,
                                     size_t num_sites);

__global__ void cuda_prp_Y_PRECISION(cu_cmplx_PRECISION* prpY, cu_cmplx_PRECISION const* phi,
                                     size_t num_sites);

__global__ void cuda_prp_X_PRECISION(cu_cmplx_PRECISION* prpX, cu_cmplx_PRECISION const* phi,
                                     size_t num_sites);

#endif  // CUDA_DIRAC_KERNELS_PRECISION_H