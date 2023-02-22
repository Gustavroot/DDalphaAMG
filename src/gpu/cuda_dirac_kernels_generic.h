/**
 * \file cuda_dirac_kernels_generic.h
 * 
 * \brief CUDA kernels to perform calculations related to application of the Wilson-Dirac operator.
 */

#ifndef CUDA_DIRAC_KERNELS_PRECISION_H
#define CUDA_DIRAC_KERNELS_PRECISION_H

#include "clifford.h"

/**
 * \brief Apply the Clover term per lattice site.
 * 
 * \param[out]  eta         Set to eta = C * phi.
 *                          I.e. is the result of applying the Clover term on phi.
 * \param[in]   phi         Quark field phi.
 * \param[in]   clover      The configuration of the Clover term.
 * \param[in]   num_sites   The number of lattice sites to process. A call which starts more threads
 *                          than needed for the number of lattice sites will have the remaining
 *                          threads idle/return.
 */
__global__ void cuda_site_clover_PRECISION(cuda_vector_PRECISION eta, cu_cmplx_PRECISION const* phi,
                                           cu_cmplx_PRECISION const* clover, size_t num_sites);

__global__ void cuda_prp_T_PRECISION(cu_cmplx_PRECISION* prpT, cu_cmplx_PRECISION const* phi,
                                     size_t num_sites);

__global__ void cuda_prn_T_PRECISION(cu_cmplx_PRECISION* prnT, cu_cmplx_PRECISION const* phi,
                                     size_t num_sites);

__global__ void cuda_prp_Z_PRECISION(cu_cmplx_PRECISION* prpZ, cu_cmplx_PRECISION const* phi,
                                     size_t num_sites);

__global__ void cuda_prn_Z_PRECISION(cu_cmplx_PRECISION* prnZ, cu_cmplx_PRECISION const* phi,
                                     size_t num_sites);

__global__ void cuda_prp_Y_PRECISION(cu_cmplx_PRECISION* prpY, cu_cmplx_PRECISION const* phi,
                                     size_t num_sites);

__global__ void cuda_prn_Y_PRECISION(cu_cmplx_PRECISION* prnY, cu_cmplx_PRECISION const* phi,
                                     size_t num_sites);

__global__ void cuda_prp_X_PRECISION(cu_cmplx_PRECISION* prpX, cu_cmplx_PRECISION const* phi,
                                     size_t num_sites);

__global__ void cuda_prn_X_PRECISION(cu_cmplx_PRECISION* prnX, cu_cmplx_PRECISION const* phi,
                                     size_t num_sites);
                                     
__global__ void cuda_prn_mvmh_PRECISION(cu_cmplx_PRECISION* prp_buf, cu_cmplx_PRECISION const* D,
                                        cu_cmplx_PRECISION const * pbuf, int const * neighbors,
                                        LatticeAxis dim, size_t num_sites);

__global__ void cuda_pbp_su3_mvm_PRECISION(cu_cmplx_PRECISION* pbuf, cu_cmplx_PRECISION const * D,
                                           cu_cmplx_PRECISION const * prn_buf, int const *neighbors,
                                           LatticeAxis dim, size_t num_sites);

__global__ void cuda_pbp_su3_T_PRECISION(cu_cmplx_PRECISION* eta, cu_cmplx_PRECISION const * pbuf,
                                         size_t num_sites);

__global__ void cuda_pbp_su3_Z_PRECISION(cu_cmplx_PRECISION* eta, cu_cmplx_PRECISION const * pbuf,
                                         size_t num_sites);

__global__ void cuda_pbp_su3_Y_PRECISION(cu_cmplx_PRECISION* eta, cu_cmplx_PRECISION const * pbuf,
                                         size_t num_sites);

__global__ void cuda_pbp_su3_X_PRECISION(cu_cmplx_PRECISION* eta, cu_cmplx_PRECISION const * pbuf,
                                         size_t num_sites);

__global__ void cuda_pbn_su3_T_PRECISION(cu_cmplx_PRECISION* eta, cu_cmplx_PRECISION const * prpT,
                                         size_t num_sites);

__global__ void cuda_pbn_su3_Z_PRECISION(cu_cmplx_PRECISION* eta, cu_cmplx_PRECISION const * prpZ,
                                         size_t num_sites);

__global__ void cuda_pbn_su3_Y_PRECISION(cu_cmplx_PRECISION* eta, cu_cmplx_PRECISION const * prpY,
                                         size_t num_sites);

__global__ void cuda_pbn_su3_X_PRECISION(cu_cmplx_PRECISION* eta, cu_cmplx_PRECISION const * prpX,
                                         size_t num_sites);

#endif  // CUDA_DIRAC_KERNELS_PRECISION_H