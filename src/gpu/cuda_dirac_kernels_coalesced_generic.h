/**
 * \file cuda_dirac_kernels_coalesced_generic.h
 *
 * \brief CUDA kernels to perform calculations related to application of the Wilson-Dirac operator.
 */

#ifndef CUDA_DIRAC_KERNELS_COALESCED_PRECISION_H
#define CUDA_DIRAC_KERNELS_COALESCED_PRECISION_H

#include "clifford.h"

constexpr unsigned int diracCommonBlockSize = 64;

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
__global__ void cuda_site_clover_coalesced_PRECISION(cuda_vector_PRECISION eta,
                                                     cu_cmplx_PRECISION const* phi,
                                                     cu_cmplx_PRECISION const* clover,
                                                     size_t num_sites);

__global__ void cuda_prp_T_coalesced_PRECISION(cu_cmplx_PRECISION* prpT,
                                               cu_cmplx_PRECISION const* phi, size_t num_sites);

#endif  // CUDA_DIRAC_KERNELS_COALESCED_PRECISION_H
