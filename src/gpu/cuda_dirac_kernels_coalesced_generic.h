/**
 * \file cuda_dirac_kernels_coalesced_generic.h
 *
 * \brief CUDA kernels to perform calculations related to application of the Wilson-Dirac operator.
 */

#ifndef CUDA_DIRAC_KERNELS_COALESCED_PRECISION_H
#define CUDA_DIRAC_KERNELS_COALESCED_PRECISION_H

#include "cuda_vectors_PRECISION.h"

__global__ void cuda_prp_T_coalesced_PRECISION(cu_cmplx_PRECISION* prpT,
                                               cu_cmplx_PRECISION const* phi, size_t num_sites);

#endif  // CUDA_DIRAC_KERNELS_COALESCED_PRECISION_H
