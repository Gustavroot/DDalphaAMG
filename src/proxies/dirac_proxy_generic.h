/**
 * \file dirac_proxy_generic.h
 * \brief Mediates application of Dirac operator between CPU and GPU compilation options.
 *
 * The purpose of this file is that other parts of the code may stay ignorant to the implementation
 * details of the Wilson-Dirac operator. A GPU or CPU implementation of the operator will be used
 * depending on compilation options.
 */

#ifndef DIRAC_PROXY_PRECISION_H
#define DIRAC_PROXY_PRECISION_H

#include "complex_types_PRECISION.h"
#include "level_struct.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Applies the Wilson-Dirac operator including clover term.
 *
 * Updates eta with eta + D_W eta. For more information about the Wilson-Dirac operator and clover
 * term refer to relevant research papers (Matthias Rottmann's 2014 thesis).
 * The implementation is delegated to a CPU function or GPU function depending on compilation
 * option CUDA_OPT.
 *
 * \see d_plus_clover_PRECISION_cpu
 * \see cuda_d_plus_clover_PRECISION
 *
 * \param[out]  eta         Will be updated to eta + D_W eta.
 * \param[in]   phi         Spinor vector.
 * \param[in]   op          Data structure with various operator-specific information.
 * \param[in]   l           Data structure with various level-specific information.
 * \param[in]   threading   Thread struct containing information about thread configuration.
 */
void d_plus_clover_PRECISION(vector_PRECISION eta, vector_PRECISION phi,
                             operator_PRECISION_struct *op, level_struct *l,
                             struct Thread *threading);

void apply_schur_complement_PRECISION(vector_PRECISION out, vector_PRECISION in,
                                      operator_PRECISION_struct *op, level_struct *l,
                                      struct Thread *threading);

#ifdef __cplusplus
}
#endif

#endif  // DIRAC_PROXY_PRECISION_H
