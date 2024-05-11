/**
 * \file cuda_dirac_kernels_componentwise_generic.h
 *
 * \brief CUDA kernels to perform calculations related to application of the Wilson-Dirac operator.
 */

#ifndef CUDA_DIRAC_KERNELS_COMPONENTWISE_PRECISION_H
#define CUDA_DIRAC_KERNELS_COMPONENTWISE_PRECISION_H

#include "cuda_vectors_PRECISION.h"
#include "global_enums.h"

// Better to have this known in advance, so we can predetermine shared
// memory requirements.
constexpr size_t diracCommonBlockSize = 128;

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
__global__ void cuda_site_clover_componentwise_PRECISION(cuda_vector_PRECISION eta,
                                                         cu_cmplx_PRECISION const* phi,
                                                         cu_cmplx_PRECISION const* clover,
                                                         size_t num_sites);

__global__ void cuda_site_diag_ee_componentwise_PRECISION(cuda_vector_PRECISION eta,
                                                          cu_cmplx_PRECISION const* phi,
                                                          cu_cmplx_PRECISION const* clover,
                                                          size_t num_sites);
__global__ void cuda_site_diag_oo_inv_componentwise_PRECISION(cuda_vector_PRECISION eta,
                                                              cu_cmplx_PRECISION const* phi,
                                                              cu_cmplx_PRECISION const* clover,
                                                              size_t num_sites);

__global__ void cuda_prp_T_componentwise_PRECISION(cu_cmplx_PRECISION* prpT,
                                                   cu_cmplx_PRECISION const* phi, size_t num_sites);

__global__ void cuda_prn_T_componentwise_PRECISION(cu_cmplx_PRECISION* prnT,
                                                   cu_cmplx_PRECISION const* phi, size_t num_sites);

__global__ void cuda_prp_Z_componentwise_PRECISION(cu_cmplx_PRECISION* prpZ,
                                                   cu_cmplx_PRECISION const* phi, size_t num_sites);

__global__ void cuda_prn_Z_componentwise_PRECISION(cu_cmplx_PRECISION* prnZ,
                                                   cu_cmplx_PRECISION const* phi, size_t num_sites);

__global__ void cuda_prp_Y_componentwise_PRECISION(cu_cmplx_PRECISION* prpY,
                                                   cu_cmplx_PRECISION const* phi, size_t num_sites);

__global__ void cuda_prn_Y_componentwise_PRECISION(cu_cmplx_PRECISION* prnY,
                                                   cu_cmplx_PRECISION const* phi, size_t num_sites);

__global__ void cuda_prp_X_componentwise_PRECISION(cu_cmplx_PRECISION* prpX,
                                                   cu_cmplx_PRECISION const* phi, size_t num_sites);

__global__ void cuda_prn_X_componentwise_PRECISION(cu_cmplx_PRECISION* prnX,
                                                   cu_cmplx_PRECISION const* phi, size_t num_sites);

/**
 * \brief Applies the matrix vector products as required for the projection in negative direction.
 *
 * This kernel must be invoked with double as many (total across blocks) threads as there are
 * inner lattice sites. This is done in order to expose full parralelism for the 3x3 matrix vector
 * multiplications.
 *
 * \param[out]  prp_buf   The projection buffer to which the result is written.
 * \param[in]   D         The Wilson-Dirac matrix in the form used throughout this project.
 * \param[in]   pbuf      The vector prepared by a previous call to cuda_prn_*_PRECISION.
 * \param[in]   neighbors The neighbor table containing the neighbors in all 4 directions.
 *                        See also operator_PRECISION_struct::neighbor_table.
 * \param[in]   dim       The direction in which the projection is performed.
 * \param[in]   num_sites The number of lattice sites for which the projection will be performed.
 *                        Note that even though this is the number of inner lattice site, sites
 *                        outside of that range will be written to in prp_buf if they are neighbors
 *                        to an inner lattice site.
 */
__global__ void cuda_prn_mvmh_componentwise_PRECISION(cu_cmplx_PRECISION* prp_buf,
                                                      cu_cmplx_PRECISION const* D,
                                                      cu_cmplx_PRECISION const* pbuf,
                                                      int const* neighbors, LatticeAxis dim,
                                                      size_t num_sites);

/**
 * \brief Applies the matrix vector products as required for the projection in positive direction.
 *
 * This kernel must be invoked with double as many (total across blocks) threads as there are
 * inner lattice sites. This is done in order to expose full parralelism for the 3x3 matrix vector
 * multiplications.
 *
 * \param[in]   pbuf      The result vector prepared for a subsequent call to
 *                        cuda_pbp_su3_*_PRECISION.
 * \param[in]   D         The Wilson-Dirac matrix in the form used throughout this project.
 * \param[out]  prn_buf   The negative projection vector as previously calculated and communicated.
 * \param[in]   neighbors The neighbor table containing the neighbors in all 4 directions.
 *                        See also operator_PRECISION_struct::neighbor_table.
 * \param[in]   dim       The direction in which the projection is performed.
 * \param[in]   num_sites The number of lattice sites for which the projection will be performed.
 *                        Note that even though this is the number of inner lattice site, sites
 *                        outside of that range will be read in prn_buf if they are neighbors
 *                        to an inner lattice site.
 */
__global__ void cuda_pbp_su3_mvm_componentwise_PRECISION(cu_cmplx_PRECISION* pbuf,
                                                         cu_cmplx_PRECISION const* D,
                                                         cu_cmplx_PRECISION const* prn_buf,
                                                         int const* neighbors, LatticeAxis dim,
                                                         size_t num_sites);

__global__ void cuda_pbp_su3_T_componentwise_PRECISION(cu_cmplx_PRECISION* eta,
                                                       cu_cmplx_PRECISION const* pbuf,
                                                       size_t num_sites);

__global__ void cuda_pbp_su3_Z_componentwise_PRECISION(cu_cmplx_PRECISION* eta,
                                                       cu_cmplx_PRECISION const* pbuf,
                                                       size_t num_sites);

__global__ void cuda_pbp_su3_Y_componentwise_PRECISION(cu_cmplx_PRECISION* eta,
                                                       cu_cmplx_PRECISION const* pbuf,
                                                       size_t num_sites);

__global__ void cuda_pbp_su3_X_componentwise_PRECISION(cu_cmplx_PRECISION* eta,
                                                       cu_cmplx_PRECISION const* pbuf,
                                                       size_t num_sites);

__global__ void cuda_pbn_su3_T_componentwise_PRECISION(cu_cmplx_PRECISION* eta,
                                                       cu_cmplx_PRECISION const* prpT,
                                                       size_t num_sites);

__global__ void cuda_pbn_su3_Z_componentwise_PRECISION(cu_cmplx_PRECISION* eta,
                                                       cu_cmplx_PRECISION const* prpZ,
                                                       size_t num_sites);

__global__ void cuda_pbn_su3_Y_componentwise_PRECISION(cu_cmplx_PRECISION* eta,
                                                       cu_cmplx_PRECISION const* prpY,
                                                       size_t num_sites);

__global__ void cuda_pbn_su3_X_componentwise_PRECISION(cu_cmplx_PRECISION* eta,
                                                       cu_cmplx_PRECISION const* prpX,
                                                       size_t num_sites);

#endif  // CUDA_DIRAC_KERNELS_COMPONENTWISE_PRECISION_H
