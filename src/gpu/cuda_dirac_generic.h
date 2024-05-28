/** \file cuda_dirac_generic.h
 *  \brief CUDA implementations of the Wilson-Dirac operator functions.
 *  \see dirac_proxy_generic.h
 */
#ifndef CUDA_DIRAC_PRECISION_HEADER_CUDA
#define CUDA_DIRAC_PRECISION_HEADER_CUDA

#include "cuda_schwarz_PRECISION.h"
#include "cuda_vectors_PRECISION.h"

// device functions

__global__ void cuda_block_d_plus_clover_PRECISION_6threads_naive(
    cu_cmplx_PRECISION *out, cu_cmplx_PRECISION *in,
    schwarz_PRECISION_struct_on_gpu *s, int thread_id, double csw,
    int nr_threads_per_DD_block, int *DD_blocks_to_compute,
    int num_latt_site_var, block_struct *block, int ext_dir);

// host functions

extern void cuda_block_d_plus_clover_PRECISION(
    cuda_vector_PRECISION eta, cuda_vector_PRECISION phi,
    int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s, level_struct *l,
    struct Thread *threading, int stream_id, cudaStream_t *streams, int color,
    int *DD_blocks_to_compute_gpu, int *DD_blocks_to_compute_cpu);

#ifdef __cplusplus
/** \brief Apply Wilson-Dirac operator with GPU acceleration.
 *
 *  \see d_plus_clover_PRECISION
 */
void cuda_d_plus_clover_PRECISION(cuda_vector_PRECISION eta,
                                  cuda_vector_PRECISION phi,
                                  operator_PRECISION_struct *op,
                                  level_struct *l, struct Thread *threading);

/** \brief Earlier version of cuda_d_plus_clover_PRECISION with unoptimized
 *         vector access.
 *
 *  Can be changed in cuda_d_plus_clover_PRECISION_vectorwrapper if results need
 *  to be reproduced. Can be safely removed.
 *
 *  \see d_plus_clover_PRECISION
 */
void cuda_d_plus_clover_PRECISION_awarempi(cuda_vector_PRECISION eta,
                                           cuda_vector_PRECISION phi,
                                           operator_PRECISION_struct *op,
                                           level_struct *l,
                                           struct Thread *threading);

/** \brief Earlier version of cuda_d_plus_clover_PRECISION with unoptimized MPI
 *         communication.
 *
 *  Can be changed in cuda_d_plus_clover_PRECISION_vectorwrapper if results need
 *  to be reproduced. Can be safely removed.
 *
 *  \see d_plus_clover_PRECISION
 */
void cuda_d_plus_clover_PRECISION_naive(cuda_vector_PRECISION eta,
                                        cuda_vector_PRECISION phi,
                                        operator_PRECISION_struct *op,
                                        level_struct *l,
                                        struct Thread *threading);

#endif  // __cplusplus

/** \brief Copy vectors eta and phi from CPU to GPU and back and call actual
 *         operator.
 *
 *  As d_plus_clover_PRECISION was initially a CPU function, eta and phi are
 *  pointers to CPU arrays. For the CUDA version the vectors operations on eta
 *  and phi are preformed on the GPU, so there is a need to copy these vectors.
 *
 *  Does also perofrm reordering of vectors eta and phi from and to
 *  componentwise format.
 *
 *  In the future this function may become obsolete if FGMRES (and possibly
 *  other methods) support handling GPU vectors x and w instead, which would
 *  improve performance.
 *
 *  \param[out] eta     CPU vector that the GPU result will be copied to.
 *  \param[in]  phi     Vector that will be copied to the GPU.
 *
 *  \see d_plus_clover_PRECISION
 */
extern void cuda_d_plus_clover_PRECISION_vectorwrapper(
    vector_PRECISION eta, complex_PRECISION const *phi,
    operator_PRECISION_struct *op, level_struct *l, struct Thread *threading);

extern void _cuda_clover_componentwise_PRECISION(cuda_vector_PRECISION eta,
                                                 cuda_vector_PRECISION phi,
                                                 cuda_config_PRECISION clover, int num_sites,
                                                 level_struct *l);

extern void cuda_diag_ee_componentwise_PRECISION( cuda_vector_PRECISION eta,
                                                  cuda_vector_PRECISION phi,
                                                  cuda_config_PRECISION clover,
                                                  int num_sites, level_struct *l );

extern void cuda_diag_oo_inv_componentwise_PRECISION( cuda_vector_PRECISION eta,
                                                      cuda_vector_PRECISION phi,
                                                      cuda_config_PRECISION clover,
                                                      int num_sites, level_struct *l );

extern __constant__ cu_cmplx_PRECISION gamma_info_vals_PRECISION[16];
extern __constant__ int gamma_info_coo_PRECISION[16];

#endif
