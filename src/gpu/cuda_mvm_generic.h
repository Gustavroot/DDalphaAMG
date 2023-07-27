/**
 * \file cuda_mvm_generic.h
 * \brief 3x3 matrix vector multiplications.
 */

#ifndef CUDA_MVM_PRECISION_H
#define CUDA_MVM_PRECISION_H

#include "cuda_complex.h"
#include "cuda_componentwise.h"

/**
 * \brief Calculate the 3x3 matrix vector product y = M x.
 *
 * \param[out]  y   The vector y that the result will be written to. Must be a
 *                  pointer to 3 consecutive complex values.
 * \param[in]   M   The matrix M. Must be a pointer to 9 consecutive complex
 *                  values. Must be in row-major ordering.
 * \param[in]   x   The vector x. Must be a pointer to 3 consecutive complex
 *                  values.
 */
__device__ void cuda_mvm_PRECISION(cu_cmplx_PRECISION *y,
                                   cu_cmplx_PRECISION const *M,
                                   cu_cmplx_PRECISION const *x);

/**
 * \brief Calculate the 3x3 matrix vector product y = M' x.
 *
 * Where M' is the hermitian transpose of M.
 *
 * \param[out]  y   The vector y that the result will be written to. Must be a
 *                  pointer to 3 consecutive complex values.
 * \param[in]   M   The matrix M. Must be a pointer to 9 consecutive complex
 *                  values. Must be in row-major ordering.
 * \param[in]   x   The vector x. Must be a pointer to 3 consecutive complex
 *                  values.
 */
__device__ void cuda_mvmh_PRECISION(cu_cmplx_PRECISION *y,
                                    cu_cmplx_PRECISION const *M,
                                    cu_cmplx_PRECISION const *x);

/**
 * \brief Calculate the 3x3 matrix vector product y = M x.
 *
 * \param[out]  y   The vector y that the result will be written to. Must be a
 *                  ComponentAccess wrapper around a componentwise array with
 *                  at least 3 components.
 * \param[in]   M   The matrix M. Must be a ComponentAccess wrapper around a
 *                  componentwise array with at least 9 components.
 * \param[in]   x   The vector x. Must be a pointer to 3 consecutive complex
 *                  values.
 */
__device__ void cuda_mvm_componentwise_PRECISION(
    ComponentAccess<cu_cmplx_PRECISION> y,
    ComponentAccess<cu_cmplx_PRECISION const> M, cu_cmplx_PRECISION const *x);

/**
 * \brief Calculate the 3x3 matrix vector product y = M' x.
 *
 * Where M' is the hermitian transpose of M.
 *
 * \param[out]  y   The vector y that the result will be written to. Must be a
 *                  pointer to 3 consecutive complex values.
 * \param[in]   M   The matrix M. Must be a ComponentAccess wrapper around a
 *                  componentwise array with at least 9 components.
 * \param[in]   x   The vector x. Must be a ComponentAccess wrapper around a
 *                  componentwise array with at least 3 components.
 */
__device__ void cuda_mvmh_componentwise_PRECISION(
    cu_cmplx_PRECISION *y, ComponentAccess<cu_cmplx_PRECISION const> M,
    ComponentAccess<cu_cmplx_PRECISION const> x);

#endif  // CUDA_MVM_PRECISION_H
