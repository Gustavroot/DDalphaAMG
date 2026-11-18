/**
 * \file interpolation_proxy_generic.h
 * \brief Mediates the interpolation and restriction operators between CPU and GPU compilation options.
 *
 * The rest of the code calls the functions declared here and stays ignorant of where the
 * interpolation operator P is applied: with CUDA_OPT the GPU implementation is used (the
 * *_vectorwrapper functions in gpu/cuda_interpolation_generic.cu, which take host vectors),
 * otherwise the CPU implementation (*_cpu functions in interpolation_generic.c or
 * sse_interpolation_generic.c, depending on the memory layout of P).
 */

#ifndef INTERPOLATION_PROXY_PRECISION_H
#define INTERPOLATION_PROXY_PRECISION_H

#include "complex_types_PRECISION.h"
#include "level_struct.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Thread;

/** \brief Allocates the interpolation data of level l (and initialises its GPU part). */
void interpolation_PRECISION_alloc( level_struct *l );

/** \brief Frees the interpolation data of level l (CPU and GPU). */
void interpolation_PRECISION_free( level_struct *l );

/**
 * \brief To be called (by all threads) once the interpolation operator l->is_PRECISION.op is final.
 *
 * CPU: does nothing. GPU: copies the operator to the device (allocating the device buffers at the
 * first call).
 */
void interpolation_PRECISION_op_to_device( level_struct *l, struct Thread *threading );

/** \brief phi += P phi_c (phi on level l, phi_c on level l->next_level). */
void interpolate_PRECISION( vector_PRECISION phi, vector_PRECISION phi_c, level_struct *l, struct Thread *threading );

/** \brief phi = P phi_c (phi on level l, phi_c on level l->next_level). */
void interpolate3_PRECISION( vector_PRECISION phi, vector_PRECISION phi_c, level_struct *l, struct Thread *threading );

/** \brief phi_c = P^dagger phi (phi on level l, phi_c on level l->next_level). */
void restrict_PRECISION( vector_PRECISION phi_c, vector_PRECISION phi, level_struct *l, struct Thread *threading );

#ifdef __cplusplus
}
#endif

#endif  // INTERPOLATION_PROXY_PRECISION_H
