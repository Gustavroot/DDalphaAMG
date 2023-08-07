/** \file cuda_dirac.h
 *  \brief Additional setup for Dirac operator on GPUs.
 */

#ifndef CUDA_DIRAC_H
#define CUDA_DIRAC_H

#include "complex_types_double.h"
#include "level_struct.h"

#ifdef __cplusplus
extern "C" {
#endif

/** \brief Copy self and neighbor coupling coefficients to GPU.
 * 
 *  Also performs componentwise reordering of those arrays.
 * 
 *  \param[in]      hopp    unused
 *  \param[in]      clover  unused
 *  \param[in,out]  l       Has the appropriate GPU arrays populated from the
 *                          contained CPU arrays.
 */
void cuda_dirac_setup(config_double hopp, config_double clover, level_struct *l);

#ifdef __cplusplus
}
#endif

#endif  // CUDA_DIRAC_H
