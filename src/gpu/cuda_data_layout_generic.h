/** \file cuda_data_layout_generic.h
 *  \brief CUDA implementation of cuda_define_nt_bt_tt_PRECISION.
 *  \see data_layout_proxy_generic.h
 */

#ifndef CUDA_DATA_LAYOUT_PRECISION_H
#define CUDA_DATA_LAYOUT_PRECISION_H
#include "algorithm_structs_PRECISION.h"
#include "level_struct.h"

#ifdef __cplusplus
extern "C" {
#endif

/** \brief Copies neighbor table and boundary table from CPU to GPU storage.
 *
 *  \param op   Holds the neighbor tables and communication structs.
 *  \param bt   unused, for compatibility only
 *  \param dt   unused, for compatibility only
 *  \param l    see level_struct
 */
void cuda_define_nt_bt_tt_PRECISION(operator_PRECISION_struct *op, int **bt, int *dt,
                                    level_struct *l);

#ifdef __cplusplus
}
#endif
#endif  // CUDA_DATA_LAYOUT_PRECISION_H
