/** \file data_layout_proxy_generic.h
 *  \brief Mediates execution between data_layout_generic.h and cuda_data_layout_generic.h
 */

#ifndef DATA_LAYOUT_PROXY_PRECISION_H
#define DATA_LAYOUT_PROXY_PRECISION_H

#include "level_struct.h"

/**
 * Defines neighbor table (for the application of the entire operator), negative
 * inner boundary table (for communication) and translation table (for translation
 * to lexicographical site ordnering).
 * 
 * \param       op  Operator struct that will be both written to and read from. The neighbor table
 *                  from that struct will be used as output.
 * \param[out]  bt  Boundary table that will be written. For details see
 *                  boundary_table::boundary_table.
 * \param[out]  dt  Dimension table.
 * \param[in]   l   see level_struct
 */
void define_nt_bt_tt_PRECISION(operator_PRECISION_struct *op, int **bt, int *dt, level_struct *l);
#endif  // DATA_LAYOUT_PROXY_PRECISION_H
