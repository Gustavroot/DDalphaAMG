#ifndef CUDA_DATA_LAYOUT_PRECISION_H
#define CUDA_DATA_LAYOUT_PRECISION_H
#include "algorithm_structs_PRECISION.h"
#include "level_struct.h"

void cuda_define_nt_bt_tt_PRECISION(operator_PRECISION_struct *op, int **bt, int *dt,
                                    level_struct *l);

#endif // CUDA_DATA_LAYOUT_PRECISION_H