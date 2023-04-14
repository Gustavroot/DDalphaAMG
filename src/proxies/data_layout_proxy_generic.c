#include "data_layout_proxy_PRECISION.h"
#include "data_layout_PRECISION.h"
#include "gpu/cuda_data_layout_PRECISION.h"

void define_nt_bt_tt_PRECISION(operator_PRECISION_struct *op, int **bt, int *dt, level_struct *l){
  define_nt_bt_tt_PRECISION_cpu(op, bt, dt, l);
#ifdef CUDA_OPT
  cuda_define_nt_bt_tt_PRECISION(op, bt, dt, l);
#endif
}
