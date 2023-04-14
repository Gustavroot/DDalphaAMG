#include "cuda_data_layout_PRECISION.h"
#include "miscellaneous.h"
#include "util_macros.h"

extern "C" void cuda_define_nt_bt_tt_PRECISION(operator_PRECISION_struct *op, int **bt, int *dt,
                                    level_struct *l) {
  if(l->depth != 0){
    // nothing to do
    return;
  }
  cuda_safe_call(cudaMemcpy(op->neighbor_table_gpu, op->neighbor_table,
                            4 * l->num_inner_lattice_sites * sizeof(int), cudaMemcpyHostToDevice));
  for (size_t i = 0; i < 8; i++) {
    ASSERT(op->c.num_boundary_sites[i] == op->cuda_c.num_boundary_sites[i]);
    cuda_safe_call(cudaMemcpy(op->cuda_c.boundary_table_gpu[i], bt[i],
                              op->c.num_boundary_sites[i] * sizeof(int), cudaMemcpyHostToDevice));
  }
  
}
