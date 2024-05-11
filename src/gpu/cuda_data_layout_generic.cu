#include "cuda_data_layout_PRECISION.h"
#include "miscellaneous.h"
#include "util_macros.h"

extern "C" void cuda_define_nt_bt_tt_PRECISION(operator_PRECISION_struct *op, int **bt, int *dt,
                                    level_struct *l) {
  if(l->depth != 0){
    // nothing to do
    return;
  }

  if ( g.oddeven_copy_nt_2_gpu==1 ) {

    int mu, le[4], N[4];

    //int j, k, k_e, k_o, n=l->num_inner_lattice_sites, oe_offset=0, mu, nu,
    //    sc_size = 42, lu_dec_size = 42, bs, **bt = NULL,
    //    //*eot = NULL, *nt = NULL, *tt = NULL, t, z, y, x, le[4], N[4];
    //    *eot = NULL, t, z, y, x, le[4], N[4];
    //config_double sc_in = in->clover, nc_in = in->D;
    //config_PRECISION Aee = NULL, Aoo = NULL;
    operator_PRECISION_struct *opx = &(l->oe_op_PRECISION);

    for ( mu=0; mu<4; mu++ ) {
      le[mu] = l->local_lattice[mu];
      N[mu] = le[mu]+1;
      //op->table_dim[mu] = N[mu];
    }

    //CUDA_MALLOC( op->neighbor_table_gpu, int, 5*N[T]*N[Z]*N[Y]*N[X] );

    cuda_safe_call(cudaMemcpy(opx->neighbor_table_gpu, opx->neighbor_table,
                              4*N[T]*N[Z]*N[Y]*N[X] * sizeof(int), cudaMemcpyHostToDevice));
  } else {
    cuda_safe_call(cudaMemcpy(op->neighbor_table_gpu, op->neighbor_table,
                              4 * l->num_inner_lattice_sites * sizeof(int), cudaMemcpyHostToDevice));
  }

  // this case does not happen when odd-even
  if ( bt!=NULL ) {
    for (size_t i = 0; i < 8; i++) {
      ASSERT(op->c.num_boundary_sites[i] == op->cuda_c.num_boundary_sites[i]);
      cuda_safe_call(cudaMemcpy(op->cuda_c.boundary_table_gpu[i], bt[i],
                                op->c.num_boundary_sites[i] * sizeof(int), cudaMemcpyHostToDevice));
    }
  }
  //}
}
