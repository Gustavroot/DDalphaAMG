#include "alloc_control.h"
#include "miscellaneous.h"
#include "cuda_ghost_PRECISION.h"

extern "C" {

#include "operator.h"

void cuda_operator_PRECISION_init(operator_PRECISION_struct *op) {
  op->clover_gpu = NULL;
  op->clover_componentwise_gpu = NULL;
  op->D_gpu = NULL;
  for ( int mu=0; mu<4; mu++ ) {
    op->Ds_componentwise_gpu[mu] = NULL;
  }
  op->x_gpu = NULL;
  op->x_componentwise_gpu = NULL;
  op->w_gpu = NULL;
  op->w_componentwise_gpu = NULL;
  op->pbuf_gpu = NULL;
  op->prpT_gpu = NULL;
  op->prpZ_gpu = NULL;
  op->prpY_gpu = NULL;
  op->prpX_gpu = NULL;
  op->prnT_gpu = NULL;
  op->prnZ_gpu = NULL;
  op->prnY_gpu = NULL;
  op->prnX_gpu = NULL;
  op->neighbor_table_gpu = NULL;

  // Communication stuff
  for ( int i=0; i<8; i++ ) {
    op->cuda_c.boundary_table_gpu[i] = NULL;
    op->cuda_c.buffer_gpu[i] = NULL;
#ifdef GPU2GPU_COMMS_VIA_CPUS
    op->cuda_c.buffer[i] = NULL;
    op->cuda_c.buffer2[i] = NULL;
#endif
    op->cuda_c.in_use[i] = 0;
  }
  op->cuda_c.comm = 1;
}

void cuda_operator_PRECISION_alloc(operator_PRECISION_struct *op, const int type, level_struct *l) {
  if (l->depth != 0) {
    return;
  }
  unsigned int css = clover_site_size(l->num_lattice_site_var, l->depth);
  size_t pbs = projection_buffer_size(l->num_lattice_site_var, l->num_lattice_sites);
  CUDA_MALLOC(op->clover_gpu, cu_cmplx_PRECISION, css * l->num_inner_lattice_sites);
  CUDA_MALLOC(op->clover_componentwise_gpu, cu_cmplx_PRECISION, css * l->num_inner_lattice_sites);
  CUDA_MALLOC(op->x_gpu, cu_cmplx_PRECISION, l->inner_vector_size);
  CUDA_MALLOC(op->x_componentwise_gpu, cu_cmplx_PRECISION, l->inner_vector_size);
  CUDA_MALLOC(op->w_gpu, cu_cmplx_PRECISION, l->inner_vector_size);
  CUDA_MALLOC(op->w_componentwise_gpu, cu_cmplx_PRECISION, l->inner_vector_size);

  CUDA_MALLOC(op->D_gpu, cu_cmplx_PRECISION, 4 * 9 * l->num_inner_lattice_sites);
  for ( int mu=0; mu<4; mu++ ) {
    CUDA_MALLOC(op->Ds_componentwise_gpu[mu], cu_cmplx_PRECISION, 9 * l->num_inner_lattice_sites);
  };
  CUDA_MALLOC(op->neighbor_table_gpu, int, 4 * l->num_inner_lattice_sites);

  CUDA_MALLOC(op->pbuf_gpu, cu_cmplx_PRECISION, pbs);
  CUDA_MALLOC(op->prpT_gpu, cu_cmplx_PRECISION, pbs);
  CUDA_MALLOC(op->prpZ_gpu, cu_cmplx_PRECISION, pbs);
  CUDA_MALLOC(op->prpY_gpu, cu_cmplx_PRECISION, pbs);
  CUDA_MALLOC(op->prpX_gpu, cu_cmplx_PRECISION, pbs);
  CUDA_MALLOC(op->prnT_gpu, cu_cmplx_PRECISION, pbs);
  CUDA_MALLOC(op->prnZ_gpu, cu_cmplx_PRECISION, pbs);
  CUDA_MALLOC(op->prnY_gpu, cu_cmplx_PRECISION, pbs);
  CUDA_MALLOC(op->prnX_gpu, cu_cmplx_PRECISION, pbs);


  // Communication stuff
  cuda_ghost_alloc_PRECISION( 0, &(op->cuda_c), l );
  
  for ( int mu=0; mu<4; mu++ ) {
    int its = 1;
    for ( int nu=0; nu<4; nu++ ) {
      if ( mu != nu ) {
        its *= l->local_lattice[nu];
      }
    }
    op->cuda_c.num_boundary_sites[2*mu] = its;
    op->cuda_c.num_boundary_sites[2*mu+1] = its;
    CUDA_MALLOC( op->cuda_c.boundary_table_gpu[2*mu], int, its );
    if ( type == _SCHWARZ ) {
      CUDA_MALLOC( op->cuda_c.boundary_table_gpu[2*mu+1], int, its );
    } else {
      op->cuda_c.boundary_table_gpu[2*mu+1] = op->cuda_c.boundary_table_gpu[2*mu];
    }
  }
}

void cuda_operator_PRECISION_free(operator_PRECISION_struct *op, const int type, level_struct *l) {
  if (l->depth != 0) {
    return;
  }
  unsigned int css = clover_site_size(l->num_lattice_site_var, l->depth);
  size_t pbs = projection_buffer_size(l->num_lattice_site_var, l->num_lattice_sites);
  CUDA_FREE(op->clover_gpu, cu_cmplx_PRECISION, css * l->num_inner_lattice_sites);
  CUDA_FREE(op->clover_componentwise_gpu, cu_cmplx_PRECISION, css * l->num_inner_lattice_sites);
  CUDA_FREE(op->D_gpu, cu_cmplx_PRECISION, 4 * 9 * l->num_inner_lattice_sites);
  for ( int mu=0; mu<4; mu++ ) {
    CUDA_FREE(op->Ds_componentwise_gpu[mu], cu_cmplx_PRECISION, 9 * l->num_inner_lattice_sites);
  };
  CUDA_FREE(op->x_gpu, cu_cmplx_PRECISION, l->inner_vector_size);
  CUDA_FREE(op->x_componentwise_gpu, cu_cmplx_PRECISION, l->inner_vector_size);
  CUDA_FREE(op->w_gpu, cu_cmplx_PRECISION, l->inner_vector_size);
  CUDA_FREE(op->w_componentwise_gpu, cu_cmplx_PRECISION, l->inner_vector_size);
  CUDA_FREE(op->pbuf_gpu, cu_cmplx_PRECISION, pbs);

  CUDA_FREE(op->neighbor_table_gpu, int, 4 * l->num_inner_lattice_sites);
  CUDA_FREE(op->prpT_gpu, cu_cmplx_PRECISION, pbs);
  CUDA_FREE(op->prpZ_gpu, cu_cmplx_PRECISION, pbs);
  CUDA_FREE(op->prpY_gpu, cu_cmplx_PRECISION, pbs);
  CUDA_FREE(op->prpX_gpu, cu_cmplx_PRECISION, pbs);
  CUDA_FREE(op->prnT_gpu, cu_cmplx_PRECISION, pbs);
  CUDA_FREE(op->prnZ_gpu, cu_cmplx_PRECISION, pbs);
  CUDA_FREE(op->prnY_gpu, cu_cmplx_PRECISION, pbs);
  CUDA_FREE(op->prnX_gpu, cu_cmplx_PRECISION, pbs);

  // Communication stuff
  cuda_ghost_free_PRECISION( &(op->cuda_c), l );
  
  for ( int mu=0; mu<4; mu++ ) {
    int its = 1;
    for ( int nu=0; nu<4; nu++ ) {
      if ( mu != nu ) {
        its *= l->local_lattice[nu];
      }
    }
    
    CUDA_FREE( op->cuda_c.boundary_table_gpu[2*mu], int, its );
    if ( type == _SCHWARZ ) {
      CUDA_FREE( op->cuda_c.boundary_table_gpu[2*mu+1], int, its );
    } else {
      op->cuda_c.boundary_table_gpu[2*mu+1] = NULL;
    }
  }
}
}
