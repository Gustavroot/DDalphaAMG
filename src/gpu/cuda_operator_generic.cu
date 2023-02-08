#include "alloc_control.h"

extern "C" {

#include "cuda_operator_PRECISION.h"
#include "operator.h"



void cuda_operator_PRECISION_init(operator_PRECISION_struct *op) {
  op->clover_gpu = NULL;
  op->x_gpu = NULL;
  op->w_gpu = NULL;
  op->prpT_gpu = NULL;
  op->prpZ_gpu = NULL;
  op->prpY_gpu = NULL;
  op->prpX_gpu = NULL;
}

void cuda_operator_PRECISION_alloc(operator_PRECISION_struct *op, const int type, level_struct *l) {
  unsigned int css = clover_site_size(l->num_lattice_site_var, l->depth);
  size_t pbs = projection_buffer_size(l->num_lattice_site_var, l->num_lattice_sites);
  CUDA_MALLOC(op->clover_gpu, cu_cmplx_PRECISION, css * l->num_inner_lattice_sites);
  CUDA_MALLOC(op->x_gpu, cu_cmplx_PRECISION, l->inner_vector_size);
  CUDA_MALLOC(op->w_gpu, cu_cmplx_PRECISION, l->inner_vector_size);

  // projection buffers are only used on the finest level.
  if (l->depth == 0) {
    CUDA_MALLOC(op->prpT_gpu, cu_cmplx_PRECISION, pbs);
    CUDA_MALLOC(op->prpZ_gpu, cu_cmplx_PRECISION, pbs);
    CUDA_MALLOC(op->prpY_gpu, cu_cmplx_PRECISION, pbs);
    CUDA_MALLOC(op->prpX_gpu, cu_cmplx_PRECISION, pbs);
  }
  
}

void cuda_operator_PRECISION_free(operator_PRECISION_struct *op, const int type, level_struct *l) {
  unsigned int css = clover_site_size(l->num_lattice_site_var, l->depth);
  size_t pbs = projection_buffer_size(l->num_lattice_site_var, l->num_lattice_sites);
  CUDA_FREE(op->clover_gpu, cu_cmplx_PRECISION, css * l->num_inner_lattice_sites);
  CUDA_FREE(op->x_gpu, cu_cmplx_PRECISION, l->vector_size);
  CUDA_FREE(op->w_gpu, cu_cmplx_PRECISION, l->vector_size);

  // projection buffers are only used on the finest level.
  if (l->depth == 0) {
    CUDA_FREE(op->prpT_gpu, cu_cmplx_PRECISION, pbs);
    CUDA_FREE(op->prpZ_gpu, cu_cmplx_PRECISION, pbs);
    CUDA_FREE(op->prpY_gpu, cu_cmplx_PRECISION, pbs);
    CUDA_FREE(op->prpX_gpu, cu_cmplx_PRECISION, pbs);
  }
}
}