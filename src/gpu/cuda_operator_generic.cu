#include "alloc_control.h"

extern "C" {

#include "cuda_operator_PRECISION.h"
#include "operator.h"



void cuda_operator_PRECISION_init(operator_PRECISION_struct *op) {
  op->clover_gpu = NULL;
  op->x_gpu = NULL;
  op->w_gpu = NULL;
}

void cuda_operator_PRECISION_alloc(operator_PRECISION_struct *op, const int type, level_struct *l) {
  unsigned int css = clover_site_size(l->num_lattice_site_var, l->depth);
  CUDA_MALLOC(op->clover_gpu, cu_cmplx_PRECISION, css * l->num_inner_lattice_sites);
  CUDA_MALLOC(op->x_gpu, cu_cmplx_PRECISION, l->vector_size);
  CUDA_MALLOC(op->w_gpu, cu_cmplx_PRECISION, l->vector_size);
}

void cuda_operator_PRECISION_free(operator_PRECISION_struct *op, const int type, level_struct *l) {
  unsigned int css = clover_site_size(l->num_lattice_site_var, l->depth);
  CUDA_FREE(op->clover_gpu, cu_cmplx_PRECISION, css * l->num_inner_lattice_sites);
  CUDA_FREE(op->x_gpu, cu_cmplx_PRECISION, l->vector_size);
  CUDA_FREE(op->w_gpu, cu_cmplx_PRECISION, l->vector_size);
}
}