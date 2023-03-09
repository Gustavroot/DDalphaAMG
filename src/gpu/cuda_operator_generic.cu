#include "alloc_control.h"
#include "miscellaneous.h"

extern "C" {

#include "operator.h"

void cuda_operator_PRECISION_init(operator_PRECISION_struct *op) {
  op->w_test = NULL;
  op->clover_gpu = NULL;
  op->D_gpu = NULL;
  op->x_gpu = NULL;
  op->w_gpu = NULL;
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
}

void cuda_operator_PRECISION_alloc(operator_PRECISION_struct *op, const int type, level_struct *l) {
  if (l->depth != 0) {
    error0("cuda_operator_PRECISION_alloc is a finest level only function.");
  }
  MALLOC(op->w_test, complex_PRECISION, l->inner_vector_size);
  unsigned int css = clover_site_size(l->num_lattice_site_var, l->depth);
  size_t pbs = projection_buffer_size(l->num_lattice_site_var, l->num_lattice_sites);
  CUDA_MALLOC(op->clover_gpu, cu_cmplx_PRECISION, css * l->num_inner_lattice_sites);
  CUDA_MALLOC(op->x_gpu, cu_cmplx_PRECISION, l->inner_vector_size);
  CUDA_MALLOC(op->w_gpu, cu_cmplx_PRECISION, l->inner_vector_size);

  CUDA_MALLOC(op->D_gpu, cu_cmplx_PRECISION, 4 * 9 * l->num_inner_lattice_sites);
  cuda_safe_call(cudaMemcpy(op->D_gpu, op->D,
                            4 * 9 * l->num_inner_lattice_sites * sizeof(cu_cmplx_PRECISION),
                            cudaMemcpyHostToDevice));

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
}

void cuda_operator_PRECISION_free(operator_PRECISION_struct *op, const int type, level_struct *l) {
  if (l->depth != 0) {
    error0("cuda_operator_PRECISION_alloc is a finest level only function.");
  }
  FREE(op->w_test, complex_PRECISION, l->inner_vector_size);
  unsigned int css = clover_site_size(l->num_lattice_site_var, l->depth);
  size_t pbs = projection_buffer_size(l->num_lattice_site_var, l->num_lattice_sites);
  CUDA_FREE(op->clover_gpu, cu_cmplx_PRECISION, css * l->num_inner_lattice_sites);
  CUDA_FREE(op->D_gpu, cu_cmplx_PRECISION, 4 * 9 * l->num_inner_lattice_sites);
  CUDA_FREE(op->x_gpu, cu_cmplx_PRECISION, l->inner_vector_size);
  CUDA_FREE(op->w_gpu, cu_cmplx_PRECISION, l->inner_vector_size);
  CUDA_FREE(op->pbuf_gpu, cu_cmplx_PRECISION, l->inner_vector_size);

  CUDA_FREE(op->neighbor_table_gpu, int, 4 * l->num_inner_lattice_sites);
  CUDA_FREE(op->prpT_gpu, cu_cmplx_PRECISION, pbs);
  CUDA_FREE(op->prpZ_gpu, cu_cmplx_PRECISION, pbs);
  CUDA_FREE(op->prpY_gpu, cu_cmplx_PRECISION, pbs);
  CUDA_FREE(op->prpX_gpu, cu_cmplx_PRECISION, pbs);
  CUDA_FREE(op->prnT_gpu, cu_cmplx_PRECISION, pbs);
  CUDA_FREE(op->prnZ_gpu, cu_cmplx_PRECISION, pbs);
  CUDA_FREE(op->prnY_gpu, cu_cmplx_PRECISION, pbs);
  CUDA_FREE(op->prnX_gpu, cu_cmplx_PRECISION, pbs);
}
}