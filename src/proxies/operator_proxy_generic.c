#include "operator_PRECISION.h"
#ifdef CUDA_OPT
#include "gpu/cuda_operator_PRECISION.h"
#endif

void operator_PRECISION_init(operator_PRECISION_struct *op) {
#ifdef CUDA_OPT
  cuda_operator_PRECISION_init(op);
#endif
  cpu_operator_PRECISION_init(op);
}

void operator_PRECISION_alloc(operator_PRECISION_struct *op, const int type, level_struct *l) {
#ifdef CUDA_OPT
  cuda_operator_PRECISION_alloc(op, type, l);
#endif
  cpu_operator_PRECISION_alloc(op, type, l);
}

void operator_PRECISION_free(operator_PRECISION_struct *op, const int type, level_struct *l) {
#ifdef CUDA_OPT
  cuda_operator_PRECISION_free(op, type, l);
#endif
  cpu_operator_PRECISION_free(op, type, l);
}