#include "operator_PRECISION.h"
#ifdef CUDA_OPT
#include "gpu/cuda_operator_PRECISION.h"
#endif

void operator_PRECISION_init(operator_PRECISION_struct *op) {
  cpu_operator_PRECISION_init(op);
#ifdef CUDA_OPT
  cuda_operator_PRECISION_init(op);
#endif
}

void operator_PRECISION_alloc(operator_PRECISION_struct *op, const int type, level_struct *l) {
  cpu_operator_PRECISION_alloc(op, type, l);
#ifdef CUDA_OPT
  cuda_operator_PRECISION_alloc(op, type, l);
#endif
}

void operator_PRECISION_free(operator_PRECISION_struct *op, const int type, level_struct *l) {
  cpu_operator_PRECISION_free(op, type, l);
#ifdef CUDA_OPT
  cuda_operator_PRECISION_free(op, type, l);
#endif
}
