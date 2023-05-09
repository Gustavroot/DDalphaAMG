#ifndef CUDA_OEPRATOR_PRECISION_H
#define CUDA_OEPRATOR_PRECISION_H

#include "algorithm_structs_PRECISION.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_operator_PRECISION_init(operator_PRECISION_struct *op);
void cuda_operator_PRECISION_alloc(operator_PRECISION_struct *op, const int type, level_struct *l);
void cuda_operator_PRECISION_free(operator_PRECISION_struct *op, const int type, level_struct *l);

#ifdef __cplusplus
}
#endif


#endif // CUDA_OEPRATOR_PRECISION_H
