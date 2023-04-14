#ifndef OPERATOR_PROXY_PRECISION_H
#define OPERATOR_PROXY_PRECISION_H

#include "algorithm_structs_PRECISION.h"

void operator_PRECISION_init(operator_PRECISION_struct *op);
void operator_PRECISION_alloc(operator_PRECISION_struct *op, const int type, level_struct *l);
void operator_PRECISION_free(operator_PRECISION_struct *op, const int type, level_struct *l);

#endif // OPERATOR_PROXY_PRECISION_H
