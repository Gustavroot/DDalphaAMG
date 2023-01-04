#ifndef DIRAC_PROXY_PRECISION_H
#define DIRAC_PROXY_PRECISION_H

#include "complex_types_PRECISION.h"
#include "level_struct.h"

void d_plus_clover_PRECISION(vector_PRECISION eta, vector_PRECISION phi,
                             operator_PRECISION_struct *op, level_struct *l,
                             struct Thread *threading);

#endif // DIRAC_PROXY_PRECISION_H
