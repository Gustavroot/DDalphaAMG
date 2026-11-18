#ifndef ODDEVEN_PROXY_PRECISION_H
#define ODDEVEN_PROXY_PRECISION_H

#include "algorithm_structs_PRECISION.h"
#include "level_struct.h"


void solve_oddeven_PRECISION( gmres_PRECISION_struct *p, operator_PRECISION_struct *op, level_struct *l, struct Thread *threading );
int smoother_resident_PRECISION_check( vector_PRECISION phi, vector_PRECISION eta, int res,
                                       level_struct *l, struct Thread *threading );
void solve_oddeven_resident_PRECISION( vector_PRECISION phi, gmres_PRECISION_struct *p, operator_PRECISION_struct *op,
                                       level_struct *l, struct Thread *threading );

#endif  // ODDEVEN_PROXY_PRECISION_H
