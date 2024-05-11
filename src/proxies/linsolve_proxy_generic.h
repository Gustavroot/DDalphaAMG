#ifndef LINSOLVE_PROXY_PRECISION_H
#define LINSOLVE_PROXY_PRECISION_H

#include "algorithm_structs_PRECISION.h"
#include "level_struct.h"

void fgmres_PRECISION_struct_init(gmres_PRECISION_struct *p);

void fgmres_PRECISION_struct_alloc(int m, int n, int vl, PRECISION tol, const int type,
                                   const int prec_kind, void (*precond)(), void (*eval_op)(),
                                   gmres_PRECISION_struct *p, level_struct *l);

void fgmres_PRECISION_struct_free(gmres_PRECISION_struct *p, level_struct *l);

#ifdef RICHARDSON_SMOOTHER
int richardson_PRECISION( gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading );
#endif

#endif  // LINSOLVE_PROXY_PRECISION_H
