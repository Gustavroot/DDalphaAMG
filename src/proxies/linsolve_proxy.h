#ifndef LINSOLVE_PROXY_H
#define LINSOLVE_PROXY_H
#include "algorithm_structs.h"
#include "level_struct.h"

void fgmres_MP_struct_init(gmres_MP_struct *p);

void fgmres_MP_struct_alloc(int m, int n, int vl, double tol, const int prec_kind,
                            void (*precond)(), gmres_MP_struct *p, level_struct *l);

void fgmres_MP_struct_free(gmres_MP_struct *p);

#endif  // LINSOLVE_PROXY_H
