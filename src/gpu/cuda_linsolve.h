#ifndef CUDA_LINSOLVE_H
#define CUDA_LINSOLVE_H

#include "algorithm_structs.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_fgmres_MP_struct_init(gmres_MP_struct *p);

void cuda_fgmres_MP_struct_alloc(int m, int n, int vl, double tol, const int prec_kind,
                                 void (*precond)(), gmres_MP_struct *p, level_struct *l);

void cuda_fgmres_MP_struct_free(gmres_MP_struct *p);

#ifdef __cplusplus
}
#endif

#endif  // CUDA_LINSOLVE_H
