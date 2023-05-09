#include "alloc_control.h"
#include "cuda_linsolve.h"
#include "cuda_linsolve_double.h"
#include "cuda_linsolve_float.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_fgmres_MP_struct_init(gmres_MP_struct *p) {
  cuda_fgmres_float_struct_init(&(p->float_section));
  cuda_fgmres_double_struct_init(&(p->double_section));
}

void cuda_fgmres_MP_struct_alloc(int m, int n, int vl, double tol, const int prec_kind,
                                 void (*precond)(), gmres_MP_struct *p, level_struct *l) {}

void cuda_fgmres_MP_struct_free(gmres_MP_struct *p) {}

#ifdef __cplusplus
}
#endif
