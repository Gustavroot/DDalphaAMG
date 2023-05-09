#include "linsolve_proxy.h"

#include "linsolve.h"

#ifdef CUDA_OPT
#include "gpu/cuda_linsolve.h"
#endif

void fgmres_MP_struct_init(gmres_MP_struct *p) {
#ifdef CUDA_OPT
  cuda_fgmres_MP_struct_init(p);
#endif
  cpu_fgmres_MP_struct_init(p);
}

void fgmres_MP_struct_alloc(int m, int n, int vl, double tol, const int prec_kind,
                            void (*precond)(), gmres_MP_struct *p, level_struct *l) {
#ifdef CUDA_OPT
  cuda_fgmres_MP_struct_alloc(m, n, vl, tol, prec_kind, precond, p, l);
#endif
  cpu_fgmres_MP_struct_alloc(m, n, vl, tol, prec_kind, precond, p, l);
}

void fgmres_MP_struct_free(gmres_MP_struct *p) {
#ifdef CUDA_OPT
  cuda_fgmres_MP_struct_free(p);
#endif
  cpu_fgmres_MP_struct_free(p);
}
