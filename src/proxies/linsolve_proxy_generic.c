#include "linsolve_PRECISION.h"
#include "linsolve_proxy_PRECISION.h"
#include "alloc_control.h"
#include "linalg_PRECISION.h"

#ifdef CUDA_OPT
#include "gpu/cuda_linsolve_PRECISION.h"
#endif

#ifdef RICHARDSON_SMOOTHER
#include "linsolve_PRECISION.h"
#endif

void fgmres_PRECISION_struct_init(gmres_PRECISION_struct *p) {
#ifdef CUDA_OPT
  cuda_fgmres_PRECISION_struct_init(p);
#endif
  cpu_fgmres_PRECISION_struct_init(p);
}

void fgmres_PRECISION_struct_alloc(int m, int n, int vl, PRECISION tol, const int type,
                                   const int prec_kind, void (*precond)(), void (*eval_op)(),
                                   gmres_PRECISION_struct *p, level_struct *l) {
#ifdef CUDA_OPT
  cuda_fgmres_PRECISION_struct_alloc(m, n, vl, tol, type, prec_kind, precond, eval_op, p, l);
#endif
  cpu_fgmres_PRECISION_struct_alloc(m, n, vl, tol, type, prec_kind, precond, eval_op, p, l);
}

void fgmres_PRECISION_struct_free(gmres_PRECISION_struct *p, level_struct *l) {
#ifdef CUDA_OPT
  cuda_fgmres_PRECISION_struct_free(p, l);
#endif
  cpu_fgmres_PRECISION_struct_free(p, l);
}

#ifdef RICHARDSON_SMOOTHER
int richardson_PRECISION( gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading ) {
#ifdef CUDA_OPT
  return cuda_richardson_PRECISION_vectorwrapper( p, l, threading );
#else
  return richardson_PRECISION_cpu( p, l, threading );
#endif
}
#endif
