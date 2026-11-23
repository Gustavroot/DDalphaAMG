#include "linsolve_proxy.h"

#include "linsolve.h"
#include "linsolve_double.h"

#ifdef CUDA_OPT
#include "gpu/cuda_linsolve.h"
#include "gpu/cuda_outer_fgmres.h"
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

// outer FGMRES of the solve: on the GPU when cuda_outer_fgmres_applicable (the fine vectors then stay
// on the GPU for the whole solve, -DOUTER_FGMRES_ON_CPU to switch it off), otherwise fgmres_double
int outer_fgmres_double(gmres_double_struct *p, level_struct *l, struct Thread *threading) {
#ifdef CUDA_OPT
  if (cuda_outer_fgmres_applicable(p, l))
    return cuda_outer_fgmres_double(p, l, threading);
#endif
  return fgmres_double(p, l, threading);
}
