#include "linsolve_PRECISION.h"
#include "linsolve_proxy_PRECISION.h"
#include "alloc_control.h"
#include "linalg_PRECISION.h"
#include "oddeven_PRECISION.h"

#ifdef CUDA_OPT
#include "gpu/cuda_oddeven_PRECISION.h"
#endif

void solve_oddeven_PRECISION( gmres_PRECISION_struct *p, operator_PRECISION_struct *op, level_struct *l, struct Thread *threading ){
#ifdef CUDA_OPT
  cuda_solve_oddeven_PRECISION_vectorwrapper( p, op, l, threading );
#else
  solve_oddeven_PRECISION_cpu( p, op, l, threading );
#endif
}

