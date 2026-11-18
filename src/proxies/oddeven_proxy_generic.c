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

// Depth-0 smoother on the GPU with phi and eta already there (left by the interpolation and by the
// restriction): returns 1 if it can be used for this call (see cuda_smoother_resident_PRECISION_check),
// always 0 without CUDA. Called by all threads.
int smoother_resident_PRECISION_check( vector_PRECISION phi, vector_PRECISION eta, int res,
                                       level_struct *l, struct Thread *threading ){
#ifdef CUDA_OPT
  return cuda_smoother_resident_PRECISION_check( phi, eta, res, l, threading );
#else
  return 0;
#endif
}

// phi <- phi + M^-1 ( eta - D phi ) at depth 0, on the GPU; only after smoother_resident_PRECISION_check
// has returned 1
void solve_oddeven_resident_PRECISION( vector_PRECISION phi, gmres_PRECISION_struct *p, operator_PRECISION_struct *op,
                                       level_struct *l, struct Thread *threading ){
#ifdef CUDA_OPT
  cuda_solve_oddeven_PRECISION_resident( phi, p, op, l, threading );
#else
  error0("solve_oddeven_resident_PRECISION(...) needs CUDA\n");
#endif
}
