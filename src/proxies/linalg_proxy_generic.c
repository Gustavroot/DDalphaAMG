#include "linsolve_PRECISION.h"
#include "linsolve_proxy_PRECISION.h"
#include "alloc_control.h"
#include "linalg_PRECISION.h"
#ifdef CUDA_OPT
#include "gpu/cuda_linalg_PRECISION.h"
#endif


complex_PRECISION global_inner_product_PRECISION( vector_PRECISION* V, vector_PRECISION psi, complex_PRECISION *result,
                  int n, int start, int end, gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading ) {

  // V and psi are host vectors: the dot product stays on the CPU, also with CUDA_OPT, instead of
  // allocating GPU buffers and copying all the vectors to the GPU at every call
  // (cuda_global_inner_product_PRECISION_vectorwrapper). The callers (BiCGStab, CGN, GCR, power
  // iteration) run on the CPU at all levels; the GPU solvers call cuda_global_inner_product_PRECISION
  // on their GPU vectors directly.
  return global_inner_product_PRECISION_cpu( V, psi, result, n, start, end, p, l, threading );
}
