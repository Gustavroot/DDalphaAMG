#include "linsolve_PRECISION.h"
#include "linsolve_proxy_PRECISION.h"
#include "alloc_control.h"
#include "linalg_PRECISION.h"


complex_PRECISION global_inner_product_PRECISION( vector_PRECISION* V, vector_PRECISION psi, complex_PRECISION *result,
                  int n, int start, int end, level_struct *l, struct Thread *threading ) {

  // we generalize this to be a multi inner dot product by default

//#ifdef CUDA_OPT
//  cuda_global_inner_product_PRECISION_vectorwrapper( V, psi, result, n, start, end, l, threading );
//  return 1.0;
//#else
  return global_inner_product_PRECISION_cpu( V, psi, result, n, start, end, l, threading );
//#endif
}
