#include "linsolve_PRECISION.h"
#include "linsolve_proxy_PRECISION.h"
#include "alloc_control.h"
#include "linalg_PRECISION.h"
#ifdef CUDA_OPT
#include "gpu/cuda_linalg_PRECISION.h"
#endif


complex_PRECISION global_inner_product_PRECISION( vector_PRECISION* V, vector_PRECISION psi, complex_PRECISION *result,
                  int n, int start, int end, gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading ) {

  // we have generalized this to be a multi inner dot product by default

#ifdef CUDA_OPT
  int was_result_NULL = 0;
  if ( result==NULL ) {
    was_result_NULL = 1;
    PUBLIC_MALLOC( result, complex_PRECISION, 1 );
  }
  cuda_global_inner_product_PRECISION_vectorwrapper( V, psi, result, n, start, end, p, l, threading );
  complex_PRECISION outx = result[0];
  if ( was_result_NULL==1 ) {
    PUBLIC_FREE( result, complex_PRECISION, 1 );
  }
  return outx;
#else
  return global_inner_product_PRECISION_cpu( V, psi, result, n, start, end, p, l, threading );
#endif
}
