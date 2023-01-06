#include "dirac_proxy_PRECISION.h"

#ifdef CUDA_OPT
#include "gpu/cuda_dirac_PRECISION.h"
#endif

#include "dirac_PRECISION.h"


void d_plus_clover_PRECISION(vector_PRECISION eta, vector_PRECISION phi,
                             operator_PRECISION_struct *op, level_struct *l,
                             struct Thread *threading)
{
#ifdef CUDA_OPT
  // cuda_d_plus_clover_PRECISION(eta, phi, op, l, threading);
#else
#endif
  d_plus_clover_PRECISION_cpu(eta, phi, op, l, threading);
}
