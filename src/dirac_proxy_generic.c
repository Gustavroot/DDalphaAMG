#include "dirac_proxy_PRECISION.h"

#ifdef CUDA_OPT
#include "cuda_dirac_PRECISION.h"
#else
// VALIDATE
// #include "dirac_PRECISION.h" well somehow this is required here, but it is already
// included everywhere and there seems to be an issue when compiling the clifford header
// with SSE defined.
#endif

void d_plus_clover_PRECISION(vector_PRECISION eta, vector_PRECISION phi,
                             operator_PRECISION_struct *op, level_struct *l,
                             struct Thread *threading)
{
  // TODO
  d_plus_clover_PRECISION_cpu(eta, phi, op, l, threading);
#ifndef CUDA_OPT
#else
  // cuda_d_plus_clover_PRECISION(eta, phi, op, l, threading);
#endif
}
