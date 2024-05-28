#include "dirac_proxy_PRECISION.h"

#ifdef CUDA_OPT
#include "gpu/cuda_dirac_PRECISION.h"
#include "gpu/cuda_oddeven_PRECISION.h"
#endif

#include <complex.h>
#include "dirac_PRECISION.h"
#include "console_out.h"
#include "linalg_PRECISION.h"
#include "operator.h"
#include "oddeven_PRECISION.h"


void d_plus_clover_PRECISION(vector_PRECISION eta, vector_PRECISION phi,
                             operator_PRECISION_struct *op, level_struct *l,
                             struct Thread *threading)
{
#ifdef CUDA_OPT
  cuda_d_plus_clover_PRECISION_vectorwrapper(eta, phi, op, l, threading);
#else
  d_plus_clover_PRECISION_cpu(eta, phi, op, l, threading);
#endif
}

void apply_schur_complement_PRECISION(vector_PRECISION out, vector_PRECISION in,
                                      operator_PRECISION_struct *op, level_struct *l,
                                      struct Thread *threading)
{
#ifdef CUDA_OPT
  cuda_apply_schur_complement_PRECISION_vectorwrapper(out, in, op, l, threading);
#else
  apply_schur_complement_PRECISION_cpu(out, in, op, l, threading);
#endif
}
