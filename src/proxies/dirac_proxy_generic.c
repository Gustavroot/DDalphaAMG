#include "dirac_proxy_PRECISION.h"

#ifdef CUDA_OPT
#include "gpu/cuda_dirac_PRECISION.h"
#endif

#include <complex.h>
#include "dirac_PRECISION.h"
#include "console_out.h"
#include "linalg_PRECISION.h"
#include "operator.h"

void d_plus_clover_PRECISION(vector_PRECISION eta, complex_PRECISION const * phi,
                             operator_PRECISION_struct *op, level_struct *l,
                             struct Thread *threading)
{
#ifdef CUDA_OPT
  cuda_d_plus_clover_PRECISION_vectorwrapper(eta, phi, op, l, threading);
#else
  d_plus_clover_PRECISION_cpu(eta, phi, op, l, threading);
#endif
#if 0
  d_plus_clover_PRECISION_cpu(op->w_test, phi, op, l, threading);
  START_LOCKED_MASTER(threading)
  int fail_fish = 0;
  for(size_t i = 0; i < l->inner_vector_size; i++) {
    complex_PRECISION cuda_value = *(eta+i);
    complex_PRECISION cpu_value = *(op->w_test+i);
    if (cabs(cuda_value - cpu_value) > 1.0e-7) {
      fail_fish = 1;
      warning0("mismatch at index %d: CUDA is %.4f + %.4fi <> CPU is %.4f + %.4fi\n",
               i, creal(cuda_value), cimag(cuda_value), creal(cpu_value), cimag(cpu_value));
    }
  }
  if (fail_fish != 0) {
    error0("Fail fish!");
  }
  END_LOCKED_MASTER(threading)
#endif
}
