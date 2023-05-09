#include "dirac_proxy.h"

#include "dirac.h"
#ifdef CUDA_OPT
#include "gpu/cuda_dirac.h"
#endif

void dirac_setup(config_double hopp, config_double clover, level_struct *l) {
  cpu_dirac_setup(hopp, clover, l);
#ifdef CUDA_OPT
  cuda_dirac_setup(hopp, clover, l);
#endif
}
