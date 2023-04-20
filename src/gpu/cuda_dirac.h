#ifndef CUDA_DIRAC_H
#define CUDA_DIRAC_H

#include "complex_types_double.h"
#include "level_struct.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_dirac_setup(config_double hopp, config_double clover, level_struct *l);

#ifdef __cplusplus
}
#endif

#endif  // CUDA_DIRAC_H
