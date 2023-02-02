#include "cuda_complex.h"
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

cu_cmplx_double to_cuda_cmplx_double(double _Complex from){
  return make_cu_cmplx_double(creall(from), cimagl(from));
}

cu_cmplx_float to_cuda_cmplx_float(float _Complex from){
  return make_cu_cmplx_float(crealf(from), cimagf(from));
}

#ifdef __cplusplus
}
#endif