#include "main.h"
#include "proxies/interpolation_proxy_PRECISION.h"

#ifdef CUDA_OPT
#include "gpu/cuda_interpolation_PRECISION.h"
#endif

// the GPU versions use l->is_PRECISION.op, which is not available with HALF_PREC_STORAGE
#if defined( CUDA_OPT ) && !defined( HALF_PREC_STORAGE )
#define INTERPOLATION_ON_GPU_PRECISION
#endif

void interpolation_PRECISION_alloc( level_struct *l ) {
  cpu_interpolation_PRECISION_alloc( l );
#ifdef CUDA_OPT
  cuda_interpolation_PRECISION_alloc( l );
#endif
}

void interpolation_PRECISION_free( level_struct *l ) {
#ifdef CUDA_OPT
  cuda_interpolation_PRECISION_free( l );
#endif
  cpu_interpolation_PRECISION_free( l );
}

void interpolation_PRECISION_op_to_device( level_struct *l, struct Thread *threading ) {
#ifdef INTERPOLATION_ON_GPU_PRECISION
  START_LOCKED_MASTER(threading)
  cuda_interpolation_PRECISION_op_to_gpu( l );
  END_LOCKED_MASTER(threading)
#endif
}

void interpolate_PRECISION( vector_PRECISION phi, vector_PRECISION phi_c, level_struct *l, struct Thread *threading ) {
#ifdef INTERPOLATION_ON_GPU_PRECISION
  cuda_interpolate_PRECISION_vectorwrapper( phi, phi_c, l, threading );
#else
  interpolate_PRECISION_cpu( phi, phi_c, l, threading );
#endif
}

void interpolate3_PRECISION( vector_PRECISION phi, vector_PRECISION phi_c, level_struct *l, struct Thread *threading ) {
#ifdef INTERPOLATION_ON_GPU_PRECISION
  cuda_interpolate3_PRECISION_vectorwrapper( phi, phi_c, l, threading );
#else
  interpolate3_PRECISION_cpu( phi, phi_c, l, threading );
#endif
}

void restrict_PRECISION( vector_PRECISION phi_c, vector_PRECISION phi, level_struct *l, struct Thread *threading ) {
#ifdef INTERPOLATION_ON_GPU_PRECISION
  cuda_restrict_PRECISION_vectorwrapper( phi_c, phi, l, threading );
#else
  restrict_PRECISION_cpu( phi_c, phi, l, threading );
#endif
}
