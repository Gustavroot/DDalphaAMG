#include "dirac_proxy_PRECISION.h"

#ifdef CUDA_OPT
#include "gpu/cuda_dirac_PRECISION.h"
#include "gpu/cuda_oddeven_PRECISION.h"
#endif

// this is here for testing purposes, for now,
// of CPU->GPU odd-even smoothers
#include "oddeven_PRECISION.h"

#include <complex.h>
#include "dirac_PRECISION.h"
#include "console_out.h"
#include "linalg_PRECISION.h"
#include "operator.h"

#include <string.h>
#include "data_PRECISION.h"

#include "alloc_control.h"

void d_plus_clover_PRECISION(vector_PRECISION eta, vector_PRECISION phi,
                             operator_PRECISION_struct *op, level_struct *l,
                             struct Thread *threading)
{

  /*

  int start,end;
  vector_PRECISION v1=NULL,v2=NULL,v3=NULL;

  start = 0;
  end   = l->inner_vector_size;

  PUBLIC_MALLOC( v1, complex_PRECISION, l->inner_vector_size );
  PUBLIC_MALLOC( v2, complex_PRECISION, l->inner_vector_size );
  PUBLIC_MALLOC( v3, complex_PRECISION, l->inner_vector_size );

  cuda_d_plus_clover_PRECISION_vectorwrapper(v1, phi, op, l, threading);
  d_plus_clover_PRECISION_cpu(v2, phi, op, l, threading);

  vector_PRECISION_minus( v3, v1, v2, start, end, l );
  PRECISION norm1 = global_norm_PRECISION( v1, start, end, l, threading );
  PRECISION norm2 = global_norm_PRECISION( v2, start, end, l, threading );
  PRECISION norm3 = global_norm_PRECISION( v3, start, end, l, threading );

  START_MASTER(threading)
  //printf0("relative error = %f\n",norm1/norm2);
  printf0("norm1 = %f\n",norm1);
  printf0("norm2 = %f\n",norm2);
  printf0("norm3 = %f\n\n",norm3);
  //MPI_Finalize();
  //exit(0);
  END_MASTER(threading)

  */

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

  /*

  gmres_PRECISION_struct *p = &(l->sp_PRECISION);

  int start,end;
  vector_PRECISION v1=NULL,v2=NULL,b1=NULL;

  // create backup of p->b
  compute_core_start_end(p->v_start, p->v_end, &start, &end, l, threading);

  PUBLIC_MALLOC( v1, complex_PRECISION, l->inner_vector_size );
  PUBLIC_MALLOC( v2, complex_PRECISION, l->inner_vector_size );
  PUBLIC_MALLOC( b1, complex_PRECISION, l->inner_vector_size );

  SYNC_MASTER_TO_ALL(threading)
  SYNC_CORES(threading)

  //vector_PRECISION_copy( b1, p->b, start, end, l );
  //SYNC_MASTER_TO_ALL(threading)
  //SYNC_CORES(threading)

  //cuda_richardson_PRECISION_vectorwrapper( p, l, threading );
  cuda_apply_schur_complement_PRECISION_vectorwrapper( v1, in, op, l, threading );

  //vector_PRECISION_copy( v1, p->x, start, end, l );
  //SYNC_MASTER_TO_ALL(threading)
  //SYNC_CORES(threading)

  //vector_PRECISION_copy( p->b, b1, start, end, l );
  //SYNC_MASTER_TO_ALL(threading)
  //SYNC_CORES(threading)

  //richardson_PRECISION_cpu( p, l, threading );
  apply_schur_complement_PRECISION_cpu( v2, in, op, l, threading );

  //vector_PRECISION_copy( v2, p->x, start, end, l );
  //SYNC_MASTER_TO_ALL(threading)
  //SYNC_CORES(threading)

  vector_PRECISION_minus( v1, v1, v2, start, end, l );
  SYNC_MASTER_TO_ALL(threading)
  SYNC_CORES(threading)

  PRECISION norm1 = global_norm_PRECISION( v1, p->v_start, p->v_end, l, threading );
  PRECISION norm2 = global_norm_PRECISION( v2, p->v_start, p->v_end, l, threading );

  START_MASTER(threading)
  printf0("relative error = %f\n",norm1/norm2);
  MPI_Finalize();
  exit(0);
  END_MASTER(threading)

  */

#ifdef CUDA_OPT
  cuda_apply_schur_complement_PRECISION_vectorwrapper(out, in, op, l, threading);
#else
  apply_schur_complement_PRECISION_cpu(out, in, op, l, threading);
#endif
}
