#include "linsolve_PRECISION.h"
#include "linsolve_proxy_PRECISION.h"
#include "alloc_control.h"
#include "linalg_PRECISION.h"

#ifdef CUDA_OPT
#include "gpu/cuda_linsolve_PRECISION.h"
#endif

#ifdef RICHARDSON_SMOOTHER
#include "linsolve_PRECISION.h"
#endif

void fgmres_PRECISION_struct_init(gmres_PRECISION_struct *p) {
#ifdef CUDA_OPT
  cuda_fgmres_PRECISION_struct_init(p);
#endif
  cpu_fgmres_PRECISION_struct_init(p);
}

void fgmres_PRECISION_struct_alloc(int m, int n, int vl, PRECISION tol, const int type,
                                   const int prec_kind, void (*precond)(), void (*eval_op)(),
                                   gmres_PRECISION_struct *p, level_struct *l) {
#ifdef CUDA_OPT
  cuda_fgmres_PRECISION_struct_alloc(m, n, vl, tol, type, prec_kind, precond, eval_op, p, l);
#endif
  cpu_fgmres_PRECISION_struct_alloc(m, n, vl, tol, type, prec_kind, precond, eval_op, p, l);
}

void fgmres_PRECISION_struct_free(gmres_PRECISION_struct *p, level_struct *l) {
#ifdef CUDA_OPT
  cuda_fgmres_PRECISION_struct_free(p, l);
#endif
  cpu_fgmres_PRECISION_struct_free(p, l);
}

#ifdef RICHARDSON_SMOOTHER
int richardson_PRECISION( gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading ) {

  int start,end;
  vector_PRECISION v1=NULL,v2=NULL,b1=NULL;

  // create backup of p->b
  compute_core_start_end(p->v_start, p->v_end, &start, &end, l, threading);

  PUBLIC_MALLOC( v1, complex_PRECISION, l->inner_vector_size );
  PUBLIC_MALLOC( v2, complex_PRECISION, l->inner_vector_size );
  PUBLIC_MALLOC( b1, complex_PRECISION, l->inner_vector_size );

  SYNC_MASTER_TO_ALL(threading)
  SYNC_CORES(threading)

  vector_PRECISION_copy( b1, p->b, start, end, l );
  SYNC_MASTER_TO_ALL(threading)
  SYNC_CORES(threading)

  cuda_richardson_PRECISION_vectorwrapper( p, l, threading );

  vector_PRECISION_copy( v1, p->x, start, end, l );
  SYNC_MASTER_TO_ALL(threading)
  SYNC_CORES(threading)

  vector_PRECISION_copy( p->b, b1, start, end, l );
  SYNC_MASTER_TO_ALL(threading)
  SYNC_CORES(threading)

  richardson_PRECISION_cpu( p, l, threading );

  vector_PRECISION_copy( v2, p->x, start, end, l );
  SYNC_MASTER_TO_ALL(threading)
  SYNC_CORES(threading)

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

//#ifdef CUDA_OPT
//  return cuda_richardson_PRECISION_vectorwrapper( p, l, threading );
//#else
//  return richardson_PRECISION_cpu( p, l, threading );
//#endif

  //START_MASTER(threading)
  //FREE( v1, complex_PRECISION, l->inner_vector_size );
  //FREE( v1, complex_PRECISION, l->inner_vector_size );
  //FREE( b1, complex_PRECISION, l->inner_vector_size );
  //END_MASTER(threading)
}
#endif
