#include "linsolve_PRECISION.h"
#include "linsolve_proxy_PRECISION.h"
#include "alloc_control.h"
#include "linalg_PRECISION.h"
#include "gpu/cuda_linalg_PRECISION.h"

// included for debugging purposes
#include "data_PRECISION.h"


complex_PRECISION global_inner_product_PRECISION( vector_PRECISION* V, vector_PRECISION psi, complex_PRECISION *result,
                  int n, int start, int end, gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading ) {

  // we have generalized this to be a multi inner dot product by default

  //return global_inner_product_PRECISION_cpu( V, psi, result, n, start, end, l, threading );

  /*

//#ifdef CUDA_OPT
  complex_PRECISION *resultx = NULL;
  PUBLIC_MALLOC( resultx, complex_PRECISION, 1 );
  cuda_global_inner_product_PRECISION_vectorwrapper( V, psi, resultx, n, start, end, l, threading );
  complex_PRECISION resulty = resultx[0];
  PUBLIC_FREE( resultx, complex_PRECISION, 1 );
  //return resulty;
//#else
  //return 
  complex_PRECISION resultz = global_inner_product_PRECISION_cpu( V, psi, result, n, start, end, l, threading );

  START_MASTER(threading)
  PRECISION rel_err = cabs_PRECISION(resulty-resultz)/cabs_PRECISION(resultz);
  printf0( "resulty = %.15f+%.15f, resultz = %.15f+%.15f, relative = %e\n", CSPLIT(resulty), CSPLIT(resultz), rel_err );
  END_MASTER(threading)
  SYNC_CORES(threading)

  return resulty;

//#endif

  */

  int i;
  complex_PRECISION *result_gpu = NULL;
  PUBLIC_MALLOC( result_gpu, complex_PRECISION, p->restart_length );
  complex_PRECISION result_cpu[p->restart_length];
  vector_PRECISION V_cpu[p->restart_length];
  for ( i=0;i<p->restart_length;i++ ) {
    V_cpu[i] = NULL;
    PUBLIC_MALLOC( V_cpu[i], complex_PRECISION, l->inner_vector_size );
    START_MASTER(threading)
    vector_PRECISION_define_random( V_cpu[i], 0, l->inner_vector_size, l );
    END_MASTER(threading)
  }
  SYNC_CORES(threading)

  cuda_global_inner_product_PRECISION_vectorwrapper( V_cpu, psi, result_gpu, p->restart_length, start, end, p, l, threading );

  //for ( i=0;i<p->restart_length;i++ ) {
  //  result_cpu[i] = global_inner_product_PRECISION_cpu( V_cpu+i, psi, result_cpu, 1, start, end, p, l, threading );
  //}

  for ( i=0;i<p->restart_length;i++ ) {
    PUBLIC_FREE( V_cpu[i], complex_PRECISION, l->inner_vector_size );
  }

  //for ( i=0;i<p->restart_length;i++ ) {
  //  printf0("%f+%f, %f+%f, %e\n", CSPLIT(result_cpu[i]), CSPLIT(result_gpu[i]), cabs_PRECISION(result_cpu[i]-result_gpu[i])/cabs_PRECISION(result_cpu[i]));
  //}

  PUBLIC_FREE( result_gpu, complex_PRECISION, p->restart_length );

  //START_MASTER(threading)
  //MPI_Finalize();
  //exit(0);
  //END_MASTER(threading)
  //SYNC_CORES(threading)
}
