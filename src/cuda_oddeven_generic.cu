#ifdef CUDA_OPT

#include <mpi.h>

#ifdef CUDA_OPT
  #include <cuComplex.h>
  #include <cuda.h>
  #include <cuda_runtime.h>
#endif

extern "C"{

  #define IMPORT_FROM_EXTERN_C
  #include "main.h"
  #undef IMPORT_FROM_EXTERN_C

}


__global__ void cuda_block_solve_oddeven_PRECISION_dev( cuda_vector_PRECISION phi, cuda_vector_PRECISION r, cuda_vector_PRECISION latest_iter, \
                                                   cuda_schwarz_PRECISION_struct *s, int thread_id, int csw, int nr_DD_blocks_to_compute, \
                                                   int num_block_even_sites, int num_latt_site_var, int stream_id ){ }


extern "C" void cuda_block_solve_oddeven_PRECISION( cuda_vector_PRECISION phi, cuda_vector_PRECISION r, cuda_vector_PRECISION latest_iter,
                                                    int start, int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s, level_struct *l,
                                                    struct Thread *threading, int stream_id, cudaStream_t *streams ) {

  block_solve_oddeven_PRECISION( (vector_PRECISION)phi, (vector_PRECISION)r, (vector_PRECISION)latest_iter,
                                 start, s, l, threading );

  // Num even block sites
  // TODO: change for general values
  //int nr_DD_sites = 4*4*4*4;

  // Call to the CUDA kernel to compute solves on a sub-set of this process's DD blocks
  //cuda_block_solve_oddeven_PRECISION_dev <<< nr_DD_blocks_to_compute, 32, nr_DD_blocks_to_compute*sizeof(int), streams[stream_id] >>> 
  //                                       ( phi, r, latest_iter, &(s->cu_s), threading->core, g.csw, 30, nr_DD_sites/2, l->num_lattice_site_var, stream_id );

}


#endif
