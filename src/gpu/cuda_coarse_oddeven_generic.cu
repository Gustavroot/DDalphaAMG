#include <mpi.h>

extern "C"{

  #define IMPORT_FROM_EXTERN_C
  #include "main.h"
  #undef IMPORT_FROM_EXTERN_C

}

#ifdef CUDA_OPT


extern "C" void copy_2_cpu_PRECISION( vector_PRECISION out, cuda_vector_PRECISION out_gpu, vector_PRECISION in, cuda_vector_PRECISION in_gpu, level_struct *l){
  cudaStream_t *streams_gmres = (l->p_PRECISION).streams;

  cuda_vector_PRECISION_copy( (void*)out, (void*)out_gpu, 0, l->vector_size, l, _D2H, _CUDA_SYNC,
                              0, streams_gmres );
  cuda_vector_PRECISION_copy( (void*)in, (void*)in_gpu, 0, l->vector_size, l, _D2H, _CUDA_SYNC,
                              0, streams_gmres );
}


extern "C" void coarse_apply_schur_complement_PRECISION_CUDA( cuda_vector_PRECISION out_gpu, cuda_vector_PRECISION in_gpu, 
                                                              operator_PRECISION_struct *op, level_struct *l, struct Thread *threading ) {

  // TODO : finish all of the CUDA kernels associated to this function

  // start and end indices for vector functions depending on thread
  int start;
  int end;
  // compute start and end indices for core
  // this puts zero for all other hyperthreads, so we can call functions below with all hyperthreads
  compute_core_start_end(op->num_even_sites*l->num_lattice_site_var, l->inner_vector_size, &start, &end, l, threading);

  vector_PRECISION *tmp = op->buffer;

  vector_PRECISION out=NULL, in=NULL;
  out = (complex_PRECISION*) malloc( l->vector_size * sizeof(complex_PRECISION) );
  in = (complex_PRECISION*) malloc( l->vector_size * sizeof(complex_PRECISION) );

  cudaStream_t *streams_gmres = (l->p_PRECISION).streams;

  SYNC_CORES(threading)
  PROF_PRECISION_START( _SC, threading );

  // 1. 

  coarse_diag_ee_PRECISION_CUDA( out_gpu, in_gpu, op, l, threading );
  cudaDeviceSynchronize();

  copy_2_cpu_PRECISION(out, out_gpu, in, in_gpu, l);

  PROF_PRECISION_STOP( _SC, 0, threading );
  SYNC_CORES(threading)

  vector_PRECISION_define( tmp[0], 0, start, end, l );

  cudaDeviceSynchronize();

  SYNC_CORES(threading)
  PROF_PRECISION_START( _NC, threading );

  coarse_hopping_term_PRECISION( tmp[0], in, op, _ODD_SITES, l, threading );

  cudaDeviceSynchronize();

  PROF_PRECISION_STOP( _NC, 0, threading );
  PROF_PRECISION_START( _SC, threading );

  coarse_diag_oo_inv_PRECISION( tmp[1], tmp[0], op, l, threading );

  cudaDeviceSynchronize();

  PROF_PRECISION_STOP( _SC, 1, threading );
  PROF_PRECISION_START( _NC, threading );

  coarse_n_hopping_term_PRECISION( out, tmp[1], op, _EVEN_SITES, l, threading );

  cudaDeviceSynchronize();

  PROF_PRECISION_STOP( _NC, 1, threading );

  cuda_vector_PRECISION_copy( (void*)out_gpu, (void*)out, 0, l->vector_size, l, _H2D, _CUDA_SYNC,
                              threading->core, streams_gmres );
  cuda_vector_PRECISION_copy( (void*)in_gpu, (void*)in, 0, l->vector_size, l, _H2D, _CUDA_SYNC,
                              threading->core, streams_gmres );

  free(out);
  free(in);
}


extern "C" void coarse_self_couplings_PRECISION_CUDA( cuda_vector_PRECISION eta, cuda_vector_PRECISION phi, cuda_config_PRECISION clover, 
                                                      int length, level_struct *l, struct Thread *threading ){

  cudaStream_t *streams_gmres = (l->p_PRECISION).streams;

  int nr_threads, threads_per_cublock, num_eig_vect, offset;
  size_t tot_shared_mem;

  offset = l->num_lattice_site_var;

  num_eig_vect = offset/2;

  // Add some "padding" such that each CUDA block corresponds to a coarse-site,
  // but also such that its size is a multiple of g.warp_size
  if (num_eig_vect%g.warp_size == 0) {
    threads_per_cublock = num_eig_vect;
  } else {
    threads_per_cublock = (num_eig_vect/g.warp_size + 1) * g.warp_size;
  }

  // <end-start> (=length/offset) gives us, based on n1, the nr of sites per OpenMP thread
  nr_threads = length/offset;
  // and divided by 2 due to hermiticity of the matrix
  nr_threads *= num_eig_vect;

  tot_shared_mem = 1;
  // shared memory associated to clover
  tot_shared_mem += (offset*offset+offset)/2;
  // shared memory associated to buffer spinors (two spinors)
  tot_shared_mem += 2*offset;
  // and, add shared memory for the lower-triangular off-diagonal part in the two diagonal-blocks
  //tot_shared_mem += 2*((num_eig_vect*num_eig_vect-num_eig_vect)/2);
  tot_shared_mem *= sizeof(cu_cmplx_PRECISION);

  // even sites
  coarse_self_couplings_PRECISION_CUDA_kernel<<<nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams_gmres[threading->core]>>>
                                             (eta, phi, clover, offset);
}


extern "C" void coarse_diag_ee_PRECISION_CUDA( cuda_vector_PRECISION y, cuda_vector_PRECISION x, 
                                               operator_PRECISION_struct *op, level_struct *l, struct Thread *threading ) {

  int n1, start, end, offset;

  n1 = op->num_even_sites;
  compute_core_start_end_custom(0, n1, &start, &end, l, threading, 1);

  offset = l->num_lattice_site_var;

  coarse_self_couplings_PRECISION_CUDA( y+start*offset, x+start*offset, op->clover_gpu+start*(offset*offset+offset)/2, (end-start)*offset, l, threading );
}

#endif
