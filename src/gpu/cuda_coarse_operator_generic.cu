#include <mpi.h>

extern "C"{

  #define IMPORT_FROM_EXTERN_C
  #include "main.h"
  #undef IMPORT_FROM_EXTERN_C

}

#ifdef CUDA_OPT


extern "C" void copy_2_cpu_PRECISION_v2( vector_PRECISION out, cuda_vector_PRECISION out_gpu, vector_PRECISION in, cuda_vector_PRECISION in_gpu, level_struct *l){
  cudaStream_t *streams_gmres = (l->p_PRECISION).streams;

  cuda_vector_PRECISION_copy( (void*)out, (void*)out_gpu, 0, l->vector_size, l, _D2H, _CUDA_SYNC,
                              0, streams_gmres );
  cuda_vector_PRECISION_copy( (void*)in, (void*)in_gpu, 0, l->vector_size, l, _D2H, _CUDA_SYNC,
                              0, streams_gmres );
}


__global__ void coarse_self_couplings_PRECISION_CUDA_kernel( cuda_vector_PRECISION eta, cuda_vector_PRECISION phi, cuda_config_PRECISION clover, int num_lattice_site_var ) {

  int pipe_width;
  int i,j;

  int site_var = num_lattice_site_var,
      num_eig_vect = site_var/2,
      clover_step_size1 = (num_eig_vect * (num_eig_vect+1))/2,
      clover_step_size2 = SQUARE(num_lattice_site_var/2);

  cuda_config_PRECISION clover_pt = clover;
  cuda_vector_PRECISION phi_pt=phi, eta_pt=eta;

  // U(x) = [ A B      , A=A*, D=D*, C = -B*
  //          C D ]
  // storage order: upper triangle of A, upper triangle of D, B, columnwise
  // diagonal coupling

  // PART 1 : setting up variables to work within the corresponding CUDA block

  phi_pt    += blockIdx.x*site_var;
  eta_pt    += blockIdx.x*site_var;
  clover_pt += blockIdx.x*(2*clover_step_size1 + clover_step_size2);

  // PART 2 : retrieving memory into __shared__, collaborative effort by ALL the threads within the CUDA block

  extern __shared__ cu_cmplx_PRECISION shared_data[];

  cuda_vector_PRECISION phi_shared    = (cuda_vector_PRECISION)((cuda_vector_PRECISION)shared_data + 0*site_var);
  cuda_vector_PRECISION eta_shared    = (cuda_vector_PRECISION)((cuda_vector_PRECISION)shared_data + 1*site_var);
  cuda_config_PRECISION clover_shared = (cuda_config_PRECISION)((cuda_vector_PRECISION)shared_data + 2*site_var);

  // loading phi into shared memory
  i = 0;
  pipe_width = site_var;
  while (i*blockDim.x < pipe_width) {
    j = i*blockDim.x + threadIdx.x;
    if (j<pipe_width) phi_shared[j] = phi_pt[j];
    i++;
  }

  // loading eta=0 into shared memory
  i = 0;
  pipe_width = site_var;
  while (i*blockDim.x < pipe_width) {
    j = i*blockDim.x + threadIdx.x;
    //if (j<pipe_width) eta_shared[j] = eta_pt[j];
    if (j<pipe_width) eta_shared[j] = make_cu_cmplx_PRECISION(0.0,0.0);
    i++;
  }

  // loading clover into shared memory
  i = 0;
  pipe_width = 2*clover_step_size1 + clover_step_size2;
  while (i*blockDim.x < pipe_width) {
    j = i*blockDim.x + threadIdx.x;
    if (j<pipe_width) clover_shared[j] = clover_pt[j];
    i++;
  }

  // sync after loading into shared memory
  __syncthreads();

  // PART 3 : computations within each CUDA block, only a subset of the threads do work

  if (threadIdx.x < num_eig_vect) {

    // A
    //mvp_PRECISION( eta_pt, clover_pt, phi_pt, num_eig_vect );
    mvp_PRECISION_CUDA( eta_shared, clover_shared, phi_shared, num_eig_vect, threadIdx.x );

    clover_shared += clover_step_size1; eta_shared += num_eig_vect; phi_shared += num_eig_vect;

    // D
    //mvp_PRECISION( eta_pt, clover_pt, phi_pt, num_eig_vect );
    mvp_PRECISION_CUDA( eta_shared, clover_shared, phi_shared, num_eig_vect, threadIdx.x );

    clover_shared += clover_step_size1; phi_shared -= num_eig_vect;

    // C = -B*
    //nmvh_PRECISION( eta_pt, clover_pt, phi_pt, num_eig_vect );
    nmvh_PRECISION_CUDA( eta_shared, clover_shared, phi_shared, num_eig_vect, threadIdx.x );

    phi_shared += num_eig_vect; eta_shared -= num_eig_vect;

    // B
    //mv_PRECISION( eta_pt, clover_pt, phi_pt, num_eig_vect );
    mv_PRECISION_CUDA( eta_shared, clover_shared, phi_shared, num_eig_vect, threadIdx.x );
  }

  // sync before loading back into global memory
  __syncthreads();

  // PART 4 : putting results into global memory, collaborative effort by ALL the threads within the CUDA block

  i = 0;
  pipe_width = site_var;
  while (i*blockDim.x < pipe_width) {
    j = i*blockDim.x + threadIdx.x;
    if (j<pipe_width) eta_pt[j] = eta_shared[j];
    i++;
  }
}


/*
extern "C" void apply_coarse_operator_PRECISION_CUDA( cuda_vector_PRECISION eta_gpu, cuda_vector_PRECISION phi_gpu,
                                                      operator_PRECISION_struct *op, level_struct *l, struct Thread *threading ) {

  cudaStream_t *streams_gmres = (l->p_PRECISION).streams;

  vector_PRECISION eta=NULL, phi=NULL;
  eta = (complex_PRECISION*) malloc( l->vector_size * sizeof(complex_PRECISION) );
  phi = (complex_PRECISION*) malloc( l->vector_size * sizeof(complex_PRECISION) );

  copy_2_cpu_PRECISION_v2(eta, eta_gpu, phi, phi_gpu, l);

  PROF_PRECISION_START( _SC, threading );
  START_LOCKED_MASTER(threading)

  coarse_self_couplings_PRECISION( eta, phi, op->clover, l->inner_vector_size, l );

  //coarse_self_couplings_PRECISION_CUDA( y_start*offset, x+start*offset, op->clover_gpu+start*(offset*offset+offset)/2, (end-start)*offset, l, threading );
  //coarse_self_couplings_PRECISION_CUDA( eta_gpu, phi_gpu, op->clover_gpu, l->inner_vector_size, l, threading );

  //cudaDeviceSynchronize();

  //copy_2_cpu_PRECISION_v2(eta, eta_gpu, phi, phi_gpu, l);

  END_LOCKED_MASTER(threading)
  PROF_PRECISION_STOP( _SC, 1, threading );
  PROF_PRECISION_START( _NC, threading );
  coarse_hopping_term_PRECISION( eta, phi, op, _FULL_SYSTEM, l, threading );
  PROF_PRECISION_STOP( _NC, 1, threading );

  cuda_vector_PRECISION_copy( (void*)eta_gpu, (void*)eta, 0, l->vector_size, l, _H2D, _CUDA_SYNC,
                              threading->core, streams_gmres );
  cuda_vector_PRECISION_copy( (void*)phi_gpu, (void*)phi, 0, l->vector_size, l, _H2D, _CUDA_SYNC,
                              threading->core, streams_gmres );

  free(eta);
  free(phi);
}
*/


#endif
