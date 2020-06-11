#include <mpi.h>

extern "C"{

  #define IMPORT_FROM_EXTERN_C
  #include "main.h"
  #undef IMPORT_FROM_EXTERN_C

}

#ifdef CUDA_OPT


// sites_to_solve = {_EVEN_SITES, _ODD_SITES, _FULL_SYSTEM}
extern "C" void local_minres_PRECISION_CUDA( cuda_vector_PRECISION phi, cuda_vector_PRECISION eta, cuda_vector_PRECISION latest_iter,
                                             schwarz_PRECISION_struct *s, level_struct *l, int nr_DD_blocks_to_compute,
                                             int* DD_blocks_to_compute, cudaStream_t *streams, int stream_id, int sites_to_solve ) {

  if( nr_DD_blocks_to_compute==0 ){ return; }

  // This local_minres performs an inversion on EVEN sites only

  int i, n = l->block_iter;
  cuda_vector_PRECISION Dr = (s->cu_s).local_minres_buffer[0];
  cuda_vector_PRECISION r = (s->cu_s).local_minres_buffer[1];
  cuda_vector_PRECISION lphi = (s->cu_s).local_minres_buffer[2];

  int threads_per_cublock, nr_threads, nr_threads_per_DD_block;
  size_t tot_shared_mem;

  // -*-*-*-*-* COPY r <----- eta (tunable! -- type2)

  // the use of _EVEN_SITES comes from the CPU code: end = (g.odd_even&&l->depth==0)?start+12*s->num_block_even_sites:start+s->block_vector_size
  //vector_PRECISION_copy( r, eta, start, end, l );
  nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
  nr_threads = nr_threads*12; // threads per site
  nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
  nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

  // it's important to accomodate for the factor of 3 in 4*3=12
  threads_per_cublock = 3 * g.CUDA_threads_per_CUDA_block_type2[0];

  cuda_block_oe_vector_PRECISION_copy_12threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                                  (r, eta, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute,
                                                   l->num_lattice_site_var, (s->cu_s).block, sites_to_solve);

  // -*-*-*-*-* DEFINE lphi <- (0.0,0.0)

  //vector_PRECISION_define( lphi, 0, start, end, l );
  nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
  nr_threads = nr_threads*(12/2); // threads per site
  nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
  nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
  threads_per_cublock = 96;
  cuda_block_oe_vector_PRECISION_define_6threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                                    (lphi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, \
                                                    l->num_lattice_site_var, (s->cu_s).block, sites_to_solve, make_cu_cmplx_PRECISION(0.0,0.0));

  for ( i=0; i<n; i++ ) {

    // Dr = blockD*r
    //block_op( Dr, r, start, s, l, no_threading );

    // -*-*-*-*-* SCHUR COMPLEMENT
    cuda_apply_block_schur_complement_PRECISION( Dr, r, s, l, nr_DD_blocks_to_compute, DD_blocks_to_compute, streams, stream_id, _EVEN_SITES );

    // -*-*-*-*-* LOCAL BLOCK SUMMATIONS xy/xx

    // To be able to call the current implementation of the dot product,
    // threads_per_cublock has to be a power of 2
    threads_per_cublock = 64;
    nr_threads = threads_per_cublock;
    nr_threads = nr_threads*nr_DD_blocks_to_compute;
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
    // buffer to store partial sums of the overall-per-DD-block dot product
    tot_shared_mem = 2*(threads_per_cublock)*sizeof(cu_cmplx_PRECISION);
    cuda_local_xy_over_xx_PRECISION<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>>
                                   ( Dr, r, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, sites_to_solve );

    // -*-*-*-*-* SAXPY

    PRECISION prefctr_alpha;

    prefctr_alpha = 1.0;
    // phi += alpha * r
    //vector_PRECISION_saxpy( lphi, lphi, r, alpha, start, end, l );
    nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
    nr_threads = nr_threads*(12/2); // threads per site
    nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
    threads_per_cublock = 96;
    cuda_block_oe_vector_PRECISION_saxpy_6threads_opt_onchip<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                                            (lphi, lphi, r, prefctr_alpha, (s->s_on_gpu_cpubuff).alphas, s->s_on_gpu, g.my_rank, g.csw, \
                                                            nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, sites_to_solve);

    prefctr_alpha = -1.0;
    // r -= alpha * Dr
    // vector_PRECISION_saxpy( r, r, Dr, -alpha, start, end, l );
    nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
    nr_threads = nr_threads*(12/2); // threads per site
    nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
    threads_per_cublock = 96;
    cuda_block_oe_vector_PRECISION_saxpy_6threads_opt_onchip<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                                            (r, r, Dr, prefctr_alpha, (s->s_on_gpu_cpubuff).alphas, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, \
                                                            DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, sites_to_solve);

  }

  // -*-*-*-*-* COPY latest_iter <- lphi (tunable! -- type2)

  //vector_PRECISION_copy( latest_iter, lphi, start, end, l );
  if ( latest_iter != NULL ){
    nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
    nr_threads = nr_threads*12; // threads per site
    nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

    // it's important to accomodate for the factor of 3 in 4*3=12
    threads_per_cublock = 3 * g.CUDA_threads_per_CUDA_block_type2[0];

    cuda_block_oe_vector_PRECISION_copy_12threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                                    (latest_iter, lphi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute,
                                                     l->num_lattice_site_var, (s->cu_s).block, sites_to_solve);
  }

  // -*-*-*-*-* PLUS

  //vector_PRECISION_plus( phi, phi, lphi, start, end, l );
  if ( phi != NULL ){
    nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
    nr_threads = nr_threads*(12/2); // threads per site
    nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
    threads_per_cublock = 96;
    cuda_block_oe_vector_PRECISION_plus_6threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                                    (phi, phi, lphi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute,
                                                     l->num_lattice_site_var, (s->cu_s).block, sites_to_solve);
  }

  // -*-*-*-*-* COPY eta <- r (tunable! -- type2)

  //vector_PRECISION_copy( eta, r, start, end, l );
  nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
  nr_threads = nr_threads*12; // threads per site
  nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
  nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

  // it's important to accomodate for the factor of 3 in 4*3=12
  threads_per_cublock = 3 * g.CUDA_threads_per_CUDA_block_type2[0];

  cuda_block_oe_vector_PRECISION_copy_12threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                                  (eta, r, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute,
                                                   l->num_lattice_site_var, (s->cu_s).block, sites_to_solve);

}

#endif
