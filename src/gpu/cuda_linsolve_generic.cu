#include <mpi.h>

#include "gpu/cuda_componentwise.h"

// this block size is for the full-size Schur complement
#ifdef RICHARDSON_SMOOTHER
constexpr uint diracDefaultBlockSize = 128;
#endif

extern "C"{

#define IMPORT_FROM_EXTERN_C
#include "main.h"
#undef IMPORT_FROM_EXTERN_C

#include "linsolve_PRECISION.h"

void cuda_fgmres_PRECISION_struct_init(gmres_PRECISION_struct* p) {
  p->xtmp = NULL;
  p->streams = NULL;
}

void cuda_fgmres_PRECISION_struct_alloc(int m, int n, int vl, PRECISION tol, const int type,
                                   const int prec_kind, void (*precond)(), void (*eval_op)(),
                                   gmres_PRECISION_struct *p, level_struct *l) {
  ASSERT(g.nr_threads > 0);
  MALLOC( p->streams, cudaStream_t, g.nr_threads );
  for(size_t i=0; i < static_cast<size_t>(g.nr_threads); i++) {
    cuda_safe_call( cudaStreamCreate( &(p->streams[i]) ) );
  }
}


void cuda_fgmres_PRECISION_struct_free(gmres_PRECISION_struct *p, level_struct *l) {
  if( l->depth==0){
    cuda_safe_call( cudaFreeHost( l->p_PRECISION.xtmp ) );
  }
  FREE( p->streams, cudaStream_t, g.nr_threads );
}

// sites_to_solve = {_EVEN_SITES, _ODD_SITES, _FULL_SYSTEM}
 void local_minres_PRECISION_CUDA( cuda_vector_PRECISION phi, cuda_vector_PRECISION eta, cuda_vector_PRECISION latest_iter,
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

  // -*-*-*-*-* DEFINE lphi <- (0.0,0.0) (tunable! -- type2)

  //vector_PRECISION_define( lphi, 0, start, end, l );

  nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
  nr_threads = nr_threads * 12; // threads per site
  nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
  nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

  threads_per_cublock = 3 * g.CUDA_threads_per_CUDA_block_type2[0];

  cuda_block_oe_vector_PRECISION_define_12threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                                     (lphi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, \
                                                     l->num_lattice_site_var, (s->cu_s).block, sites_to_solve, make_cu_cmplx_PRECISION(0.0,0.0));

  for ( i=0; i<n; i++ ) {

    // Dr = blockD*r
    //block_op( Dr, r, start, s, l, no_threading );

    // -*-*-*-*-* SCHUR COMPLEMENT
    cuda_apply_block_schur_complement_PRECISION( Dr, r, s, l, nr_DD_blocks_to_compute, DD_blocks_to_compute, streams, stream_id, _EVEN_SITES );

    // -*-*-*-*-* LOCAL BLOCK SUMMATIONS xy/xx (tunable! -- type2)

    // To be able to call the current implementation of the dot product,
    // threads_per_cublock has to be a power of 2

    threads_per_cublock = g.CUDA_threads_per_CUDA_block_type2[0];

    nr_threads = threads_per_cublock;
    nr_threads = nr_threads*nr_DD_blocks_to_compute;
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

    // buffer to store partial sums of the overall-per-DD-block dot product

    tot_shared_mem = 2*(threads_per_cublock)*sizeof(cu_cmplx_PRECISION);

    cuda_local_xy_over_xx_PRECISION<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>>
                                   ( Dr, r, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, sites_to_solve );

    // -*-*-*-*-* SAXPY (tunable! -- type2)

    PRECISION prefctr_alpha;

    prefctr_alpha = 1.0;
    // phi += alpha * r
    //vector_PRECISION_saxpy( lphi, lphi, r, alpha, start, end, l );

    nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
    nr_threads = nr_threads * 12; // threads per site
    nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

    threads_per_cublock = 3 * g.CUDA_threads_per_CUDA_block_type2[0];

    cuda_block_oe_vector_PRECISION_saxpy_12threads_opt_onchip<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                                             (lphi, lphi, r, prefctr_alpha, (s->s_on_gpu_cpubuff).alphas, s->s_on_gpu, g.my_rank, g.csw, \
                                                             nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, sites_to_solve);

    prefctr_alpha = -1.0;
    // r -= alpha * Dr
    // vector_PRECISION_saxpy( r, r, Dr, -alpha, start, end, l );

    nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
    nr_threads = nr_threads * 12; // threads per site
    nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

    threads_per_cublock = 3 * g.CUDA_threads_per_CUDA_block_type2[0];

    cuda_block_oe_vector_PRECISION_saxpy_12threads_opt_onchip<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
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

  // -*-*-*-*-* PLUS (tunable! -- type2)

  //vector_PRECISION_plus( phi, phi, lphi, start, end, l );

  if ( phi != NULL ){

    nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
    nr_threads = nr_threads * 12; // threads per site
    nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

    threads_per_cublock = 3 * g.CUDA_threads_per_CUDA_block_type2[0];

    cuda_block_oe_vector_PRECISION_plus_12threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
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

}

#ifdef RICHARDSON_SMOOTHER
int cuda_richardson_PRECISION( gmres_PRECISION_struct *p, level_struct *l,
                               struct Thread *threading ) {

  cudaStream_t stream = CU_STREAM_PER_THREAD;
  cudaStream_t* const streams = &stream;

  int start, end, i;
  start = p->v_start;
  end = p->v_end;
  int n = p->num_restart * p->restart_length;

  cuda_vector_PRECISION x, w, b, r;
  x = p->x_componentwise_gpu;
  w = p->w_componentwise_gpu;
  b = p->b_componentwise_gpu;
  r = p->r_componentwise_gpu;

  // initial guess to zero if necessary
  if ( p->initial_guess_zero == _NO_RES ) {
    cuda_vector_PRECISION_define( x, make_cu_cmplx_PRECISION(0,0), start,
                                  end, l, _CUDA_SYNC, 0, streams );
  }

  for ( i=0; i<n; i++ ) {
    // 1. compute residual
    if ( i==0 && p->initial_guess_zero==_NO_RES ) {
      cuda_vector_PRECISION_copy( r, b, start, end-start, l, _D2D, _CUDA_SYNC, 0, streams );
    } else {
      cuda_apply_schur_complement_PRECISION( w, x, p->op, l );
      cuda_vector_PRECISION_minus( r, b, w, start, end, l, _CUDA_SYNC, 0, streams );
    }

    // 2. update solution
    cu_cmplx_PRECISION om_fctr = make_cu_cmplx_PRECISION(p->omega[i%p->richardson_sub_degree],0.0);
    cuda_vector_PRECISION_saxpy( x, x, r, om_fctr, start, end, l, _CUDA_SYNC, 0, streams );
  }

  return n;
}

extern "C" int cuda_richardson_PRECISION_vectorwrapper( gmres_PRECISION_struct *p, level_struct *l,
                                                        struct Thread *threading ) {

  if ( p->richardson_update_omega==1 ) {
    richardson_update_omega_PRECISION( p, l, threading );
    START_MASTER(threading)
    p->richardson_update_omega = 0;
    END_MASTER(threading)
  }

  START_MASTER(threading)

  //operator_PRECISION_struct *op = p->op;
  operator_PRECISION_struct *op = &(l->oe_op_PRECISION);

  // CUDA stream, only one as only the master thread is in charge of this
  cudaStream_t stream = CU_STREAM_PER_THREAD;
  cudaStream_t* const streams = &stream;

  // labels for certain vectors, and assignments for in/out in a CUDA sense
  cuda_vector_PRECISION b_gpu, b_componentwise_gpu, x_gpu, x_componentwise_gpu;
  b_gpu = p->b_gpu;
  b_componentwise_gpu = p->b_componentwise_gpu;
  x_gpu = p->x_gpu;
  x_componentwise_gpu = p->x_componentwise_gpu;

  // copy from CPU to GPU the input vector
  cuda_vector_PRECISION_copy(b_gpu, p->b, 0, l->num_inner_lattice_sites*l->num_lattice_site_var, l, _H2D,
                             _CUDA_SYNC, 0, streams);

  // re-order the input vector in component-wise ordering
  uint gridSize = minGridSizeForN( op->num_even_sites, diracDefaultBlockSize );
  reorderArrayByComponent<<<gridSize, diracDefaultBlockSize>>>(
    b_componentwise_gpu, b_gpu, l->num_lattice_site_var, op->num_even_sites);
  gridSize = minGridSizeForN( op->num_odd_sites, diracDefaultBlockSize );
  reorderArrayByComponent<<<gridSize, diracDefaultBlockSize>>>(
    b_componentwise_gpu+l->num_lattice_site_var*op->num_even_sites, b_gpu+l->num_lattice_site_var*op->num_even_sites,
    l->num_lattice_site_var, op->num_odd_sites);
  cuda_safe_call(cudaDeviceSynchronize());

  cuda_richardson_PRECISION( p, l, threading );

  // re-order the output back to chuck-wise ordering
  gridSize = minGridSizeForN( op->num_even_sites, diracDefaultBlockSize );
  reorderArrayByChunks<<<gridSize, diracDefaultBlockSize>>>(
    x_gpu, x_componentwise_gpu, l->num_lattice_site_var, op->num_even_sites);
  gridSize = minGridSizeForN( op->num_odd_sites, diracDefaultBlockSize );
  reorderArrayByChunks<<<gridSize, diracDefaultBlockSize>>>(
    x_gpu+l->num_lattice_site_var*op->num_even_sites, x_componentwise_gpu+l->num_lattice_site_var*op->num_even_sites,
    l->num_lattice_site_var, op->num_odd_sites);
  cuda_safe_call(cudaDeviceSynchronize());

  cuda_vector_PRECISION_copy( p->x, x_gpu, 0, l->inner_vector_size, l, _D2H, _CUDA_SYNC, 0, streams );

  END_MASTER(threading)
  SYNC_CORES(threading)

  return p->num_restart * p->restart_length;
}
#endif
