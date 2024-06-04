#include <mpi.h>

#include "gpu/cuda_componentwise.h"
#include "gpu/cuda_linalg_PRECISION.h"

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

// to be used from within GPU GMRES
void cuda_qr_update_PRECISION( complex_PRECISION **H, complex_PRECISION *s,
                               complex_PRECISION *c, complex_PRECISION *gamma, int j,
                               level_struct *l, struct Thread *threading ) {

/*********************************************************************************
* Applies one Givens rotation to the Hessenberg matrix H in order to solve the 
* least squares problem in (F)GMRES for computing the solution.
* - complex_PRECISION **H: Hessenberg matrix from Arnoldi decomposition
* - complex_PRECISION *s: sin values from givens rotations
* - complex_PRECISION *c: cos valies from givens rotations
* - complex_PRECISION *gamma: Approximation to residual from every step
* - int j: Denotes current iteration.
*********************************************************************************/  

  //PROF_PRECISION_START_UNTHREADED( _SMALL1 );

  int i;
  complex_PRECISION beta;

  // update QR factorization
  // apply previous Givens rotation
  for ( i=0; i<j; i++ ) {
    beta = (-s[i])*H[j][i] + (c[i])*H[j][i+1];
    H[j][i] = conj_PRECISION(c[i])*H[j][i] + conj_PRECISION(s[i])*H[j][i+1];
    H[j][i+1] = beta;
  }
  // compute current Givens rotation
  beta = (complex_PRECISION) sqrt( NORM_SQUARE_PRECISION(H[j][j]) + NORM_SQUARE_PRECISION(H[j][j+1]) );
  s[j] = H[j][j+1]/beta; c[j] = H[j][j]/beta;
  // update right column
  gamma[j+1] = (-s[j])*gamma[j]; gamma[j] = conj_PRECISION(c[j])*gamma[j];
  // apply current Givens rotation
  H[j][j] = beta; H[j][j+1] = 0;

  //PROF_PRECISION_STOP_UNTHREADED( _SMALL1, 6*j+6 );
}

void cuda_compute_solution_PRECISION( cuda_vector_PRECISION x, cuda_vector_PRECISION *V, complex_PRECISION *y,
                                      complex_PRECISION *gamma, complex_PRECISION **H, int j, int ol,
                                      gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading ) {
  
  int i, k;
  // start and end indices for vector functions depending on thread
  int start;
  int end;

  cudaStream_t stream = CU_STREAM_PER_THREAD;
  cudaStream_t* const streams = &stream;

  start = p->v_start;
  end = p->v_end;

  //PROF_PRECISION_START( _SMALL2 );

  // backward substitution
  for ( i=j; i>=0; i-- ) {
    y[i] = gamma[i];
    for ( k=i+1; k<=j; k++ ) {
      y[i] -= H[k][i]*y[k];
    }
    y[i] /= H[i][i];
  }

  //PROF_PRECISION_STOP( _SMALL2, ((j+1)*(j+2))/2 + j+1 );

  // x = x + V*y
  if ( ol ) {
    for ( i=0; i<=j; i++ ) {
      cuda_vector_PRECISION_saxpy( x, x, V[i], make_cu_cmplx_PRECISION(creal_PRECISION(y[i]),cimag_PRECISION(y[i])),
                                   start, end, l, _CUDA_SYNC, 0, streams );
    }
  } else {
    cuda_vector_PRECISION_scale( x, V[0], make_cu_cmplx_PRECISION(creal_PRECISION(y[0]),cimag_PRECISION(y[0])),
                                 start, end, l, _CUDA_SYNC, 0, streams);
    for ( i=1; i<=j; i++ ) {
      cuda_vector_PRECISION_saxpy( x, x, V[i], make_cu_cmplx_PRECISION(creal_PRECISION(y[i]),cimag_PRECISION(y[i])),
                                   start, end, l, _CUDA_SYNC, 0, streams );
    }
  }
}

int cuda_arnoldi_step_PRECISION( cuda_vector_PRECISION *V, cuda_vector_PRECISION *Z, cuda_vector_PRECISION w,
                                 complex_PRECISION **H, complex_PRECISION* buffer, int j, void (*prec)(),
                                 complex_PRECISION shift, gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading ) {

  int i;
  int start, end;

  start = p->v_start;
  end = p->v_end;

  cudaStream_t stream = CU_STREAM_PER_THREAD;
  cudaStream_t* const streams = &stream;

  cuda_apply_schur_complement_PRECISION( w, V[j], p->op, l );
  if ( shift ) cuda_vector_PRECISION_saxpy( w, w, V[j], make_cu_cmplx_PRECISION(creal_PRECISION(shift),cimag_PRECISION(shift)),
                                            start, end, l, _CUDA_SYNC, 0, streams );

  cuda_global_inner_product_PRECISION( V, w, H[j], j+1, start, end, p, l, threading );

  for( i=0; i<=j; i++ ) {
    cuda_vector_PRECISION_saxpy( w, w, V[i], make_cu_cmplx_PRECISION(-creal_PRECISION(H[j][i]),cimag_PRECISION(H[j][i])),
                                 start, end, l, _CUDA_SYNC, 0, streams );
  }

  complex_PRECISION tmp2;
  cuda_global_inner_product_PRECISION( &w, w, &tmp2, 1, start, end, p, l, threading );
  tmp2 = (complex_PRECISION)sqrt(creal_PRECISION(tmp2));
  H[j][j+1] = tmp2;

  // V_j+1 = w / H_j+1,j
  if ( cabs_PRECISION( H[j][j+1] ) > 1e-15 ) {
    cuda_vector_PRECISION_scale( V[j+1], w, make_cu_cmplx_PRECISION(1.0/creal_PRECISION(H[j][j+1]),0.0),
                                 start, end, l, _CUDA_SYNC, 0, streams);
  }

  return 1;
}

int cuda_fgmres_PRECISION( gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading ) {

  // start and end indices for vector functions depending on thread
  int start, end;

  int j=-1, finish=0, iter=0, il, ol, res;
  complex_PRECISION gamma0 = 0;
  complex_PRECISION beta = 0;

  cudaStream_t stream = CU_STREAM_PER_THREAD;
  cudaStream_t* const streams = &stream;

  double norm_r0=1, gamma_jp1=1, t0=0, t1=0;

  cuda_vector_PRECISION x  = p->x_componentwise_gpu;
  cuda_vector_PRECISION r  = p->r_componentwise_gpu;
  cuda_vector_PRECISION b  = p->b_componentwise_gpu;
  cuda_vector_PRECISION w  = p->w_componentwise_gpu;
  // TODO : allocate the memory for these V vectors
  cuda_vector_PRECISION *V = p->V_componentwise_gpu;

  start = p->v_start;
  end = p->v_end;

  for( ol=0; ol<p->num_restart && finish==0; ol++ )  {

    if( ol == 0 && p->initial_guess_zero ) {
      res = _NO_RES;
      cuda_vector_PRECISION_copy( r, b, start, end-start, l, _D2D, _CUDA_SYNC, 0, streams );
    } else {
      res = _RES;
      cuda_apply_schur_complement_PRECISION( w, x, p->op, l );
      cuda_vector_PRECISION_minus( r, b, w, start, end, l, _CUDA_SYNC, 0, streams );
    }

    cuda_global_inner_product_PRECISION( &r, r, &gamma0, 1, start, end, p, l, threading );
    gamma0 = (complex_PRECISION)sqrt(creal_PRECISION(gamma0));
    p->gamma[0] = gamma0;

    if( ol == 0) {
      norm_r0 = creal(p->gamma[0]);
    }

    cuda_vector_PRECISION_scale( V[0], r, make_cu_cmplx_PRECISION(1.0/creal_PRECISION(p->gamma[0]),0.0),
                                 start, end, l, _CUDA_SYNC, 0, streams);

    for( il=0; il<p->restart_length && finish==0; il++) {

      j = il; iter++;

      // one step of Arnoldi
      cuda_arnoldi_step_PRECISION( V, NULL, w, p->H, p->y, j, NULL, p->shift, p, l, threading );
      
      if ( cabs( p->H[j][j+1] ) > p->tol/10 ) {
        cuda_qr_update_PRECISION( p->H, p->s, p->c, p->gamma, j, l, threading );
        gamma_jp1 = cabs( p->gamma[j+1] );

        if( gamma_jp1/norm_r0 < p->tol || gamma_jp1/norm_r0 > 1E+5 ) { // if satisfied ... stop
          finish = 1;
          if ( gamma_jp1/norm_r0 > 1E+5 ) printf0("Divergence of fgmres_PRECISION, iter = %d, level=%d\n", iter, l->level );
        }
      } else {
        printf0("depth: %d, iter: %d, p->H(%d,%d) = %+lf+%lfi\n", l->depth, iter, j+1, j, CSPLIT( p->H[j][j+1] ) );
        finish = 1;
        break;
      }
    } // end of a single restart

    cuda_compute_solution_PRECISION( x, V, p->y, p->gamma, p->H, j, (res==_NO_RES)?ol:1, p, l, threading );
  } // end of fgmres

  return iter;
}
