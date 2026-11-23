// Outer FGMRES of the solve on the GPU.
//
// Same algorithm as fgmres_double for the outer solver (right preconditioning with the float
// V-cycle, classical Gram-Schmidt, restarts, stop on the estimated relative residual), with the
// fine vectors kept on the GPU: V, Z, x, b, w in double, lexicographic ordering of g.op_double.
// The V-cycle takes its right-hand side from the GPU (eta_gpu of the depth-0 interpolation struct,
// written here, see eta_preloaded_host in cuda_restrict_PRECISION_vectorwrapper) and the depth-0
// smoother leaves its result there (phi_gpu, see keep_phi_on_gpu in
// cuda_solve_oddeven_PRECISION_resident). Only scalars go through the host: the MPI reductions of
// the dot products, the Hessenberg matrix and the Givens rotations.
// Vector operations: the GPU linear algebra of the depth-0 smoother (cuda_linalg_double).
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex>
#include <vector>

#include "gpu/cuda_componentwise.h"
#include "gpu/cuda_linalg_double.h"

extern "C"{

  #define IMPORT_FROM_EXTERN_C
  #include "main.h"
  #undef IMPORT_FROM_EXTERN_C

  #include "gpu/cuda_oddeven_linalg_double.h"

  // vcycle_float.h is not included by main.h in CUDA files (IMPORT_FROM_EXTERN_C)
  void vcycle_float( vector_float phi, vector_float Dphi, vector_float eta,
                     int res, level_struct *l, struct Thread *threading );
}

#include "gpu/cuda_miscellaneous.h"
#include "gpu/cuda_dirac_double.h"
#include "gpu/cuda_outer_fgmres.h"

constexpr uint diracDefaultBlockSize = 128;

#ifdef CUDA_OPT

typedef std::complex<double> cd_outer;

// GPU vectors of the outer solver, allocated at the first solve (and again if the sizes change)
static struct {
  cuda_vector_double *V, *Z;   // m+1 and m columns
  cuda_vector_double x, b, w;
  int *tt;                     // lexicographic site -> Schwarz site (l->s_float.op.translation_table)
  int vl, m;
  int finish, stop_restart;    // written by the master, read by all threads after a barrier
} og = { NULL, NULL, NULL, NULL, NULL, NULL, 0, 0, 0, 0 };


// double, lexicographic ordering -> float, Schwarz ordering (trans_float on the GPU)
__global__ void outer_trans_float_kernel( cu_cmplx_float *out, const cu_cmplx_double *in, const int *tt,
                                          size_t n_sites, int nsv ) {
  size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
  if ( i >= n_sites*nsv ) return;
  size_t site = i / nsv, c = i % nsv;
  cu_cmplx_double v = in[i];
  out[(size_t)tt[site]*nsv + c] = make_cuFloatComplex( (float)v.x, (float)v.y );
}

// float, Schwarz ordering -> double, lexicographic ordering (trans_back_float on the GPU)
__global__ void outer_trans_back_float_kernel( cu_cmplx_double *out, const cu_cmplx_float *in, const int *tt,
                                               size_t n_sites, int nsv ) {
  size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
  if ( i >= n_sites*nsv ) return;
  size_t site = i / nsv, c = i % nsv;
  cu_cmplx_float v = in[(size_t)tt[site]*nsv + c];
  out[i] = make_cuDoubleComplex( (double)v.x, (double)v.y );
}


// 1 if the outer FGMRES of this solve can run on the GPU: depth-0 FGMRES in double precision with
// the mixed-precision V-cycle (method 4, odd-even, one cycle) as right preconditioner, and the
// GPU buffers of the depth-0 operator, smoother and interpolation in place
extern "C" int cuda_outer_fgmres_applicable( gmres_double_struct *p, level_struct *l ) {
#if defined( OUTER_FGMRES_ON_CPU ) || defined( GCR_SMOOTHER ) || defined( HALF_PREC_STORAGE )
  return 0;
#else
  return ( l->depth == 0 && g.mixed_precision == 1 && g.method == 4 && g.odd_even && g.interpolation &&
           l->level > 0 && l->n_cy == 1 && p->preconditioner != NULL && p->kind == _RIGHT &&
           creal_double(p->shift) == 0 && cimag_double(p->shift) == 0 && p->op == &(g.op_double) &&
           g.vt.p_end == NULL && !g.vt.evaluation &&
           p->op->x_componentwise_gpu != NULL && p->op->w_componentwise_gpu != NULL &&
           l->is_float.op_gpu != NULL && l->is_float.eta_gpu != NULL && l->is_float.phi_gpu != NULL &&
           l->num_lattice_site_var == 12 );
#endif
}


static void outer_buffers( gmres_double_struct *p, level_struct *l ) {
  int vl = l->inner_vector_size, m = p->restart_length;
  if ( og.vl != vl || og.m != m ) {
    if ( og.V != NULL ) {
      for ( int i=0; i<=og.m; i++ ) cuda_safe_call( cudaFree( og.V[i] ) );
      for ( int i=0; i<og.m; i++ )  cuda_safe_call( cudaFree( og.Z[i] ) );
      free( og.V ); free( og.Z );
      cuda_safe_call( cudaFree( og.x ) ); cuda_safe_call( cudaFree( og.b ) ); cuda_safe_call( cudaFree( og.w ) );
    }
    og.V = (cuda_vector_double*) malloc( (m+1)*sizeof(cuda_vector_double) );
    og.Z = (cuda_vector_double*) malloc( m*sizeof(cuda_vector_double) );
    if ( og.V == NULL || og.Z == NULL ) {
      fprintf( stderr, "cuda_outer_fgmres_double: host allocation failed\n" );
      MPI_Abort( MPI_COMM_WORLD, 1 );
    }
    for ( int i=0; i<=m; i++ ) cuda_safe_call( cudaMalloc( (void**)&(og.V[i]), (size_t)vl*sizeof(cu_cmplx_double) ) );
    for ( int i=0; i<m; i++ )  cuda_safe_call( cudaMalloc( (void**)&(og.Z[i]), (size_t)vl*sizeof(cu_cmplx_double) ) );
    cuda_safe_call( cudaMalloc( (void**)&og.x, (size_t)vl*sizeof(cu_cmplx_double) ) );
    cuda_safe_call( cudaMalloc( (void**)&og.b, (size_t)vl*sizeof(cu_cmplx_double) ) );
    cuda_safe_call( cudaMalloc( (void**)&og.w, (size_t)vl*sizeof(cu_cmplx_double) ) );
    og.vl = vl; og.m = m;
  }
  if ( og.tt == NULL ) {
    // the translation table is fixed after the setup of the Schwarz layout
    size_t n = (size_t)l->num_inner_lattice_sites;
    cuda_safe_call( cudaMalloc( (void**)&og.tt, n*sizeof(int) ) );
    cuda_safe_call( cudaMemcpy( og.tt, l->s_float.op.translation_table, n*sizeof(int), cudaMemcpyHostToDevice ) );
  }
}

// out = D in, D = g.op_double on the GPU (as cuda_d_plus_clover_double_vectorwrapper, without the host copies)
static void outer_apply_D( cuda_vector_double out, cuda_vector_double in, operator_double_struct *op,
                           level_struct *l, struct Thread *threading ) {
  const size_t n = (size_t)l->num_inner_lattice_sites, nsv = (size_t)l->num_lattice_site_var;
  const uint grid = minGridSizeForN( n, diracDefaultBlockSize );
  reorderArrayByComponent<<< grid, diracDefaultBlockSize >>>( op->x_componentwise_gpu, in, nsv, n );
  cuda_safe_call( cudaDeviceSynchronize() );
  cuda_d_plus_clover_double( op->w_componentwise_gpu, op->x_componentwise_gpu, op, l, threading );
  reorderArrayByChunks<<< grid, diracDefaultBlockSize >>>( out, op->w_componentwise_gpu, nsv, n );
  cuda_safe_call( cudaDeviceSynchronize() );
}

// global 2-norm (MPI reduction inside cuda_global_inner_product_double)
static double outer_norm( cuda_vector_double v, gmres_double_struct *p, level_struct *l, struct Thread *threading ) {
  complex_double d;
  cuda_global_inner_product_double( &v, v, &d, 1, 0, og.vl, p, l, threading );
  return sqrt( creal_double(d) );
}

// z += sum_i alpha[i] V[i], i < n, in groups of CUDA_MULTI_SAXPY_MAX vectors
static void outer_multi_saxpy( cuda_vector_double z, cuda_vector_double *V, const cd_outer *alpha, int n,
                               level_struct *l, cudaStream_t *streams ) {
  cu_cmplx_double a[CUDA_MULTI_SAXPY_MAX];
  for ( int i0=0; i0<n; i0+=CUDA_MULTI_SAXPY_MAX ) {
    int k = ( n-i0 < CUDA_MULTI_SAXPY_MAX ) ? n-i0 : CUDA_MULTI_SAXPY_MAX;
    for ( int i=0; i<k; i++ ) a[i] = make_cuDoubleComplex( alpha[i0+i].real(), alpha[i0+i].imag() );
    cuda_vector_double_multi_saxpy( z, V+i0, a, k, 0, og.vl, l, _CUDA_SYNC, 0, streams );
  }
}


extern "C" int cuda_outer_fgmres_double( gmres_double_struct *p, level_struct *l, struct Thread *threading ) {

  // master-only state (the other threads only take part in the V-cycle)
  static std::vector<cd_outer> H, gam, sn, cs, y, h;
  static std::vector<complex_double> hd;
  static double norm_r0, gamma_jp1, t0, t1;
  static int iter, j, res;

  const int m = p->restart_length;
  const size_t n_sites = (size_t)l->num_inner_lattice_sites;
  const int nsv = l->num_lattice_site_var;
  auto Hc = [&]( int col, int row ) -> cd_outer& { return H[ (size_t)col*(m+1) + row ]; };   // p->H[col][row]
  cudaStream_t stream = CU_STREAM_PER_THREAD;
  cudaStream_t* const streams = &stream;

  START_LOCKED_MASTER(threading)
  if ( p->timing || p->print ) prof_init( l );
  t0 = MPI_Wtime();
#if defined(TRACK_RES) && !defined(WILSON_BENCHMARK)
  if ( p->print && g.print > 0 ) printf0("+----------------------------------------------------------+\n");
#endif
  outer_buffers( p, l );
  H.assign( (size_t)m*(m+1), 0 ); gam.assign( m+1, 0 ); sn.assign( m, 0 ); cs.assign( m, 0 );
  y.assign( m, 0 ); h.assign( m+1, 0 ); hd.resize( m+1 );
  norm_r0 = 1; gamma_jp1 = 1; iter = 0; j = -1;
  og.finish = 0;
  // the only host-to-device copies of fine vectors in the solve
  cuda_vector_double_copy( og.b, p->b, 0, og.vl, l, _H2D, _CUDA_SYNC, 0, streams );
  if ( !p->initial_guess_zero )
    cuda_vector_double_copy( og.x, p->x, 0, og.vl, l, _H2D, _CUDA_SYNC, 0, streams );
  END_LOCKED_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)

  for ( int ol=0; ol<p->num_restart && og.finish==0; ol++ ) {

    START_LOCKED_MASTER(threading)
    if ( ol == 0 && p->initial_guess_zero ) {
      res = _NO_RES;
      cuda_vector_double_copy( og.w, og.b, 0, og.vl, l, _D2D, _CUDA_SYNC, 0, streams );        // r = b
    } else {
      res = _RES;
      outer_apply_D( og.w, og.x, p->op, l, threading );                                       // w = D x
      cuda_vector_double_minus( og.w, og.b, og.w, 0, og.vl, l, _CUDA_SYNC, 0, streams );      // r = b - D x
    }
    double gamma0 = outer_norm( og.w, p, l, threading );
    gam[0] = gamma0;
    if ( ol == 0 ) norm_r0 = gamma0;
    cuda_vector_double_scale( og.V[0], og.w, make_cuDoubleComplex( 1.0/gamma0, 0 ), 0, og.vl, l,
                              _CUDA_SYNC, 0, streams );                                       // v_0 = r / gamma_0
    og.stop_restart = 0;
    END_LOCKED_MASTER(threading)
    SYNC_MASTER_TO_ALL(threading)

    for ( int il=0; il<m && og.finish==0 && og.stop_restart==0; il++ ) {

      // Z_j = M V_j: V_j into the depth-0 restriction buffer eta_gpu (float, Schwarz ordering)
      START_LOCKED_MASTER(threading)
      j = il; iter++;
      interpolation_float_struct *is = &(l->is_float);
      outer_trans_float_kernel<<< minGridSizeForN( og.vl, 256 ), 256 >>>( is->eta_gpu, og.V[j], og.tt, n_sites, nsv );
      cuda_safe_call( cudaDeviceSynchronize() );
      is->eta_preloaded_host = l->sbuf_float[0];
      is->keep_phi_on_gpu = 1;
      is->phi_result_host = NULL;
      is->defer_phi_copy = 1;
      END_LOCKED_MASTER(threading)
      SYNC_MASTER_TO_ALL(threading)

      vcycle_float( l->sbuf_float[1], NULL, l->sbuf_float[0], _NO_RES, l, threading );

      START_LOCKED_MASTER(threading)
      interpolation_float_struct *is = &(l->is_float);
      if ( is->phi_result_host != l->sbuf_float[1] ) {
        // the smoother did not leave its result on the GPU (it ran on the host vectors)
        cuda_safe_call( cudaMemcpy( is->phi_gpu, l->sbuf_float[1], (size_t)og.vl*sizeof(cu_cmplx_float),
                                    cudaMemcpyHostToDevice ) );
      }
      is->keep_phi_on_gpu = 0; is->phi_result_host = NULL; is->defer_phi_copy = 0;
      is->eta_preloaded_host = NULL; is->eta_host_stale = NULL;
      outer_trans_back_float_kernel<<< minGridSizeForN( og.vl, 256 ), 256 >>>( og.Z[j], is->phi_gpu, og.tt, n_sites, nsv );
      cuda_safe_call( cudaDeviceSynchronize() );

      cuda_vector_double w = og.V[j+1];
      outer_apply_D( w, og.Z[j], p->op, l, threading );                                       // w = D Z_j

      // h_i = <v_i, w>, i = 0..j (reduced over the processes); w = w - sum_i h_i v_i
      cuda_global_inner_product_double( og.V, w, hd.data(), j+1, 0, og.vl, p, l, threading );
      for ( int i=0; i<=j; i++ ) {
        Hc(j,i) = cd_outer( creal_double(hd[i]), cimag_double(hd[i]) );
        h[i] = -Hc(j,i);
      }
      outer_multi_saxpy( w, og.V, h.data(), j+1, l, streams );

      // h_{j+1} = ||w||, v_{j+1} = w / h_{j+1}
      double hn = outer_norm( w, p, l, threading );
      Hc(j,j+1) = hn;
      if ( hn > 1e-15 )
        cuda_vector_double_scale( w, w, make_cuDoubleComplex( 1.0/hn, 0 ), 0, og.vl, l, _CUDA_SYNC, 0, streams );

      if ( std::abs( Hc(j,j+1) ) > p->tol/10 ) {
        // Givens rotations (qr_update_double)
        for ( int i=0; i<j; i++ ) {
          cd_outer beta = (-sn[i])*Hc(j,i) + cs[i]*Hc(j,i+1);
          Hc(j,i) = std::conj(cs[i])*Hc(j,i) + std::conj(sn[i])*Hc(j,i+1);
          Hc(j,i+1) = beta;
        }
        double beta = sqrt( std::norm( Hc(j,j) ) + std::norm( Hc(j,j+1) ) );
        sn[j] = Hc(j,j+1)/beta; cs[j] = Hc(j,j)/beta;
        gam[j+1] = (-sn[j])*gam[j]; gam[j] = std::conj(cs[j])*gam[j];
        Hc(j,j) = beta; Hc(j,j+1) = 0;
        gamma_jp1 = std::abs( gam[j+1] );
#if defined(TRACK_RES) && !defined(WILSON_BENCHMARK)
        if ( p->print && g.print > 0 )
          printf0("| approx. rel. res. after  %-6d iterations: %e |\n", iter, gamma_jp1/norm_r0 );
#endif
        if ( gamma_jp1/norm_r0 < p->tol || gamma_jp1/norm_r0 > 1E+5 ) {
          og.finish = 1;
          if ( gamma_jp1/norm_r0 > 1E+5 ) printf0("Divergence of fgmres_double, iter = %d, level=%d\n", iter, l->level );
        }
      } else {
        printf0("depth: %d, iter: %d, p->H(%d,%d) = %+lf+%lfi\n", l->depth, iter, j+1, j,
                Hc(j,j+1).real(), Hc(j,j+1).imag() );
        og.finish = 1;
        og.stop_restart = 1;
      }
      END_LOCKED_MASTER(threading)
      SYNC_MASTER_TO_ALL(threading)
    }

    // x = x + Z y (x = Z y after a zero initial guess), as compute_solution_double
    START_LOCKED_MASTER(threading)
    for ( int i=j; i>=0; i-- ) {
      y[i] = gam[i];
      for ( int k=i+1; k<=j; k++ ) y[i] -= Hc(k,i)*y[k];
      y[i] /= Hc(i,i);
    }
    if ( res == _NO_RES && ol == 0 )
      cuda_vector_double_define( og.x, make_cuDoubleComplex( 0, 0 ), 0, og.vl, l, _CUDA_SYNC, 0, streams );
    outer_multi_saxpy( og.x, og.Z, y.data(), j+1, l, streams );
    END_LOCKED_MASTER(threading)
    SYNC_MASTER_TO_ALL(threading)
  }

  START_LOCKED_MASTER(threading)
  t1 = MPI_Wtime(); g.total_time = t1-t0; g.iter_count = iter; g.norm_res = gamma_jp1/norm_r0;
  double beta = gamma_jp1;
#ifdef FGMRES_RESTEST
  if ( p->print ) {
    // exact residual, on the GPU
    outer_apply_D( og.w, og.x, p->op, l, threading );
    cuda_vector_double_minus( og.w, og.b, og.w, 0, og.vl, l, _CUDA_SYNC, 0, streams );
    beta = outer_norm( og.w, p, l, threading );
  }
#endif
  // the only device-to-host copy of a fine vector in the solve
  cuda_vector_double_copy( p->x, og.x, 0, og.vl, l, _D2H, _CUDA_SYNC, 0, streams );

  if ( p->print ) {
    g.norm_res = beta/norm_r0;
#if defined(TRACK_RES) && !defined(WILSON_BENCHMARK)
    if ( g.print > 0 ) printf0("+----------------------------------------------------------+\n\n");
#endif
    printf0("+----------------------------------------------------------+\n");
    printf0("|       FGMRES iterations: %-6d coarse average: %-6.2lf   |\n", iter,
            ((double)g.coarse_iter_count)/((double)iter) );
    printf0("| exact relative residual: ||r||/||b|| = %e      |\n", beta/norm_r0 );
    printf0("| elapsed wall clock time: %-8.4lf seconds                |\n", t1-t0 );
    if ( g.coarse_time > 0 )
      printf0("|        coarse grid time: %-8.4lf seconds (%04.1lf%%)        |\n",
              g.coarse_time, 100*(g.coarse_time/(t1-t0)) );
    printf0("|        coarsest grid time: %-8.4lf seconds (%04.1lf%%)        |\n",
              g.coarsest_time, 100*(g.coarsest_time/(t1-t0)) );
    printf0("| coarsest grid matmul time: %-8.4lf seconds (%04.1lf%%)        |\n",
              g.matmul_time, 100*(g.matmul_time/(t1-t0)) );
    printf0("|  consumed core minutes*: %-8.2le (solve only)           |\n", ((t1-t0)*g.num_processes*MAX(1,threading->n_core))/60.0 );
    printf0("|    max used mem/MPIproc: %-8.2le GB                     |\n", g.max_storage/1024.0 );
    printf0("|        max used mem/GPU: %-8.2le GB                     |\n", g.max_gpu_storage/1024.0 );
    printf0("|           outer FGMRES on the GPU                        |\n" );
    printf0("+----------------------------------------------------------+\n");
    printf0("*: only correct if #MPIprocs*#threads == #CPUs\n\n");
  }
  if ( p->timing || p->print ) prof_print( l );
  END_LOCKED_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)

  return iter;
}

#endif
