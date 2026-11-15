#include "main.h"
#include "proxies/interpolation_proxy_PRECISION.h"

#ifdef CUDA_OPT
#include "gpu/cuda_interpolation_PRECISION.h"
#endif

// the GPU versions use l->is_PRECISION.op, which is not available with HALF_PREC_STORAGE
#if defined( CUDA_OPT ) && !defined( HALF_PREC_STORAGE )
#define INTERPOLATION_ON_GPU_PRECISION
#endif

#if defined( INTERPOLATION_ON_GPU_PRECISION ) && defined( CUDA_INTERP_BENCHMARK )

// Benchmark of restriction and interpolation (build with -DCUDA_INTERP_BENCHMARK): at the first
// call of each operation on each level, the CPU implementation (*_cpu, all threads) and the GPU one
// (*_vectorwrapper, with the host<->device copies and the MPI gather/distribute) are run
// INTERP_BENCH_REPS times on scratch output vectors, then the GPU parts alone are timed with
// CUDA events (cuda_interpolation_PRECISION_bench). Medians in ms, rank 0, one line per case:
//   interp bench: op=... depth=... prec=... layout=... threads=... cpu_ms=... gpu_ms=... h2d_ms=... kernel_ms=... d2h_ms=...
// The results of the run are not changed: the production call follows with the original vectors.

#define INTERP_BENCH_REPS 20

static int interp_bench_done_PRECISION[3][64];
static vector_PRECISION interp_bench_out_PRECISION = NULL;
static double interp_bench_t0_PRECISION, interp_bench_t_PRECISION[INTERP_BENCH_REPS];

static int interp_bench_cmp_PRECISION( const void *a, const void *b ) {
  double x = *(const double*)a, y = *(const double*)b;
  return (x > y) - (x < y);
}

// median time (ms, master thread) of INTERP_BENCH_REPS calls of operation op on the CPU (cpu = 1)
// or on the GPU (cpu = 0); called by all threads
static double interp_bench_time_PRECISION( int op, int cpu, vector_PRECISION in, level_struct *l,
                                           struct Thread *threading ) {
  vector_PRECISION out = interp_bench_out_PRECISION;
  for ( int r=0; r<INTERP_BENCH_REPS; r++ ) {
    SYNC_MASTER_TO_ALL(threading)
    START_MASTER(threading)
    interp_bench_t0_PRECISION = MPI_Wtime();
    END_MASTER(threading)
    SYNC_MASTER_TO_ALL(threading)
    if ( op == 0 ) {
      if ( cpu ) restrict_PRECISION_cpu( out, in, l, threading );
      else       cuda_restrict_PRECISION_vectorwrapper( out, in, l, threading );
    } else if ( op == 1 ) {
      if ( cpu ) interpolate_PRECISION_cpu( out, in, l, threading );
      else       cuda_interpolate_PRECISION_vectorwrapper( out, in, l, threading );
    } else {
      if ( cpu ) interpolate3_PRECISION_cpu( out, in, l, threading );
      else       cuda_interpolate3_PRECISION_vectorwrapper( out, in, l, threading );
    }
    SYNC_MASTER_TO_ALL(threading)
    START_MASTER(threading)
    interp_bench_t_PRECISION[r] = 1e3*( MPI_Wtime() - interp_bench_t0_PRECISION );
    END_MASTER(threading)
  }
  SYNC_MASTER_TO_ALL(threading)
  double tmed = 0;
  START_MASTER(threading)
  qsort( interp_bench_t_PRECISION, INTERP_BENCH_REPS, sizeof(double), interp_bench_cmp_PRECISION );
  tmed = interp_bench_t_PRECISION[INTERP_BENCH_REPS/2];
  END_MASTER(threading)
  return tmed;
}

// op: 0 restriction (phi_c = P^dagger phi), 1 interpolation (phi += P phi_c), 2 interpolation (phi = P phi_c);
// out: the output vector of the production call (only its contents for op = 1), in: its input
static void interp_bench_PRECISION( int op, vector_PRECISION out, vector_PRECISION in, level_struct *l,
                                    struct Thread *threading ) {
  if ( l->depth >= 64 || interp_bench_done_PRECISION[op][l->depth] ) return;

  SYNC_MASTER_TO_ALL(threading)
  START_MASTER(threading)
  interp_bench_done_PRECISION[op][l->depth] = 1;
  size_t n = ( op == 0 ) ? (size_t)l->next_level->vector_size : (size_t)l->vector_size;
  interp_bench_out_PRECISION = (vector_PRECISION) malloc( n*sizeof(complex_PRECISION) );
  if ( interp_bench_out_PRECISION == NULL ) {
    fprintf( stderr, "interpolation benchmark: host allocation failed\n" );
    MPI_Abort( MPI_COMM_WORLD, 1 );
  }
  // only the inner sites: out may be shorter than vector_size (e.g. the pinned buffer of the setup)
  if ( op == 1 ) memcpy( interp_bench_out_PRECISION, out, (size_t)l->inner_vector_size*sizeof(complex_PRECISION) );
  else           memset( interp_bench_out_PRECISION, 0, n*sizeof(complex_PRECISION) );
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)

  double t_cpu = interp_bench_time_PRECISION( op, 1, in, l, threading );
  double t_gpu = interp_bench_time_PRECISION( op, 0, in, l, threading );

  START_MASTER(threading)
  double t[4];
  cuda_interpolation_PRECISION_bench( l, op, INTERP_BENCH_REPS, t );
  const char *name[3] = { "restrict", "interpolate", "interpolate3" };
  printf0( "interp bench: op=%s depth=%d prec=PRECISION layout=%s threads=%d n_fine=%d n_coarse=%d reps=%d "
           "cpu_ms=%.4f gpu_ms=%.4f h2d_ms=%.4f kernel_ms=%.4f d2h_ms=%.4f\n",
           name[op], l->depth, t[3] > 0.5 ? "SSE" : "plain", threading->n_core*threading->n_thread,
           l->inner_vector_size, l->is_PRECISION.num_agg*2*l->num_eig_vect, INTERP_BENCH_REPS,
           t_cpu, t_gpu, t[0], t[1], t[2] );
  free( interp_bench_out_PRECISION );
  interp_bench_out_PRECISION = NULL;
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)
}

#endif

void interpolation_PRECISION_alloc( level_struct *l ) {
  cpu_interpolation_PRECISION_alloc( l );
#ifdef CUDA_OPT
  cuda_interpolation_PRECISION_alloc( l );
#endif
}

void interpolation_PRECISION_free( level_struct *l ) {
#ifdef CUDA_OPT
  cuda_interpolation_PRECISION_free( l );
#endif
  cpu_interpolation_PRECISION_free( l );
}

void interpolation_PRECISION_op_to_device( level_struct *l, struct Thread *threading ) {
#ifdef INTERPOLATION_ON_GPU_PRECISION
  START_LOCKED_MASTER(threading)
  cuda_interpolation_PRECISION_op_to_gpu( l );
  END_LOCKED_MASTER(threading)
#endif
}

void interpolate_PRECISION( vector_PRECISION phi, vector_PRECISION phi_c, level_struct *l, struct Thread *threading ) {
#ifdef INTERPOLATION_ON_GPU_PRECISION
#ifdef CUDA_INTERP_BENCHMARK
  interp_bench_PRECISION( 1, phi, phi_c, l, threading );
#endif
  cuda_interpolate_PRECISION_vectorwrapper( phi, phi_c, l, threading );
#else
  interpolate_PRECISION_cpu( phi, phi_c, l, threading );
#endif
}

void interpolate3_PRECISION( vector_PRECISION phi, vector_PRECISION phi_c, level_struct *l, struct Thread *threading ) {
#ifdef INTERPOLATION_ON_GPU_PRECISION
#ifdef CUDA_INTERP_BENCHMARK
  interp_bench_PRECISION( 2, phi, phi_c, l, threading );
#endif
  cuda_interpolate3_PRECISION_vectorwrapper( phi, phi_c, l, threading );
#else
  interpolate3_PRECISION_cpu( phi, phi_c, l, threading );
#endif
}

void restrict_PRECISION( vector_PRECISION phi_c, vector_PRECISION phi, level_struct *l, struct Thread *threading ) {
#ifdef INTERPOLATION_ON_GPU_PRECISION
#ifdef CUDA_INTERP_BENCHMARK
  interp_bench_PRECISION( 0, phi_c, phi, l, threading );
#endif
  cuda_restrict_PRECISION_vectorwrapper( phi_c, phi, l, threading );
#else
  restrict_PRECISION_cpu( phi_c, phi, l, threading );
#endif
}
