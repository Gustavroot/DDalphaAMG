#include <mpi.h>

extern "C"{

  #define IMPORT_FROM_EXTERN_C
  #include "main.h"
  #undef IMPORT_FROM_EXTERN_C

  #include "profiling.h"
  #include "operator.h"
}

#include "cuda_miscellaneous.h"
#include "cuda_dirac_kernels_PRECISION.h"
#include "cuda_dirac_kernels_componentwise_PRECISION.h"
#include "cuda_complex_cxx.h"
#include "cuda_ghost_PRECISION.h"
#include "cuda_componentwise.h"
#ifdef CUDA_OPT

constexpr uint diracDefaultBlockSize = 128;

__constant__ cu_cmplx_PRECISION gamma_info_vals_PRECISION[16];
__constant__ int gamma_info_coo_PRECISION[16];


extern "C" void copy_2_cpu_PRECISION_v3( vector_PRECISION out, cuda_vector_PRECISION out_gpu, vector_PRECISION in, cuda_vector_PRECISION in_gpu, level_struct *l){
  cudaStream_t *streams_gmres = (l->p_PRECISION).streams;

  cuda_vector_PRECISION_copy( (void*)out, (void*)out_gpu, 0, l->inner_vector_size, l, _D2H, _CUDA_SYNC,
                              0, streams_gmres );
  cuda_vector_PRECISION_copy( (void*)in, (void*)in_gpu, 0, l->inner_vector_size, l, _D2H, _CUDA_SYNC,
                              0, streams_gmres );
}


__forceinline__ __device__ void
_cuda_block_d_plus_clover_PRECISION_6threads_naive(		cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
                                                                schwarz_PRECISION_struct_on_gpu *s, int idx,
                                                                cu_cmplx_PRECISION *buf, int ext_dir ){

  int dir, k=0, j=0, i=0, **index = s->index, *ind, *neighbor = s->op.neighbor_table, loc_ind=idx%6,
      spin, w, *gamma_coo, idx_in_cublock = idx%blockDim.x;
  cu_cmplx_PRECISION *gamma_val, *buf1, *buf2, *buf3, *buf4;

  buf += (idx_in_cublock/6)*24;
  buf1 = buf + 0*6;
  buf2 = buf + 1*6;
  buf3 = buf + 2*6;
  buf4 = buf + 3*6;

  //TODO: implement a dynamic way of loading s->op.D into __shared__

  cu_config_PRECISION* D = s->op.D + (start/12)*36;
  cu_config_PRECISION* D_pt;

  cu_cmplx_PRECISION *leta, *lphi; //eta and phi are already shifted by 'start'

  // TODO: is there a reason for this not to be integrated with extra_dir ?
  dir = ext_dir;

  ind = index[dir];

  i = idx/6;
  k = ind[i];
  j = neighbor[4*k+dir];
  D_pt = D + 36*k + 9*dir;

  // already added <start> to the original input spinors
  lphi = phi;
  leta = eta;

  spin = (loc_ind/3)*2;
  //with this setup, gamma_val[0] gives spins 0 and 1, and gamma_val[1] spins 2 and 3
  //gamma_val = s->gamma_info_vals + dir*4 + spin;
  //gamma_coo = s->gamma_info_coo  + dir*4 + spin;
  gamma_val = gamma_info_vals_PRECISION + dir*4 + spin;
  gamma_coo = gamma_info_coo_PRECISION  + dir*4 + spin;

  // prn_T_PRECISION(...)
  buf1[ loc_ind ] = cu_cadd_PRECISION( (lphi + 12*k)[ loc_ind ],
                                       cu_cmul_PRECISION(
                                       gamma_val[0], (lphi + 12*k)[ 3*gamma_coo[0] + loc_ind%3 ] )
                                       );
  // prp_T_PRECISION(...)
  buf2[ loc_ind ] = cu_csub_PRECISION( (lphi + 12*j)[ loc_ind ],
                                       cu_cmul_PRECISION(
                                       gamma_val[0], (lphi + 12*j)[ 3*gamma_coo[0] + loc_ind%3 ] )
                                       );
  __syncthreads();
  // mvmh_PRECISION(...), twice
  buf3[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
  for( w=0; w<3; w++ ){
    buf3[ loc_ind ] = cu_cadd_PRECISION( buf3[ loc_ind ],
                                         cu_cmul_PRECISION(
                                         cu_conj_PRECISION(D_pt[ loc_ind%3 + w*3 ]), buf1[ (loc_ind/3)*3 + w ] )
                                         );
  }
  // mvm_PRECISION(...), twice
  buf4[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
  for( w=0; w<3; w++ ){
    buf4[ loc_ind ] = cu_cadd_PRECISION( buf4[ loc_ind ],
                                         cu_cmul_PRECISION(
                                         D_pt[ (loc_ind*3)%9 + w ], buf2[ (loc_ind/3)*3 + w ] )
                                         );
  }
  __syncthreads();
  // pbn_su3_T_PRECISION(...)
  (leta + 12*j)[ loc_ind ] = cu_csub_PRECISION( (leta + 12*j)[ loc_ind ], buf3[ loc_ind ] );
  (leta + 12*j)[ 6 + loc_ind ] = cu_csub_PRECISION( (leta + 12*j)[ 6 + loc_ind ],
                                                    cu_cmul_PRECISION(
                                                    gamma_val[1], buf3[ 3*gamma_coo[1] + loc_ind%3 ] )
                                                    );
  // pbp_su3_T_PRECISION(...);
  (leta + 12*k)[ loc_ind ] = cu_csub_PRECISION( (leta + 12*k)[ loc_ind ], buf4[ loc_ind ] );
  (leta + 12*k)[ 6 + loc_ind ] = cu_cadd_PRECISION( (leta + 12*k)[ 6 + loc_ind ],
                                                    cu_cmul_PRECISION(
                                                    gamma_val[1], buf4[ 3*gamma_coo[1] + loc_ind%3 ] )
                                                    );
}


// gamma_val and gamma_coo are both loaded into shared memory for this kernel
__global__ void
cuda_block_d_plus_clover_PRECISION_6threads_naive(		cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block, int ext_dir ){

  int idx, DD_block_id, block_id, start;
  idx = threadIdx.x + blockDim.x * blockIdx.x;

  cu_cmplx_PRECISION *shared_data_loc, *tmp_loc;

  // not really a DD block id, but rather a linear counting of a grouping (per DD block) of CUDA threads
  DD_block_id = idx/nr_threads_per_DD_block;

  // offsetting idx to make it zero at the beginning of the threads living within a DD block
  idx = idx%nr_threads_per_DD_block;

  // this int will be the ACTUAL DD block ID, in the sense of accessing data from e.g. block_struct* block
  block_id = DD_blocks_to_compute[DD_block_id];

  // this is the DD-block start of the spinors (phi, r, latest_iter and temporary ones)
  start = block[block_id].start * num_latt_site_var;

  out += start;
  in += start;

  extern __shared__ cu_cmplx_PRECISION shared_data[];

  // a part of shared_memory is dedicated to even sites, the rest to odd sites
  shared_data_loc = shared_data;

  //the following are 'bare' values, i.e. with respect to the 0th element within a CUDA block
  // EVEN
  tmp_loc = shared_data_loc;

  //if(idx < 6*nr_block_even_sites){
    tmp_loc[threadIdx.x + 0*blockDim.x] = make_cu_cmplx_PRECISION(0.0,0.0);
    tmp_loc[threadIdx.x + 1*blockDim.x] = make_cu_cmplx_PRECISION(0.0,0.0);
    tmp_loc[threadIdx.x + 2*blockDim.x] = make_cu_cmplx_PRECISION(0.0,0.0);
    tmp_loc[threadIdx.x + 3*blockDim.x] = make_cu_cmplx_PRECISION(0.0,0.0);
  //} //even

  _cuda_block_d_plus_clover_PRECISION_6threads_naive(out, in, start, s, idx, tmp_loc, ext_dir);
}


extern "C" void
cuda_block_d_plus_clover_PRECISION(				cuda_vector_PRECISION eta, cuda_vector_PRECISION phi,
	                                                        int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s,
	                                                        level_struct *l, struct Thread *threading, int stream_id,
	                                                        cudaStream_t *streams, int color, int* DD_blocks_to_compute_gpu,
	                                                        int* DD_blocks_to_compute_cpu ){

  if( nr_DD_blocks_to_compute==0 ){ return; }

  int threads_per_cublock, nr_threads, nr_threads_per_DD_block, dir, n = s->num_block_sites;
  size_t tot_shared_mem;

  // clover term
  if ( g.csw == 0.0 ) {
    threads_per_cublock = 96;

    nr_threads = n;
    nr_threads *= l->num_lattice_site_var;
    nr_threads *= nr_DD_blocks_to_compute;

    nr_threads_per_DD_block = nr_threads / nr_DD_blocks_to_compute;

    tot_shared_mem = 0;

    cuda_clover_diag_PRECISION<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                  tot_shared_mem, streams[stream_id]
                              >>>
                              ( eta, phi, s->s_on_gpu, g.my_rank, nr_threads_per_DD_block,
                                DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block );

  } else {
    threads_per_cublock = 96;

    // diag_oo inv
    nr_threads = n; // nr sites per DD block
    nr_threads = nr_threads*(12/2); // threads per site
    nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
    tot_shared_mem = 2*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) +
                     1*42*(threads_per_cublock/6)*sizeof(cu_config_PRECISION);

    cuda_block_site_clover_PRECISION<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                  tot_shared_mem, streams[stream_id]
                              >>>
                              ( eta, phi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block );
  }

  threads_per_cublock = 96;
  tot_shared_mem = 1*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 16*sizeof(cu_cmplx_PRECISION) + 16*sizeof(int);
  for( dir=0; dir<4; dir++ ){
    nr_threads = s->dir_length[dir];
    nr_threads = nr_threads*(12/2);
    nr_threads = nr_threads*nr_DD_blocks_to_compute;
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
    cuda_block_hopping_term_PRECISION_plus_6threads_naive<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                             tot_shared_mem, streams[stream_id]
                                                         >>>
                                                         ( eta, phi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                           DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block,
                                                           dir, _FULL_SYSTEM );
    cuda_block_hopping_term_PRECISION_minus_6threads_naive<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                              tot_shared_mem, streams[stream_id]
                                                          >>>
                                                          ( eta, phi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                            DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block,
                                                            dir, _FULL_SYSTEM );
  }
}

/** \brief Calculates the self-coupling term eta = D_sc phi.
 * 
 * 
 *  \param[out]   eta           eta in eta = D_sc phi.
 *  \param[in]    phi           psi in eta = D_sc phi.
 *  \param[in]    clover        D_sc in eta = D_sc phi.
 *  \param[in]    num_sites     Number of lattice sites.
 *  \param[in]    l             The level_struct being passed around everywhere.
 */
extern "C" void _cuda_clover_componentwise_PRECISION(cuda_vector_PRECISION eta,
                                                     cuda_vector_PRECISION phi,
                                                     cuda_config_PRECISION clover, int num_sites,
                                                     level_struct *l) {
  constexpr size_t blockSize = 128;

  PROF_PRECISION_START_UNTHREADED( _SC );
  const size_t gridSize = minGridSizeForN(num_sites, blockSize);
  cuda_site_clover_componentwise_PRECISION<<< gridSize, blockSize>>>(eta, phi, clover, num_sites);
  cuda_safe_call(cudaDeviceSynchronize());
  PROF_PRECISION_STOP_UNTHREADED( _SC, 1);
}

extern "C" void cuda_diag_ee_componentwise_PRECISION(cuda_vector_PRECISION eta,
                                                     cuda_vector_PRECISION phi,
                                                     cuda_config_PRECISION clover,
                                                     int num_sites, level_struct *l) {

  constexpr size_t blockSize = 128;

  const size_t gridSize = minGridSizeForN(num_sites, blockSize);
  cuda_site_diag_ee_componentwise_PRECISION<<<gridSize, blockSize>>>(eta, phi, clover, num_sites);
  cuda_safe_call(cudaDeviceSynchronize());
}

extern "C" void cuda_diag_oo_inv_componentwise_PRECISION(cuda_vector_PRECISION eta,
                                                         cuda_vector_PRECISION phi,
                                                         cuda_config_PRECISION clover,
                                                         int num_sites, level_struct *l) {

  constexpr size_t blockSize = 128;

  const size_t gridSize = minGridSizeForN(num_sites, blockSize);
  cuda_site_diag_oo_inv_componentwise_PRECISION<<<gridSize, blockSize>>>(eta, phi, clover, num_sites);
  cuda_safe_call(cudaDeviceSynchronize());
}

/** \brief Calculates the self-coupling term eta = D_sc phi.
 * 
 *  Vectors must be in componentwise ordering.
 * 
 *  \param[out]   eta           eta in eta = D_sc phi.
 *  \param[in]    phi           psi in eta = D_sc phi.
 *  \param[in]    clover        D_sc in eta = D_sc phi.
 *  \param[in]    num_sites     Number of lattice sites.
 *  \param[in]    l             The level_struct being passed around everywhere.
 */
extern "C" void _cuda_clover_PRECISION(cuda_vector_PRECISION eta, cuda_vector_PRECISION phi,
                                       cuda_config_PRECISION clover, int num_sites,
                                       level_struct *l) {
  constexpr size_t blockSize = 128;

  PROF_PRECISION_START_UNTHREADED( _SC );
  const size_t gridSize = minGridSizeForN(num_sites, blockSize);
  cuda_site_clover_PRECISION<<< gridSize, blockSize>>>(eta, phi, clover, num_sites);
  cuda_safe_call(cudaDeviceSynchronize());
  PROF_PRECISION_STOP_UNTHREADED( _SC, 1);
}


void cuda_d_plus_clover_PRECISION(
  cuda_vector_PRECISION eta, cuda_vector_PRECISION phi,
  operator_PRECISION_struct *op, level_struct *l, struct Thread *threading ) {
  RangeHandleType profilingRangeOperator = startProfilingRange("d_plus_clover_PRECISION (CUDA)");

  // There is no need to deal with multiple streams, so the default stream (per thread) is used.
  // It shall be handled as though being an array of streams of size 1.
  cudaStream_t stream = CU_STREAM_PER_THREAD;
  cudaStream_t* const streams = &stream;

  constexpr size_t blockSize = diracCommonBlockSize;
  const size_t gridSize = minGridSizeForN(l->num_inner_lattice_sites, blockSize);

  const auto pbs = projection_buffer_size(l->num_lattice_site_var, l->num_lattice_sites);
  auto shift = to_cuda_cmplx_PRECISION(op->shift);

  // Apply Clover term
  if ( g.csw == 0.0 ) {
    cuda_vector_PRECISION_scale(eta, phi, shift, 0, l->inner_vector_size, l, _CUDA_SYNC, 0, streams);
  } else {
    _cuda_clover_componentwise_PRECISION(eta, phi, op->clover_componentwise_gpu, l->num_inner_lattice_sites, l);
  }

  PROF_PRECISION_START_UNTHREADED( _NC );

  // Project in positive directions
  cuda_prp_T_componentwise_PRECISION<<<gridSize, blockSize>>>(op->prnT_gpu, phi, l->num_inner_lattice_sites);
  cuda_prp_Z_componentwise_PRECISION<<<gridSize, blockSize>>>(op->prnZ_gpu, phi, l->num_inner_lattice_sites);
  cuda_prp_Y_componentwise_PRECISION<<<gridSize, blockSize>>>(op->prnY_gpu, phi, l->num_inner_lattice_sites);
  cuda_prp_X_componentwise_PRECISION<<<gridSize, blockSize>>>(op->prnX_gpu, phi, l->num_inner_lattice_sites);
  cuda_safe_call(cudaDeviceSynchronize());

  cuda_ghost_sendrecv_PRECISION(op->prnT_gpu, T, -1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_sendrecv_PRECISION(op->prnZ_gpu, Z, -1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_sendrecv_PRECISION(op->prnY_gpu, Y, -1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_sendrecv_PRECISION(op->prnX_gpu, X, -1, &(op->cuda_c), _FULL_SYSTEM, l);

  // project plus dir and multiply with U dagger
  cuda_prn_T_componentwise_PRECISION<<<gridSize, blockSize>>>(op->pbuf_gpu, phi,
                                                              l->num_inner_lattice_sites);
  cuda_prn_mvmh_componentwise_PRECISION<<<2*gridSize, blockSize>>>(op->prpT_gpu, op->Ds_componentwise_gpu[T], op->pbuf_gpu,
                                                     op->neighbor_table_gpu, LatticeAxis::T,
                                                     l->num_inner_lattice_sites);
  cuda_prn_Z_componentwise_PRECISION<<<gridSize, blockSize>>>(op->pbuf_gpu, phi,
                                                              l->num_inner_lattice_sites);
  cuda_prn_mvmh_componentwise_PRECISION<<<2*gridSize, blockSize>>>(op->prpZ_gpu, op->Ds_componentwise_gpu[Z], op->pbuf_gpu,
                                                     op->neighbor_table_gpu, LatticeAxis::Z,
                                                     l->num_inner_lattice_sites);
  cuda_prn_Y_componentwise_PRECISION<<<gridSize, blockSize>>>(op->pbuf_gpu, phi,
                                                              l->num_inner_lattice_sites);
  cuda_prn_mvmh_componentwise_PRECISION<<<2*gridSize, blockSize>>>(op->prpY_gpu, op->Ds_componentwise_gpu[Y], op->pbuf_gpu,
                                                     op->neighbor_table_gpu, LatticeAxis::Y,
                                                     l->num_inner_lattice_sites);
  cuda_prn_X_componentwise_PRECISION<<<gridSize, blockSize>>>(op->pbuf_gpu, phi,
                                                              l->num_inner_lattice_sites);
  cuda_prn_mvmh_componentwise_PRECISION<<<2*gridSize, blockSize>>>(op->prpX_gpu, op->Ds_componentwise_gpu[X], op->pbuf_gpu,
                                                     op->neighbor_table_gpu, LatticeAxis::X,
                                                     l->num_inner_lattice_sites);
  cuda_safe_call(cudaDeviceSynchronize());

  cuda_ghost_sendrecv_PRECISION(op->prpT_gpu, T, +1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_sendrecv_PRECISION(op->prpZ_gpu, Z, +1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_sendrecv_PRECISION(op->prpY_gpu, Y, +1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_sendrecv_PRECISION(op->prpX_gpu, X, +1, &(op->cuda_c), _FULL_SYSTEM, l);

  cuda_ghost_wait_PRECISION(op->prnT_gpu, T, -1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_wait_PRECISION(op->prnZ_gpu, Z, -1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_wait_PRECISION(op->prnY_gpu, Y, -1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_wait_PRECISION(op->prnX_gpu, X, -1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_safe_call(cudaDeviceSynchronize());

  cuda_pbp_su3_mvm_componentwise_PRECISION<<<2*gridSize, blockSize>>>(op->pbuf_gpu, op->Ds_componentwise_gpu[T], op->prnT_gpu,
                                                        op->neighbor_table_gpu, LatticeAxis::T,
                                                        l->num_inner_lattice_sites);
  cuda_pbp_su3_T_componentwise_PRECISION<<<gridSize, blockSize>>>(eta, op->pbuf_gpu, l->num_inner_lattice_sites);
  cuda_pbp_su3_mvm_componentwise_PRECISION<<<2*gridSize, blockSize>>>(op->pbuf_gpu, op->Ds_componentwise_gpu[Z], op->prnZ_gpu,
                                                        op->neighbor_table_gpu, LatticeAxis::Z,
                                                        l->num_inner_lattice_sites);
  cuda_pbp_su3_Z_componentwise_PRECISION<<<gridSize, blockSize>>>(eta, op->pbuf_gpu, l->num_inner_lattice_sites);
  cuda_pbp_su3_mvm_componentwise_PRECISION<<<2*gridSize, blockSize>>>(op->pbuf_gpu, op->Ds_componentwise_gpu[Y], op->prnY_gpu,
                                                        op->neighbor_table_gpu, LatticeAxis::Y,
                                                        l->num_inner_lattice_sites);
  cuda_pbp_su3_Y_componentwise_PRECISION<<<gridSize, blockSize>>>(eta, op->pbuf_gpu, l->num_inner_lattice_sites);
  cuda_pbp_su3_mvm_componentwise_PRECISION<<<2*gridSize, blockSize>>>(op->pbuf_gpu, op->Ds_componentwise_gpu[X], op->prnX_gpu,
                                                        op->neighbor_table_gpu, LatticeAxis::X,
                                                        l->num_inner_lattice_sites);
  cuda_pbp_su3_X_componentwise_PRECISION<<<gridSize, blockSize>>>(eta, op->pbuf_gpu, l->num_inner_lattice_sites);
  cuda_safe_call(cudaDeviceSynchronize());

  cuda_ghost_wait_PRECISION(op->prpT_gpu, T, +1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_wait_PRECISION(op->prpZ_gpu, Z, +1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_wait_PRECISION(op->prpY_gpu, Y, +1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_wait_PRECISION(op->prpX_gpu, X, +1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_safe_call(cudaDeviceSynchronize());

  cuda_pbn_su3_T_componentwise_PRECISION<<<gridSize, blockSize>>>(eta, op->prpT_gpu, l->num_inner_lattice_sites);
  cuda_pbn_su3_Z_componentwise_PRECISION<<<gridSize, blockSize>>>(eta, op->prpZ_gpu, l->num_inner_lattice_sites);
  cuda_pbn_su3_Y_componentwise_PRECISION<<<gridSize, blockSize>>>(eta, op->prpY_gpu, l->num_inner_lattice_sites);
  cuda_pbn_su3_X_componentwise_PRECISION<<<gridSize, blockSize>>>(eta, op->prpX_gpu, l->num_inner_lattice_sites);
  cuda_safe_call(cudaDeviceSynchronize());

  PROF_PRECISION_STOP_UNTHREADED( _NC, 1 );
  endProfilingRange(profilingRangeOperator);
}


void cuda_d_plus_clover_PRECISION_awarempi(
  cuda_vector_PRECISION eta, cuda_vector_PRECISION phi, operator_PRECISION_struct *op,
  level_struct *l, struct Thread *threading ) {
  RangeHandleType profilingRangeOperator = startProfilingRange("d_plus_clover_PRECISION (CUDA)");

  // There is no need to deal with multiple streams, so the default stream (per thread) is used.
  // It shall be handled as though being an array of streams of size 1.
  cudaStream_t stream = CU_STREAM_PER_THREAD;
  cudaStream_t* const streams = &stream;

  constexpr size_t blockSize = 128;  // just a guess
  const size_t gridSize = minGridSizeForN(l->num_inner_lattice_sites, blockSize);

  const auto pbs = projection_buffer_size(l->num_lattice_site_var, l->num_lattice_sites);
  auto shift = to_cuda_cmplx_PRECISION(op->shift);

  // Apply Clover term
  if ( g.csw == 0.0 ) {
    cuda_vector_PRECISION_scale(eta, phi, shift, 0, l->inner_vector_size, l, _CUDA_SYNC, 0, streams);
  } else {
    _cuda_clover_PRECISION(eta, phi, op->clover_gpu, l->num_inner_lattice_sites, l);
  }
  
  PROF_PRECISION_START_UNTHREADED( _NC );

  // Project in positive directions
  cuda_prp_T_PRECISION<<<gridSize, blockSize>>>(op->prnT_gpu, phi, l->num_inner_lattice_sites);
  cuda_prp_Z_PRECISION<<<gridSize, blockSize>>>(op->prnZ_gpu, phi, l->num_inner_lattice_sites);
  cuda_prp_Y_PRECISION<<<gridSize, blockSize>>>(op->prnY_gpu, phi, l->num_inner_lattice_sites);
  cuda_prp_X_PRECISION<<<gridSize, blockSize>>>(op->prnX_gpu, phi, l->num_inner_lattice_sites);
  cuda_safe_call(cudaDeviceSynchronize());

  cuda_ghost_sendrecv_PRECISION(op->prnT_gpu, T, -1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_sendrecv_PRECISION(op->prnZ_gpu, Z, -1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_sendrecv_PRECISION(op->prnY_gpu, Y, -1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_sendrecv_PRECISION(op->prnX_gpu, X, -1, &(op->cuda_c), _FULL_SYSTEM, l);

  // project plus dir and multiply with U dagger
  cuda_prn_T_PRECISION<<<gridSize, blockSize>>>(op->pbuf_gpu, phi, l->num_inner_lattice_sites);
  cuda_prn_mvmh_PRECISION<<<2*gridSize, blockSize>>>(op->prpT_gpu, op->D_gpu, op->pbuf_gpu,
                                                     op->neighbor_table_gpu, LatticeAxis::T,
                                                     l->num_inner_lattice_sites);
  cuda_prn_Z_PRECISION<<<gridSize, blockSize>>>(op->pbuf_gpu, phi, l->num_inner_lattice_sites);
  cuda_prn_mvmh_PRECISION<<<2*gridSize, blockSize>>>(op->prpZ_gpu, op->D_gpu, op->pbuf_gpu,
                                                     op->neighbor_table_gpu, LatticeAxis::Z,
                                                     l->num_inner_lattice_sites);
  cuda_prn_Y_PRECISION<<<gridSize, blockSize>>>(op->pbuf_gpu, phi, l->num_inner_lattice_sites);
  cuda_prn_mvmh_PRECISION<<<2*gridSize, blockSize>>>(op->prpY_gpu, op->D_gpu, op->pbuf_gpu,
                                                     op->neighbor_table_gpu, LatticeAxis::Y,
                                                     l->num_inner_lattice_sites);
  cuda_prn_X_PRECISION<<<gridSize, blockSize>>>(op->pbuf_gpu, phi, l->num_inner_lattice_sites);
  cuda_prn_mvmh_PRECISION<<<2*gridSize, blockSize>>>(op->prpX_gpu, op->D_gpu, op->pbuf_gpu,
                                                     op->neighbor_table_gpu, LatticeAxis::X,
                                                     l->num_inner_lattice_sites);
  cuda_safe_call(cudaDeviceSynchronize());

  cuda_ghost_sendrecv_PRECISION(op->prpT_gpu, T, +1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_sendrecv_PRECISION(op->prpZ_gpu, Z, +1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_sendrecv_PRECISION(op->prpY_gpu, Y, +1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_sendrecv_PRECISION(op->prpX_gpu, X, +1, &(op->cuda_c), _FULL_SYSTEM, l);

  cuda_ghost_wait_PRECISION(op->prnT_gpu, T, -1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_wait_PRECISION(op->prnZ_gpu, Z, -1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_wait_PRECISION(op->prnY_gpu, Y, -1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_wait_PRECISION(op->prnX_gpu, X, -1, &(op->cuda_c), _FULL_SYSTEM, l);

  cuda_pbp_su3_mvm_PRECISION<<<2*gridSize, blockSize>>>(op->pbuf_gpu, op->D_gpu, op->prnT_gpu,
                                                        op->neighbor_table_gpu, LatticeAxis::T,
                                                        l->num_inner_lattice_sites);
  cuda_pbp_su3_T_PRECISION<<<gridSize, blockSize>>>(eta, op->pbuf_gpu, l->num_inner_lattice_sites);
  cuda_pbp_su3_mvm_PRECISION<<<2*gridSize, blockSize>>>(op->pbuf_gpu, op->D_gpu, op->prnZ_gpu,
                                                        op->neighbor_table_gpu, LatticeAxis::Z,
                                                        l->num_inner_lattice_sites);
  cuda_pbp_su3_Z_PRECISION<<<gridSize, blockSize>>>(eta, op->pbuf_gpu, l->num_inner_lattice_sites);
  cuda_pbp_su3_mvm_PRECISION<<<2*gridSize, blockSize>>>(op->pbuf_gpu, op->D_gpu, op->prnY_gpu,
                                                        op->neighbor_table_gpu, LatticeAxis::Y,
                                                        l->num_inner_lattice_sites);
  cuda_pbp_su3_Y_PRECISION<<<gridSize, blockSize>>>(eta, op->pbuf_gpu, l->num_inner_lattice_sites);
  cuda_pbp_su3_mvm_PRECISION<<<2*gridSize, blockSize>>>(op->pbuf_gpu, op->D_gpu, op->prnX_gpu,
                                                        op->neighbor_table_gpu, LatticeAxis::X,
                                                        l->num_inner_lattice_sites);
  cuda_pbp_su3_X_PRECISION<<<gridSize, blockSize>>>(eta, op->pbuf_gpu, l->num_inner_lattice_sites);

  cuda_ghost_wait_PRECISION(op->prpT_gpu, T, +1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_wait_PRECISION(op->prpZ_gpu, Z, +1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_wait_PRECISION(op->prpY_gpu, Y, +1, &(op->cuda_c), _FULL_SYSTEM, l);
  cuda_ghost_wait_PRECISION(op->prpX_gpu, X, +1, &(op->cuda_c), _FULL_SYSTEM, l);

  cuda_pbn_su3_T_PRECISION<<<gridSize, blockSize>>>(eta, op->prpT_gpu, l->num_inner_lattice_sites);
  cuda_pbn_su3_Z_PRECISION<<<gridSize, blockSize>>>(eta, op->prpZ_gpu, l->num_inner_lattice_sites);
  cuda_pbn_su3_Y_PRECISION<<<gridSize, blockSize>>>(eta, op->prpY_gpu, l->num_inner_lattice_sites);
  cuda_pbn_su3_X_PRECISION<<<gridSize, blockSize>>>(eta, op->prpX_gpu, l->num_inner_lattice_sites);
  cuda_safe_call(cudaDeviceSynchronize());
  
  PROF_PRECISION_STOP_UNTHREADED( _NC, 1 );
  endProfilingRange(profilingRangeOperator);
}

void cuda_d_plus_clover_PRECISION_naive(
  cuda_vector_PRECISION eta, cuda_vector_PRECISION phi, operator_PRECISION_struct *op,
  level_struct *l, struct Thread *threading ) {
  RangeHandleType profilingRangeOperator = startProfilingRange("d_plus_clover_PRECISION (CUDA)");

  // There is no need to deal with multiple streams, so the default stream (per thread) is used.
  // It shall be handled as though being an array of streams of size 1.
  cudaStream_t stream = CU_STREAM_PER_THREAD;
  cudaStream_t* const streams = &stream;

  constexpr size_t blockSize = 128;  // just a guess
  const size_t gridSize = minGridSizeForN(l->num_inner_lattice_sites, blockSize);

  const auto pbs = projection_buffer_size(l->num_lattice_site_var, l->num_lattice_sites);
  auto shift = to_cuda_cmplx_PRECISION(op->shift);

  // Apply Clover term
  if ( g.csw == 0.0 ) {
    cuda_vector_PRECISION_scale(eta, phi, shift, 0, l->inner_vector_size, l, _CUDA_SYNC, 0, streams);
  } else {
    _cuda_clover_PRECISION(eta, phi, op->clover_gpu, l->num_inner_lattice_sites, l);
  }
  
  PROF_PRECISION_START_UNTHREADED( _NC );

  // Project in positive directions
  cuda_prp_T_PRECISION<<<gridSize, blockSize>>>(op->prnT_gpu, phi, l->num_inner_lattice_sites);
  cuda_prp_Z_PRECISION<<<gridSize, blockSize>>>(op->prnZ_gpu, phi, l->num_inner_lattice_sites);
  cuda_prp_Y_PRECISION<<<gridSize, blockSize>>>(op->prnY_gpu, phi, l->num_inner_lattice_sites);
  cuda_prp_X_PRECISION<<<gridSize, blockSize>>>(op->prnX_gpu, phi, l->num_inner_lattice_sites);
  cuda_safe_call(cudaDeviceSynchronize());

  // start communication in negative direction
  cuda_vector_PRECISION_copy(op->prnT, op->prnT_gpu, 0, l->inner_vector_size/2, l, _D2H, _CUDA_SYNC, 0, streams);
  cuda_vector_PRECISION_copy(op->prnZ, op->prnZ_gpu, 0, l->inner_vector_size/2, l, _D2H, _CUDA_SYNC, 0, streams);
  cuda_vector_PRECISION_copy(op->prnY, op->prnY_gpu, 0, l->inner_vector_size/2, l, _D2H, _CUDA_SYNC, 0, streams);
  cuda_vector_PRECISION_copy(op->prnX, op->prnX_gpu, 0, l->inner_vector_size/2, l, _D2H, _CUDA_SYNC, 0, streams);

  ghost_sendrecv_PRECISION( op->prnT, T, -1, &(op->c), _FULL_SYSTEM, l );
  ghost_sendrecv_PRECISION( op->prnZ, Z, -1, &(op->c), _FULL_SYSTEM, l );
  ghost_sendrecv_PRECISION( op->prnY, Y, -1, &(op->c), _FULL_SYSTEM, l );
  ghost_sendrecv_PRECISION( op->prnX, X, -1, &(op->c), _FULL_SYSTEM, l );

  // project plus dir and multiply with U dagger
  cuda_prn_T_PRECISION<<<gridSize, blockSize>>>(op->pbuf_gpu, phi, l->num_inner_lattice_sites);
  cuda_prn_mvmh_PRECISION<<<2*gridSize, blockSize>>>(op->prpT_gpu, op->D_gpu, op->pbuf_gpu,
                                                     op->neighbor_table_gpu, LatticeAxis::T,
                                                     l->num_inner_lattice_sites);
  cuda_prn_Z_PRECISION<<<gridSize, blockSize>>>(op->pbuf_gpu, phi, l->num_inner_lattice_sites);
  cuda_prn_mvmh_PRECISION<<<2*gridSize, blockSize>>>(op->prpZ_gpu, op->D_gpu, op->pbuf_gpu,
                                                     op->neighbor_table_gpu, LatticeAxis::Z,
                                                     l->num_inner_lattice_sites);
  cuda_prn_Y_PRECISION<<<gridSize, blockSize>>>(op->pbuf_gpu, phi, l->num_inner_lattice_sites);
  cuda_prn_mvmh_PRECISION<<<2*gridSize, blockSize>>>(op->prpY_gpu, op->D_gpu, op->pbuf_gpu,
                                                     op->neighbor_table_gpu, LatticeAxis::Y,
                                                     l->num_inner_lattice_sites);
  cuda_prn_X_PRECISION<<<gridSize, blockSize>>>(op->pbuf_gpu, phi, l->num_inner_lattice_sites);
  cuda_prn_mvmh_PRECISION<<<2*gridSize, blockSize>>>(op->prpX_gpu, op->D_gpu, op->pbuf_gpu,
                                                     op->neighbor_table_gpu, LatticeAxis::X,
                                                     l->num_inner_lattice_sites);
  cuda_safe_call(cudaDeviceSynchronize());

  // start communication in positive direction
  cuda_vector_PRECISION_copy(op->prpT, op->prpT_gpu, 0, pbs, l, _D2H, _CUDA_SYNC, 0, streams);
  cuda_vector_PRECISION_copy(op->prpZ, op->prpZ_gpu, 0, pbs, l, _D2H, _CUDA_SYNC, 0, streams);
  cuda_vector_PRECISION_copy(op->prpY, op->prpY_gpu, 0, pbs, l, _D2H, _CUDA_SYNC, 0, streams);
  cuda_vector_PRECISION_copy(op->prpX, op->prpX_gpu, 0, pbs, l, _D2H, _CUDA_SYNC, 0, streams);
  ghost_sendrecv_PRECISION( op->prpT, T, +1, &(op->c), _FULL_SYSTEM, l );
  ghost_sendrecv_PRECISION( op->prpZ, Z, +1, &(op->c), _FULL_SYSTEM, l );
  ghost_sendrecv_PRECISION( op->prpY, Y, +1, &(op->c), _FULL_SYSTEM, l );
  ghost_sendrecv_PRECISION( op->prpX, X, +1, &(op->c), _FULL_SYSTEM, l );
  // wait for communication in negative direction
  ghost_wait_PRECISION( op->prnT, T, -1, &(op->c), _FULL_SYSTEM, l );
  ghost_wait_PRECISION( op->prnZ, Z, -1, &(op->c), _FULL_SYSTEM, l );
  ghost_wait_PRECISION( op->prnY, Y, -1, &(op->c), _FULL_SYSTEM, l );
  ghost_wait_PRECISION( op->prnX, X, -1, &(op->c), _FULL_SYSTEM, l );
  cuda_vector_PRECISION_copy(op->prnT_gpu, op->prnT, 0, pbs, l, _H2D, _CUDA_SYNC, 0, streams);
  cuda_vector_PRECISION_copy(op->prnZ_gpu, op->prnZ, 0, pbs, l, _H2D, _CUDA_SYNC, 0, streams);
  cuda_vector_PRECISION_copy(op->prnY_gpu, op->prnY, 0, pbs, l, _H2D, _CUDA_SYNC, 0, streams);
  cuda_vector_PRECISION_copy(op->prnX_gpu, op->prnX, 0, pbs, l, _H2D, _CUDA_SYNC, 0, streams);

  cuda_pbp_su3_mvm_PRECISION<<<2*gridSize, blockSize>>>(op->pbuf_gpu, op->D_gpu, op->prnT_gpu,
                                                        op->neighbor_table_gpu, LatticeAxis::T,
                                                        l->num_inner_lattice_sites);
  cuda_pbp_su3_T_PRECISION<<<gridSize, blockSize>>>(eta, op->pbuf_gpu, l->num_inner_lattice_sites);
  cuda_pbp_su3_mvm_PRECISION<<<2*gridSize, blockSize>>>(op->pbuf_gpu, op->D_gpu, op->prnZ_gpu,
                                                        op->neighbor_table_gpu, LatticeAxis::Z,
                                                        l->num_inner_lattice_sites);
  cuda_pbp_su3_Z_PRECISION<<<gridSize, blockSize>>>(eta, op->pbuf_gpu, l->num_inner_lattice_sites);
  cuda_pbp_su3_mvm_PRECISION<<<2*gridSize, blockSize>>>(op->pbuf_gpu, op->D_gpu, op->prnY_gpu,
                                                        op->neighbor_table_gpu, LatticeAxis::Y,
                                                        l->num_inner_lattice_sites);
  cuda_pbp_su3_Y_PRECISION<<<gridSize, blockSize>>>(eta, op->pbuf_gpu, l->num_inner_lattice_sites);
  cuda_pbp_su3_mvm_PRECISION<<<2*gridSize, blockSize>>>(op->pbuf_gpu, op->D_gpu, op->prnX_gpu,
                                                        op->neighbor_table_gpu, LatticeAxis::X,
                                                        l->num_inner_lattice_sites);
  cuda_pbp_su3_X_PRECISION<<<gridSize, blockSize>>>(eta, op->pbuf_gpu, l->num_inner_lattice_sites);

  // wait for communication in positive direction
  ghost_wait_PRECISION( op->prpT, T, +1, &(op->c), _FULL_SYSTEM, l );
  ghost_wait_PRECISION( op->prpZ, Z, +1, &(op->c), _FULL_SYSTEM, l );
  ghost_wait_PRECISION( op->prpY, Y, +1, &(op->c), _FULL_SYSTEM, l );
  ghost_wait_PRECISION( op->prpX, X, +1, &(op->c), _FULL_SYSTEM, l );
  cuda_vector_PRECISION_copy(op->prpT_gpu, op->prpT, 0, pbs, l, _H2D, _CUDA_SYNC, 0, streams);
  cuda_vector_PRECISION_copy(op->prpZ_gpu, op->prpZ, 0, pbs, l, _H2D, _CUDA_SYNC, 0, streams);
  cuda_vector_PRECISION_copy(op->prpY_gpu, op->prpY, 0, pbs, l, _H2D, _CUDA_SYNC, 0, streams);
  cuda_vector_PRECISION_copy(op->prpX_gpu, op->prpX, 0, pbs, l, _H2D, _CUDA_SYNC, 0, streams);

  cuda_pbn_su3_T_PRECISION<<<gridSize, blockSize>>>(eta, op->prpT_gpu, l->num_inner_lattice_sites);
  cuda_pbn_su3_Z_PRECISION<<<gridSize, blockSize>>>(eta, op->prpZ_gpu, l->num_inner_lattice_sites);
  cuda_pbn_su3_Y_PRECISION<<<gridSize, blockSize>>>(eta, op->prpY_gpu, l->num_inner_lattice_sites);
  cuda_pbn_su3_X_PRECISION<<<gridSize, blockSize>>>(eta, op->prpX_gpu, l->num_inner_lattice_sites);
  cuda_safe_call(cudaDeviceSynchronize());
  
  PROF_PRECISION_STOP_UNTHREADED( _NC, 1 );
  endProfilingRange(profilingRangeOperator);
}

extern "C" void cuda_d_plus_clover_PRECISION_vectorwrapper(vector_PRECISION eta, complex_PRECISION const *phi, operator_PRECISION_struct *op,
                                         level_struct *l, struct Thread *threading){
  // Performance is achieved through GPU acceleration and not multi-threading.
  START_UNTHREADED_FUNCTION(threading)
  if (l->depth != 0) {
    // It is not properly tested that this integrates properly with the way memory is allocated
    // in coarser grids. Also the interactions with the other CUDA AMG code is not yet properly
    // tested.
    error0("cuda_d_plus_clover_PRECISION_vectorwrapper may only be called from the finest level.");
  }
  cuda_vector_PRECISION eta_gpu, eta_componentwise_gpu, phi_gpu, phi_componentwise_gpu;
  eta_gpu = op->w_gpu;
  eta_componentwise_gpu = op->w_componentwise_gpu;
  phi_gpu = op->x_gpu;
  phi_componentwise_gpu = op->x_componentwise_gpu;
  cudaStream_t stream = CU_STREAM_PER_THREAD;
  cudaStream_t* const streams = &stream;
  
  cuda_vector_PRECISION_copy(phi_gpu, phi, 0, l->inner_vector_size, l, _H2D, _CUDA_SYNC, 0, streams);
  const uint gridSize = minGridSizeForN(l->num_inner_lattice_sites, diracDefaultBlockSize);
  reorderArrayByComponent<<<gridSize, diracDefaultBlockSize>>>(
    phi_componentwise_gpu, phi_gpu, l->num_lattice_site_var, l->num_inner_lattice_sites);
  cuda_safe_call(cudaDeviceSynchronize());

  cuda_d_plus_clover_PRECISION(eta_componentwise_gpu, phi_componentwise_gpu, op, l, threading);

  reorderArrayByChunks<<<gridSize, diracDefaultBlockSize>>>(
    eta_gpu, eta_componentwise_gpu, l->num_lattice_site_var, l->num_inner_lattice_sites);
  // implicit device synchronize
  cuda_vector_PRECISION_copy(eta, eta_gpu, 0, l->inner_vector_size, l, _D2H, _CUDA_SYNC, 0, streams);
  END_UNTHREADED_FUNCTION(threading)
}
#endif
