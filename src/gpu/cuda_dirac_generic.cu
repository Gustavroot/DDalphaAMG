#include <mpi.h>

extern "C"{

  #define IMPORT_FROM_EXTERN_C
  #include "main.h"
  #undef IMPORT_FROM_EXTERN_C

}

#ifdef CUDA_OPT


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

    cuda_site_clover_PRECISION<<< nr_threads/threads_per_cublock, threads_per_cublock,
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


//void sse_clover_PRECISION( vector_PRECISION eta, vector_PRECISION phi, operator_PRECISION_struct *op,
//                           int start, int end, level_struct *l, struct Thread *threading );

/*
extern "C" void
d_plus_clover_PRECISION_CUDA(					cuda_vector_PRECISION eta_gpu, cuda_vector_PRECISION phi_gpu, operator_PRECISION_struct *op,
				                                level_struct *l, struct Thread *threading ) {

  cudaStream_t *streams_gmres = (l->p_PRECISION).streams;

  vector_PRECISION eta=NULL, phi=NULL;
  eta = (complex_PRECISION*) malloc( l->inner_vector_size * sizeof(complex_PRECISION) );
  phi = (complex_PRECISION*) malloc( l->inner_vector_size * sizeof(complex_PRECISION) );

  copy_2_cpu_PRECISION_v3(eta, eta_gpu, phi, phi_gpu, l);

  //PROF_PRECISION_START( _SC, threading );
  //START_LOCKED_MASTER(threading)

  //coarse_self_couplings_PRECISION( eta, phi, op->clover, l->inner_vector_size, l );

  //coarse_self_couplings_PRECISION_CUDA( y_start*offset, x+start*offset, op->clover_gpu+start*(offset*offset+offset)/2, (end-start)*offset, l, threading );
  //coarse_self_couplings_PRECISION_CUDA( eta_gpu, phi_gpu, op->clover_gpu, l->inner_vector_size, l, threading );

  //cudaDeviceSynchronize();

  //copy_2_cpu_PRECISION_v2(eta, eta_gpu, phi, phi_gpu, l);

  //END_LOCKED_MASTER(threading)
  //PROF_PRECISION_STOP( _SC, 1, threading );
  //PROF_PRECISION_START( _NC, threading );
  //coarse_hopping_term_PRECISION( eta, phi, op, _FULL_SYSTEM, l, threading );
  //PROF_PRECISION_STOP( _NC, 1, threading );

  //printf0("WITHIN d_plus_clover_PRECISION(...) !!, depth=%d \n", l->depth);

  //----------------------------

  int n = l->num_inner_lattice_sites, *neighbor = op->neighbor_table, start, end;
#ifndef OPTIMIZED_NEIGHBOR_COUPLING_PRECISION
  int i, j, *nb_pt;
  complex_PRECISION pbuf[6];
  vector_PRECISION phi_pt, eta_pt, end_pt;
  config_PRECISION D_pt;
#endif

  compute_core_start_end(0, 12*n, &start, &end, l, threading );

  SYNC_MASTER_TO_ALL(threading)

  if ( g.csw == 0.0 ) {
    vector_PRECISION_scale( eta, phi, op->shift, start, end, l );
  } else {
    clover_PRECISION( eta+start, phi+start, op->clover+((start/12)*42), end-start, l, threading );
  }

  START_MASTER(threading)
  PROF_PRECISION_START( _NC );
  END_MASTER(threading)

#ifdef OPTIMIZED_NEIGHBOR_COUPLING_PRECISION
  complex_PRECISION *prn[4] = { op->prnT, op->prnZ, op->prnY, op->prnX };
  prp_PRECISION( prn, phi, start, end );
#else
  for ( i=start/2, phi_pt=phi+start; i<end/2; i+=6, phi_pt+=12 ) {
    prp_T_PRECISION( op->prnT+i, phi_pt );
    prp_Z_PRECISION( op->prnZ+i, phi_pt );
    prp_Y_PRECISION( op->prnY+i, phi_pt );
    prp_X_PRECISION( op->prnX+i, phi_pt );
  }
#endif
  // start communication in negative direction
  START_LOCKED_MASTER(threading)
  ghost_sendrecv_PRECISION( op->prnT, T, -1, &(op->c), _FULL_SYSTEM, l );
  ghost_sendrecv_PRECISION( op->prnZ, Z, -1, &(op->c), _FULL_SYSTEM, l );
  ghost_sendrecv_PRECISION( op->prnY, Y, -1, &(op->c), _FULL_SYSTEM, l );
  ghost_sendrecv_PRECISION( op->prnX, X, -1, &(op->c), _FULL_SYSTEM, l );
  END_LOCKED_MASTER(threading)

  // project plus dir and multiply with U dagger
#ifdef OPTIMIZED_NEIGHBOR_COUPLING_PRECISION
  complex_PRECISION *prp[4] = { op->prpT, op->prpZ, op->prpY, op->prpX };
  prn_su3_PRECISION( prp, phi, op, neighbor, start, end );
#else
  for ( phi_pt=phi+start, end_pt=phi+end, D_pt = op->D+(start*3), nb_pt=neighbor+((start/12)*4); phi_pt<end_pt; phi_pt+=12 ) {
    // T dir
    j = 6*(*nb_pt); nb_pt++;
    prn_T_PRECISION( pbuf, phi_pt );
    mvmh_PRECISION( op->prpT+j, D_pt, pbuf );
    mvmh_PRECISION( op->prpT+j+3, D_pt, pbuf+3 ); D_pt += 9;
    // Z dir
    j = 6*(*nb_pt); nb_pt++;
    prn_Z_PRECISION( pbuf, phi_pt );
    mvmh_PRECISION( op->prpZ+j, D_pt, pbuf );
    mvmh_PRECISION( op->prpZ+j+3, D_pt, pbuf+3 ); D_pt += 9;
    // Y dir
    j = 6*(*nb_pt); nb_pt++;
    prn_Y_PRECISION( pbuf, phi_pt );
    mvmh_PRECISION( op->prpY+j, D_pt, pbuf );
    mvmh_PRECISION( op->prpY+j+3, D_pt, pbuf+3 ); D_pt += 9;
    // X dir
    j = 6*(*nb_pt); nb_pt++;
    prn_X_PRECISION( pbuf, phi_pt );
    mvmh_PRECISION( op->prpX+j, D_pt, pbuf );
    mvmh_PRECISION( op->prpX+j+3, D_pt, pbuf+3 ); D_pt += 9;
  }
#endif

  // start communication in positive direction
  START_LOCKED_MASTER(threading)
  ghost_sendrecv_PRECISION( op->prpT, T, +1, &(op->c), _FULL_SYSTEM, l );
  ghost_sendrecv_PRECISION( op->prpZ, Z, +1, &(op->c), _FULL_SYSTEM, l );
  ghost_sendrecv_PRECISION( op->prpY, Y, +1, &(op->c), _FULL_SYSTEM, l );
  ghost_sendrecv_PRECISION( op->prpX, X, +1, &(op->c), _FULL_SYSTEM, l );
  // wait for communication in negative direction
  ghost_wait_PRECISION( op->prnT, T, -1, &(op->c), _FULL_SYSTEM, l );
  ghost_wait_PRECISION( op->prnZ, Z, -1, &(op->c), _FULL_SYSTEM, l );
  ghost_wait_PRECISION( op->prnY, Y, -1, &(op->c), _FULL_SYSTEM, l );
  ghost_wait_PRECISION( op->prnX, X, -1, &(op->c), _FULL_SYSTEM, l );
  END_LOCKED_MASTER(threading)

  // multiply with U and lift up minus dir
#ifdef OPTIMIZED_NEIGHBOR_COUPLING_PRECISION
  su3_pbp_PRECISION( eta, prn, op, neighbor, start, end );
#else
  for ( eta_pt=eta+start, end_pt=eta+end, D_pt = op->D+start*3, nb_pt=neighbor+(start/12)*4; eta_pt<end_pt; eta_pt+=12 ) {
    // T dir
    j = 6*(*nb_pt); nb_pt++;
    mvm_PRECISION( pbuf, D_pt, op->prnT+j );
    mvm_PRECISION( pbuf+3, D_pt, op->prnT+j+3 );
    pbp_su3_T_PRECISION( pbuf, eta_pt ); D_pt += 9;
    // Z dir
    j = 6*(*nb_pt); nb_pt++;
    mvm_PRECISION( pbuf, D_pt, op->prnZ+j );
    mvm_PRECISION( pbuf+3, D_pt, op->prnZ+j+3 );
    pbp_su3_Z_PRECISION( pbuf, eta_pt ); D_pt += 9;
    // Y dir
    j = 6*(*nb_pt); nb_pt++;
    mvm_PRECISION( pbuf, D_pt, op->prnY+j );
    mvm_PRECISION( pbuf+3, D_pt, op->prnY+j+3 );
    pbp_su3_Y_PRECISION( pbuf, eta_pt ); D_pt += 9;
    // X dir
    j = 6*(*nb_pt); nb_pt++;
    mvm_PRECISION( pbuf, D_pt, op->prnX+j );
    mvm_PRECISION( pbuf+3, D_pt, op->prnX+j+3 );
    pbp_su3_X_PRECISION( pbuf, eta_pt ); D_pt += 9;
  }
#endif

  // wait for communication in positive direction
  START_LOCKED_MASTER(threading)
  ghost_wait_PRECISION( op->prpT, T, +1, &(op->c), _FULL_SYSTEM, l );
  ghost_wait_PRECISION( op->prpZ, Z, +1, &(op->c), _FULL_SYSTEM, l );
  ghost_wait_PRECISION( op->prpY, Y, +1, &(op->c), _FULL_SYSTEM, l );
  ghost_wait_PRECISION( op->prpX, X, +1, &(op->c), _FULL_SYSTEM, l );
  END_LOCKED_MASTER(threading)

  // lift up plus dir
#ifdef OPTIMIZED_NEIGHBOR_COUPLING_PRECISION
  pbn_PRECISION( eta, prp, start, end );
#else
  for ( i=start/2, eta_pt=eta+start; i<end/2; i+=6, eta_pt+=12 ) {
    pbn_su3_T_PRECISION( op->prpT+i, eta_pt );
    pbn_su3_Z_PRECISION( op->prpZ+i, eta_pt );
    pbn_su3_Y_PRECISION( op->prpY+i, eta_pt );
    pbn_su3_X_PRECISION( op->prpX+i, eta_pt );
  }
#endif

  START_MASTER(threading)
  PROF_PRECISION_STOP( _NC, 1 );
  END_MASTER(threading)

  SYNC_MASTER_TO_ALL(threading)

  //----------------------------

  cuda_vector_PRECISION_copy( (void*)eta_gpu, (void*)eta, 0, l->inner_vector_size, l, _H2D, _CUDA_SYNC,
                              threading->core, streams_gmres );
  cuda_vector_PRECISION_copy( (void*)phi_gpu, (void*)phi, 0, l->inner_vector_size, l, _H2D, _CUDA_SYNC,
                              threading->core, streams_gmres );

  free(eta);
  free(phi);

}
*/


#endif
