#ifdef CUDA_OPT


#include <mpi.h>

extern "C"{

  #define IMPORT_FROM_EXTERN_C
  #include "main.h"
  #undef IMPORT_FROM_EXTERN_C

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
  gamma_val = s->gamma_info_vals + dir*4 + spin;
  gamma_coo = s->gamma_info_coo  + dir*4 + spin;

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
                                                                int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
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
    //clover_PRECISION( leta, lphi, clover, 12*n, l, threading ); 
    //TODO
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

#endif
