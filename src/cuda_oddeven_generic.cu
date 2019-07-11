#ifdef CUDA_OPT

#include <mpi.h>

#ifdef CUDA_OPT
  #include <cuComplex.h>
  #include <cuda.h>
  #include <cuda_runtime.h>
#endif

extern "C"{

  #define IMPORT_FROM_EXTERN_C
  #include "main.h"
  #undef IMPORT_FROM_EXTERN_C

}



// Pre-definitions of CUDA functions to be called from the CUDA kernels, force inlines
__forceinline__ __device__ void _cuda_block_diag_oo_inv_PRECISION(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, schwarz_PRECISION_struct_on_gpu *s, int idx, cu_config_PRECISION *clov_vect, int csw);
__forceinline__ __device__ void _cuda_block_n_hopping_term_PRECISION_plus(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, int amount, schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf, int ext_dir);


__global__ void cuda_block_diag_oo_inv_PRECISION( cu_cmplx_PRECISION* phi, cu_cmplx_PRECISION* r, cu_cmplx_PRECISION* latest_iter, \
                                                  schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                  int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                  int num_latt_site_var, block_struct* block ){

  int i, idx, DD_block_id, block_id, cublocks_per_DD_block, cu_block_ID, size_D_oeclov, start;

  int nr_block_even_sites, nr_block_odd_sites;
  nr_block_even_sites = s->num_block_even_sites;
  nr_block_odd_sites = s->num_block_odd_sites;

  idx = threadIdx.x + blockDim.x * blockIdx.x;

  // not really a DD block id, but rather a linear counting of a grouping (per DD block) of CUDA threads
  DD_block_id = idx/nr_threads_per_DD_block;

  // offsetting idx to make it zero at the beginning of the threads living within a DD block
  idx = idx%nr_threads_per_DD_block;

  // this int will be the ACTUAL DD block ID, in the sense of accessing data from e.g. block_struct* block
  block_id = DD_blocks_to_compute[DD_block_id];

  cublocks_per_DD_block = nr_threads_per_DD_block/blockDim.x;

  // This serves as a substitute of blockIdx.x, to have a more
  // local and DD-block treatment more independent of the other DD blocks
  cu_block_ID = blockIdx.x%cublocks_per_DD_block;

  // the size of the local matrix to apply
  size_D_oeclov = (csw!=0) ? 72 : 12;

  // this is the DD-block start of the spinors (phi, r, latest_iter and temporary ones)
  start = block[block_id].start * num_latt_site_var;

  // tmp spinors
  cu_cmplx_PRECISION** tmp = s->oe_buf;
  cu_cmplx_PRECISION* tmp2 = tmp[2];
  cu_cmplx_PRECISION* tmp3 = tmp[3];

  // shift all relevant spinors, to treat them all locally (in the DD-block sense)
  phi += start;
  r += start;
  latest_iter += start;
  tmp2 += start;
  tmp3 += start;

  // this operator is stored in column form!
  cu_config_PRECISION *op_oe_vect = s->op.oe_clover_vectorized;
  // FIXME: instead of 12, use num_latt_site_var
  op_oe_vect += (start/12)*size_D_oeclov;

  extern __shared__ cu_cmplx_PRECISION shared_data[];

  //TODO: can we trim the use of shared memory ... ?

  // a part of shared_memory is dedicated to even sites, the rest to odd sites
  cu_cmplx_PRECISION *shared_data_even = shared_data;
  cu_cmplx_PRECISION *shared_data_odd = shared_data + 2*(2*blockDim.x);
  shared_data_odd = (cu_cmplx_PRECISION*) ( (cu_config_PRECISION*)shared_data_odd + size_D_oeclov*(blockDim.x/6) );

  cu_cmplx_PRECISION *phi_b_e, *r_b_e;
  cu_cmplx_PRECISION *phi_b_o, *r_b_o, *tmp_2_o;

  cu_config_PRECISION *clov_vect_b_e;
  cu_config_PRECISION *clov_vect_b_o;

  //the following are 'bare' values, i.e. with respect to the 0th element within a CUDA block
  // EVEN
  phi_b_e = shared_data_even;
  r_b_e = shared_data_even + 1*(2*blockDim.x);
  // ODD
  phi_b_o = shared_data_odd;
  r_b_o = shared_data_odd + 1*(2*blockDim.x);
  tmp_2_o = shared_data_odd + 2*(2*blockDim.x);

  clov_vect_b_e = (cu_config_PRECISION*)shared_data_even + 2*(2*blockDim.x);
  clov_vect_b_o = (cu_config_PRECISION*)shared_data_odd + 3*(2*blockDim.x);

  // partial summary so far:
  //    ** phi_b_e has a memory reservation of size 2*6*(blockDim.x/6)
  //    ** same for r_b_e, tmp_2_e, tmp_3_e
  //    ** clov_vect_b_e has a memory reservation of size size_D_oeclov*(blockDim.x/6)
  //    ** equivalently, we can say the same about the *_o variables

  // copy phi, r and s->op.oe_clover_vectorized into shared_data memory
  // even
  if(idx < 6*nr_block_even_sites){
    for(i=0; i<2; i++){
      phi_b_e[blockDim.x*i + threadIdx.x] = ( phi + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
      r_b_e[blockDim.x*i + threadIdx.x] = ( r + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
    }

    // the factor of 12 comes from: 72*16=1152 ----> 1152/96=12. This implies 12 dumps of data into shared_memory
    // TODO: generalize this factor of 12, if possible
    for(i=0; i<12; i++){
      clov_vect_b_e[blockDim.x*i + threadIdx.x] = ( op_oe_vect + cu_block_ID*blockDim.x*12 + blockDim.x*i + threadIdx.x )[0];
    }
  }
  // odd
  if(idx < 6*nr_block_odd_sites){
    for(i=0; i<2; i++){
      phi_b_o[blockDim.x*i + threadIdx.x] = ( phi + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
      r_b_o[blockDim.x*i + threadIdx.x] = ( r + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
    }

    //the factor of 12 comes from: 72*16=1152 ----> 1152/96=12. This implies 12 dumps of data into shared_memory
    for(i=0; i<12; i++){
      clov_vect_b_o[blockDim.x*i + threadIdx.x] = ( op_oe_vect + 72*nr_block_even_sites + cu_block_ID*blockDim.x*12 + blockDim.x*i + threadIdx.x )[0];
    }
  }

  __syncthreads();

  // FUNCTION: chi = D_{oo}^{-1} * eta_{0}
  if(idx < 6*nr_block_odd_sites){
    _cuda_block_diag_oo_inv_PRECISION(tmp_2_o, r_b_o, start, s, idx, clov_vect_b_o, csw);
  }

  __syncthreads();

  // update tmp2 and tmp3
  // even
  if(idx < 6*nr_block_even_sites){
    for(i=0; i<2; i++){
      ( tmp2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = make_cu_cmplx_PRECISION(0.0,0.0);
      ( tmp3 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = r_b_e[blockDim.x*i + threadIdx.x];
    }
  }
  // odd
  if(idx < 6*nr_block_odd_sites){
    for(i=0; i<2; i++){
      ( tmp2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = tmp_2_o[blockDim.x*i + threadIdx.x];
      ( tmp3 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = r_b_o[blockDim.x*i + threadIdx.x];
    }
  }

}



__forceinline__ __device__ void _cuda_block_diag_oo_inv_PRECISION(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, \
                                                                  schwarz_PRECISION_struct_on_gpu *s, int idx, cu_config_PRECISION *op_clov_vect, int csw){

  // FIXME: extend code to include case csw==0

  int local_idx = idx%6;
  // this offset is per site and within each CUDA block
  int site_offset = (threadIdx.x/6)*12;

  cu_cmplx_PRECISION* eta_site = eta + site_offset;
  cu_cmplx_PRECISION* phi_site = phi + site_offset;
  cu_config_PRECISION* op_clov_vect_site = op_clov_vect + (site_offset/12)*72;

  eta_site[ local_idx ] = make_cu_cmplx_PRECISION(0.0,0.0);
  eta_site[ 6 + local_idx ] = make_cu_cmplx_PRECISION(0.0,0.0);

  if( csw!=0 ){
    // Run over all the sites within this block, and perform: eta = op_clov_vect * phi
    // There are 16 sites per CUDA block, in the case of choosing blockDimx.x equal to 96

    // first compute upper half of vector for each site, then lower half
    for( int i=0; i<2; i++ ){
      // outter loop for matrix*vector double loop unrolled
      for( int j=0; j<6; j++ ){
        eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ], cu_cmul_PRECISION( (op_clov_vect_site + i*36)[local_idx + 6*j], phi_site[j + 6*i] ) );
      }
    }

  }
}


__global__ void cuda_block_n_hopping_term_PRECISION_plus( cu_cmplx_PRECISION* phi, cu_cmplx_PRECISION* r, cu_cmplx_PRECISION* latest_iter, \
                                                          schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                          int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                          int num_latt_site_var, block_struct* block, int ext_dir ){

  int idx, DD_block_id, block_id, start;

  int nr_block_even_sites;
  nr_block_even_sites = s->num_block_even_sites;

  idx = threadIdx.x + blockDim.x * blockIdx.x;

  // not really a DD block id, but rather a linear counting of a grouping (per DD block) of CUDA threads
  DD_block_id = idx/nr_threads_per_DD_block;

  // offsetting idx to make it zero at the beginning of the threads living within a DD block
  idx = idx%nr_threads_per_DD_block;

  // this int will be the ACTUAL DD block ID, in the sense of accessing data from e.g. block_struct* block
  block_id = DD_blocks_to_compute[DD_block_id];

  // this is the DD-block start of the spinors (phi, r, latest_iter and temporary ones)
  start = block[block_id].start * num_latt_site_var;

  cu_cmplx_PRECISION** tmp = s->oe_buf;
  cu_cmplx_PRECISION* tmp2 = tmp[2];
  cu_cmplx_PRECISION* tmp3 = tmp[3];

  tmp2 += start;
  tmp3 += start;

  extern __shared__ cu_cmplx_PRECISION shared_data[];

  // a part of shared_memory is dedicated to even sites, the rest to odd sites
  cu_cmplx_PRECISION *shared_data_even = shared_data;
  cu_cmplx_PRECISION *tmp_2_e;

  //the following are 'bare' values, i.e. with respect to the 0th element within a CUDA block
  // EVEN
  tmp_2_e = shared_data_even;
  if(idx < 6*nr_block_even_sites){
    tmp_2_e[threadIdx.x] = make_cu_cmplx_PRECISION(0.0,0.0);
    tmp_2_e[threadIdx.x + blockDim.x] = make_cu_cmplx_PRECISION(0.0,0.0);
  }

  _cuda_block_n_hopping_term_PRECISION_plus(tmp3, tmp2, start, _EVEN_SITES, s, idx, tmp_2_e, ext_dir);
}


__forceinline__ __device__ void _cuda_block_n_hopping_term_PRECISION_plus(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, int amount, schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf, int ext_dir){
  //if amount==0 then even sites, if amount==1 then odd sites, else no oddeven split

  //int dir; // dir '=' {0,1,2,3} = {T,Z,Y,X}
  //int dir, a1, a2, n1, n2, k=0, j=0, i=0, **index = s->oe_index, *ind, *neighbor = s->op.neighbor_table, loc_ind=idx%6, spin, w, *gamma_coo, idx_in_cublock = idx%blockDim.x;
  int dir, a1, n1, k=0, j=0, i=0, **index = s->oe_index, *ind, *neighbor = s->op.neighbor_table, loc_ind=idx%6, spin, w, *gamma_coo, idx_in_cublock = idx%blockDim.x;
  cu_cmplx_PRECISION *gamma_val;

  cu_cmplx_PRECISION *buf1, *buf2;
  buf += (idx_in_cublock/6)*12;
  buf1 = buf;
  buf2 = buf + 6;

  //TODO: implement a dynamic way of loading s->op.D into __shared__

  cu_config_PRECISION* D = s->op.D + (start/12)*36;
  cu_config_PRECISION* D_pt;

  cu_cmplx_PRECISION *leta, *lphi; //eta and phi are already shifted by 'start'

  // TODO: is there a reason for this not to be integrated with extra_dir ?
  dir = ext_dir;

  if( amount==_EVEN_SITES ){
    a1=0; n1=s->dir_length_even[dir]; //for the + part
    //a2=n1; n2=a2+s->dir_length_odd[dir]; //for the - part
  }
  else if( amount==_ODD_SITES ){
    a1=s->dir_length_even[dir]; n1=a1+s->dir_length_odd[dir];
    //a2=0; n2=a1;
  }
  else{
    //TODO
  }

  ind = index[dir];

  //less threads in charge of this portion of execution, compute contribution due to even sites
  if( idx<6*(n1-a1) ){
    i = idx/6 + a1;
    k = ind[i];
    j = neighbor[4*k+dir];
    D_pt = D + 36*k + 9*dir;

    lphi = phi + 12*j;
    leta = eta + 12*k;

    spin = (loc_ind/3)*2;
    //with this setup, gamma_val[0] gives spins 0 and 1, and gamma_val[1] spins 2 and 3
    gamma_val = s->gamma_info_vals + dir*4 + spin;
    gamma_coo = s->gamma_info_coo  + dir*4 + spin;

    // prp_T_PRECISION(...)
    buf1[ loc_ind ] = cu_csub_PRECISION( lphi[ loc_ind ], cu_cmul_PRECISION( gamma_val[0], lphi[ 3*gamma_coo[0] + loc_ind%3 ] ) );
  }

  __syncthreads();

  if( idx<6*(n1-a1) ){
    // nmvm_PRECISION(...), twice
    buf2[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
    for( w=0; w<3; w++ ){
      buf2[ loc_ind ] = cu_csub_PRECISION( buf2[ loc_ind ], cu_cmul_PRECISION( D_pt[ (loc_ind*3)%9 + w ], buf1[ (loc_ind/3)*3 + w ] ) );
    }
  }

  __syncthreads();

  if( idx<6*(n1-a1) ){
    // pbp_su3_T_PRECISION(...)
    leta[ loc_ind ] = cu_csub_PRECISION( leta[ loc_ind ], buf2[ loc_ind ] );
    leta[ 6 + loc_ind ] = cu_cadd_PRECISION( leta[ 6 + loc_ind ], cu_cmul_PRECISION( gamma_val[1], buf2[ 3*gamma_coo[1] + loc_ind%3 ] ) );
  }

  // FIXME: is this sync necessary ?
  __syncthreads();
}


__global__ void cuda_block_solve_update( cu_cmplx_PRECISION* phi, cu_cmplx_PRECISION* r, cu_cmplx_PRECISION* latest_iter, \
                                         schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                         int csw, int kernel_id, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                         int num_latt_site_var, block_struct* block ){

  int i, idx, DD_block_id, block_id, cublocks_per_DD_block, cu_block_ID, start;

  idx = threadIdx.x + blockDim.x * blockIdx.x;

  // not really a DD block id, but rather a linear counting of a grouping (per DD block) of CUDA threads
  DD_block_id = idx/nr_threads_per_DD_block;

  // offsetting idx to make it zero at the beginning of the threads living within a DD block
  idx = idx%nr_threads_per_DD_block;

  // this int will be the ACTUAL DD block ID, in the sense of accessing data from e.g. block_struct* block
  block_id = DD_blocks_to_compute[DD_block_id];

  cublocks_per_DD_block = nr_threads_per_DD_block/blockDim.x;

  // This serves as a substitute of blockIdx.x, to have a more
  // local and DD-block treatment more independent of the other DD blocks
  cu_block_ID = blockIdx.x%cublocks_per_DD_block;

  // this is the DD-block start of the spinors (phi, r, latest_iter and temporary ones)
  start = block[block_id].start * num_latt_site_var;

  cu_cmplx_PRECISION** tmp = s->oe_buf;
  cu_cmplx_PRECISION* tmp2 = tmp[2];
  cu_cmplx_PRECISION* tmp3 = tmp[3];

  phi += start;
  r += start;
  latest_iter += start;
  tmp2 += start;
  tmp3 += start;

  int nr_block_even_sites, nr_block_odd_sites;
  nr_block_even_sites = s->num_block_even_sites;
  nr_block_odd_sites = s->num_block_odd_sites;

  // update phi, latest_iter, r
  // even
  if(idx < 6*nr_block_even_sites){
    for(i=0; i<2; i++){
      ( latest_iter + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = ( tmp2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
      ( phi + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = make_cu_cmplx_PRECISION( cu_creal_PRECISION(( phi + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) +\
                                                                                                   cu_creal_PRECISION(( tmp2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]),
                                                                                                   cu_cimag_PRECISION(( phi + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                                                                                                   cu_cimag_PRECISION(( tmp2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) );
      ( r + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = ( tmp3 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
    }
  }
  // odd
  if(idx < 6*nr_block_odd_sites){
    for(i=0; i<2; i++){
      ( latest_iter + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = ( tmp2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
      ( phi + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = make_cu_cmplx_PRECISION( 
                                                                     cu_creal_PRECISION(( phi + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                                                                     cu_creal_PRECISION(( tmp2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]),
                                                                     cu_cimag_PRECISION(( phi + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                                                                     cu_cimag_PRECISION(( tmp2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) );
      ( r + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = make_cu_cmplx_PRECISION(0.0,0.0);
    }
  }

}



extern "C" void cuda_block_solve_oddeven_PRECISION( cuda_vector_PRECISION phi, cuda_vector_PRECISION r, cuda_vector_PRECISION latest_iter,
                                                    int start, int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s, level_struct *l,
                                                    struct Thread *threading, int stream_id, cudaStream_t *streams, int solve_at_cpu, int color,
                                                    int* DD_blocks_to_compute ) {

  if(solve_at_cpu){
    block_solve_oddeven_PRECISION( (vector_PRECISION)phi, (vector_PRECISION)r, (vector_PRECISION)latest_iter,
                                   start, s, l, threading );
  } else {

    int threads_per_cublock, nr_threads, size_D_oeclov, nr_threads_per_DD_block, dir;
    size_t tot_shared_mem;

    // we choose here a multiple of 96 due to being the smallest nr divisible by 32, but also divisible by 6
    threads_per_cublock = 96;

    //nr_DD_sites = s->num_block_sites;

    // the nr of threads needed is computed like this: max between num_block_even_sites and num_block_odd_sites, and then
    //                                                 for each lattice site (of those even-odd), we need 12/2 really independent
    //                                                 components, due to gamma5 symmetry. I.e. each thread is in charge of
    //                                                 one site component !
    nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites;
    nr_threads = nr_threads*(12/2);
    nr_threads = nr_threads*nr_DD_blocks_to_compute;

    // this is the size of the local matrix, i.e. per lattice site. 12^2=144, but ... (??)
    size_D_oeclov = (g.csw!=0) ? 72 : 12;

    // ingredients composing shared memory:
    //                                     1. for memory associated to spinors, we first multiply threads_per_cublock by 2, this is to
    //                                        account for gamma5 symmetry (because we're thinking this way: 6 CUDA threads correspond to
    //                                        a single lattice site), then, the factor of 4 comes from the sub-spinors we need to use within
    //                                        the kernel: phi_?, r_?, tmp_2_?, tmp_3_?, and finally the factor of 2 comes from the odd-even
    //                                        preconditioning taken here
    //                                     2. size_D_oeclov gives us the size of the local matrix per site, hence we need to multiply by
    //                                        threads_per_cublock/6 (which gives us the nr of sites per CUDA block), and then the factor
    //                                        of 2 comes from odd-even
    //
    // it's fundamental to think about the implementation here in the following way:
    //
    //                                     each CUDA block computes a certain nr of lattice sites, say X, but we're using odd-even preconditioning,
    //                                     therefore that same CUDA block is in charge not only of computing those X (say, even) sites, but also of
    //                                     computing the associated X (then, odd) sites through odd-even preconditioning
    //tot_shared_mem = 2*4*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 2*size_D_oeclov*(threads_per_cublock/6)*sizeof(cu_config_PRECISION);
    // UPDATE: the factor (2+3) means that we are asking for 2 even local buffers and 3 odd local buffers, all these within the kernel
    tot_shared_mem = (2+3)*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 2*size_D_oeclov*(threads_per_cublock/6)*sizeof(cu_config_PRECISION);

    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

    // diag_oo inv
    cuda_block_diag_oo_inv_PRECISION<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                    (phi, r, latest_iter, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block);



    // hopping term, even sites
    // TODO: add call to code already implemented

    //tot_shared_mem = 2*4*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 2*size_D_oeclov*(threads_per_cublock/6)*sizeof(cu_config_PRECISION);
    tot_shared_mem = 1*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION);
    for( dir=0; dir<4; dir++ ){
      cuda_block_n_hopping_term_PRECISION_plus<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                              (phi, r, latest_iter, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, dir);
      //cuda_block_n_hopping_term_PRECISION_minus<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
      //                                         (phi, r, latest_iter, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, dir);
      break;
    }

    //cuda_check_error( _HARD_CHECK );

    // local minres
    // TODO: add call to code already implemented

    // hpping term, odd sites
    // TODO: add call to code already implemented

    // update phi and latest_iter
    // TODO: add call to code already implemented

    // update r
    // TODO: add call to code already implemented

    cuda_block_solve_update<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>>
                           (phi, r, latest_iter, s->s_on_gpu, g.my_rank, g.csw, 0, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block);

    // TODO: eventually, remove this line
    //cuda_check_error( _HARD_CHECK );

  }
}


#endif
