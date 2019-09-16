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



// Pre-definitions of CUDA functions to be called from the CUDA kernels, force inlining on some device functions

__forceinline__ __device__ int get_index_in_shared(int *x_indxs, int x);

// 6 threads, naive

__forceinline__ __device__ void _cuda_block_n_hopping_term_PRECISION_plus_6threads_naive(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, int amount, schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf, int ext_dir, int* gamma_coo, cu_cmplx_PRECISION* gamma_val);
__forceinline__ __device__ void _cuda_block_n_hopping_term_PRECISION_minus_6threads_naive(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, int amount, schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf, int ext_dir);

__global__ void cuda_block_n_hopping_term_PRECISION_plus_6threads_naive( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, schwarz_PRECISION_struct_on_gpu *s, int thread_id, int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block, int ext_dir, int amount );
__global__ void cuda_block_n_hopping_term_PRECISION_minus_6threads_naive( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, schwarz_PRECISION_struct_on_gpu *s, int thread_id, int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block, int ext_dir, int amount );


__forceinline__ __device__ void _cuda_block_hopping_term_PRECISION_plus_6threads_naive(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, int amount, schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf, int ext_dir, int* gamma_coo, cu_cmplx_PRECISION* gamma_val);
__forceinline__ __device__ void _cuda_block_hopping_term_PRECISION_minus_6threads_naive(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, int amount, schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf, int ext_dir);

__global__ void cuda_block_hopping_term_PRECISION_plus_6threads_naive( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, schwarz_PRECISION_struct_on_gpu *s, int thread_id, int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block, int ext_dir, int amount );
__global__ void cuda_block_hopping_term_PRECISION_minus_6threads_naive( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, schwarz_PRECISION_struct_on_gpu *s, int thread_id, int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block, int ext_dir, int amount );


// 6 threads, optimized

__forceinline__ __device__ void _cuda_block_diag_oo_inv_PRECISION_6threads_opt(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, schwarz_PRECISION_struct_on_gpu *s, int idx, cu_config_PRECISION *clov_vect, int csw);
__forceinline__ __device__ void _cuda_block_diag_ee_PRECISION_6threads_opt(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, schwarz_PRECISION_struct_on_gpu *s, int idx, cu_config_PRECISION *clov_vect, int csw);

__global__ void cuda_block_diag_oo_inv_PRECISION_6threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, schwarz_PRECISION_struct_on_gpu *s, int thread_id, int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block );
__global__ void cuda_block_diag_ee_PRECISION_6threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, schwarz_PRECISION_struct_on_gpu *s, int thread_id, int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block );

__global__ void cuda_block_oe_vector_PRECISION_copy_6threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, schwarz_PRECISION_struct_on_gpu *s, int thread_id, int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block, int sites_to_copy );
__global__ void cuda_block_oe_vector_PRECISION_define_6threads_opt( cu_cmplx_PRECISION* spinor, schwarz_PRECISION_struct_on_gpu *s, int thread_id, int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block, int sites_to_define, cu_cmplx_PRECISION val_to_assign );
__global__ void cuda_block_oe_vector_PRECISION_plus_6threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in1, cu_cmplx_PRECISION* in2, schwarz_PRECISION_struct_on_gpu *s, int thread_id, int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block, int sites_to_add );
__global__ void cuda_block_oe_vector_PRECISION_saxpy_6threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in1, cu_cmplx_PRECISION* in2, cu_cmplx_PRECISION alpha, schwarz_PRECISION_struct_on_gpu *s, int thread_id, int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block, int sites_to_add );
__global__ void cuda_block_solve_update_6threads_opt( cu_cmplx_PRECISION* phi, cu_cmplx_PRECISION* r, cu_cmplx_PRECISION* latest_iter, schwarz_PRECISION_struct_on_gpu *s, int thread_id, int csw, int kernel_id, int nr_threads_per_DD_block, int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block );

__global__ void cuda_block_oe_vector_PRECISION_minus_6threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in1, cu_cmplx_PRECISION* in2, schwarz_PRECISION_struct_on_gpu *s, int thread_id, int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block, int sites_to_add );

// 2 threads, naive



// 2 threads, optimized

__forceinline__ __device__ void _cuda_block_diag_oo_inv_PRECISION_2threads_opt(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, schwarz_PRECISION_struct_on_gpu *s, int idx, cu_config_PRECISION *clov_vect, int csw);
__forceinline__ __device__ void _cuda_block_diag_ee_PRECISION_2threads_opt(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, schwarz_PRECISION_struct_on_gpu *s, int idx, cu_config_PRECISION *clov_vect, int csw);

__global__ void cuda_block_diag_oo_inv_PRECISION_2threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, schwarz_PRECISION_struct_on_gpu *s, int thread_id, int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block );
__global__ void cuda_block_diag_ee_PRECISION_2threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, schwarz_PRECISION_struct_on_gpu *s, int thread_id, int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block );



//------------------------------------------------------------------------------------------------------------------------------------------


__global__ void cuda_block_oe_vector_PRECISION_copy_6threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                                                  schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                  int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                                  int num_latt_site_var, block_struct* block, int sites_to_copy ){

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

  out += start;
  in += start;

  int nr_block_even_sites, nr_block_odd_sites;
  nr_block_even_sites = s->num_block_even_sites;
  nr_block_odd_sites = s->num_block_odd_sites;

  if( sites_to_copy==_EVEN_SITES ){
    // even
    if(idx < 6*nr_block_even_sites){
      for(i=0; i<2; i++){
        ( out + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = ( in + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
      }
    }
  }
  else if( sites_to_copy==_ODD_SITES ){
    // odd
    if(idx < 6*nr_block_odd_sites){
      for(i=0; i<2; i++){
        ( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = ( in + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
      }
    }
  }
  else if( sites_to_copy==_FULL_SYSTEM ){
    // even
    if(idx < 6*nr_block_even_sites){
      for(i=0; i<2; i++){
        ( out + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = ( in + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
      }
    }
    // odd
    if(idx < 6*nr_block_odd_sites){
      for(i=0; i<2; i++){
        ( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = ( in + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
      }
    }
  }

}



__global__ void cuda_block_oe_vector_PRECISION_define_6threads_opt( cu_cmplx_PRECISION* spinor,\
                                                                    schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                    int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                                    int num_latt_site_var, block_struct* block, int sites_to_define,
                                                                    cu_cmplx_PRECISION val_to_assign ){

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

  spinor += start;

  int nr_block_even_sites, nr_block_odd_sites;
  nr_block_even_sites = s->num_block_even_sites;
  nr_block_odd_sites = s->num_block_odd_sites;

  if( sites_to_define==_EVEN_SITES ){
    // even
    if(idx < 6*nr_block_even_sites){
      for(i=0; i<2; i++){
        ( spinor + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = val_to_assign;
      }
    }
  }
  else if( sites_to_define==_ODD_SITES ){
    // odd
    if(idx < 6*nr_block_odd_sites){
      for(i=0; i<2; i++){
        ( spinor + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = val_to_assign;
      }
    }
  }
  else if( sites_to_define==_FULL_SYSTEM ){
    // even
    if(idx < 6*nr_block_even_sites){
      for(i=0; i<2; i++){
        ( spinor + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = val_to_assign;
      }
    }
    // odd
    if(idx < 6*nr_block_odd_sites){
      for(i=0; i<2; i++){
        ( spinor + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = val_to_assign;
      }
    }
  }

}



__global__ void cuda_block_oe_vector_PRECISION_plus_6threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in1, cu_cmplx_PRECISION* in2, \
                                                                  schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                  int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                                  int num_latt_site_var, block_struct* block, int sites_to_add ){

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

  out += start;
  in1 += start;
  in2 += start;

  int nr_block_even_sites, nr_block_odd_sites;
  nr_block_even_sites = s->num_block_even_sites;
  nr_block_odd_sites = s->num_block_odd_sites;

  // TODO: change the following additions to use cu_cadd_PRECISION(...)

  if( sites_to_add==_EVEN_SITES ){
    // even
    if(idx < 6*nr_block_even_sites){
      for(i=0; i<2; i++){
        ( out + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = make_cu_cmplx_PRECISION( cu_creal_PRECISION(( in1 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) +\
                                                                                                      cu_creal_PRECISION(( in2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]),
                                                                                                      cu_cimag_PRECISION(( in1 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                                                                                                      cu_cimag_PRECISION(( in2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) );
      }
    }
  }
  else if( sites_to_add==_ODD_SITES ){
    // odd
    if(idx < 6*nr_block_odd_sites){
      for(i=0; i<2; i++){
        ( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = make_cu_cmplx_PRECISION( 
                                                                       cu_creal_PRECISION(( in1 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                                                                       cu_creal_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]),
                                                                       cu_cimag_PRECISION(( in1 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                                                                       cu_cimag_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) );
      }
    }
  }
  else if( sites_to_add==_FULL_SYSTEM ){
    // even
    if(idx < 6*nr_block_even_sites){
      for(i=0; i<2; i++){
        ( out + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = make_cu_cmplx_PRECISION( cu_creal_PRECISION(( in1 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) +\
                                                                                                      cu_creal_PRECISION(( in2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]),
                                                                                                      cu_cimag_PRECISION(( in1 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                                                                                                      cu_cimag_PRECISION(( in2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) );
      }
    }
    // odd
    if(idx < 6*nr_block_odd_sites){
      for(i=0; i<2; i++){
        ( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = make_cu_cmplx_PRECISION( 
                                                                       cu_creal_PRECISION(( in1 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                                                                       cu_creal_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]),
                                                                       cu_cimag_PRECISION(( in1 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                                                                       cu_cimag_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) );
      }
    }
  }

}



__global__ void cuda_block_oe_vector_PRECISION_minus_6threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in1, cu_cmplx_PRECISION* in2, \
                                                                   schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                   int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                                   int num_latt_site_var, block_struct* block, int sites_to_add ){

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

  out += start;
  in1 += start;
  in2 += start;

  int nr_block_even_sites, nr_block_odd_sites;
  nr_block_even_sites = s->num_block_even_sites;
  nr_block_odd_sites = s->num_block_odd_sites;

  // TODO: change the following additions to use cu_csub_PRECISION(...)

  if( sites_to_add==_EVEN_SITES ){
    // even
    if(idx < 6*nr_block_even_sites){
      for(i=0; i<2; i++){
        ( out + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = make_cu_cmplx_PRECISION( cu_creal_PRECISION(( in1 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) -\
                                                                                                      cu_creal_PRECISION(( in2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]),
                                                                                                      cu_cimag_PRECISION(( in1 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) - \
                                                                                                      cu_cimag_PRECISION(( in2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) );
      }
    }
  }
  else if( sites_to_add==_ODD_SITES ){
    // odd
    if(idx < 6*nr_block_odd_sites){
      for(i=0; i<2; i++){
        ( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = make_cu_cmplx_PRECISION( 
                                                                       cu_creal_PRECISION(( in1 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) - \
                                                                       cu_creal_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]),
                                                                       cu_cimag_PRECISION(( in1 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) - \
                                                                       cu_cimag_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) );
      }
    }
  }
  else if( sites_to_add==_FULL_SYSTEM ){
    // even
    if(idx < 6*nr_block_even_sites){
      for(i=0; i<2; i++){
        ( out + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = make_cu_cmplx_PRECISION( cu_creal_PRECISION(( in1 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) -\
                                                                                                      cu_creal_PRECISION(( in2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]),
                                                                                                      cu_cimag_PRECISION(( in1 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) - \
                                                                                                      cu_cimag_PRECISION(( in2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) );
      }
    }
    // odd
    if(idx < 6*nr_block_odd_sites){
      for(i=0; i<2; i++){
        ( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = make_cu_cmplx_PRECISION( 
                                                                       cu_creal_PRECISION(( in1 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) - \
                                                                       cu_creal_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]),
                                                                       cu_cimag_PRECISION(( in1 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) - \
                                                                       cu_cimag_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) );
      }
    }
  }

}



__global__ void cuda_local_xy_over_xx_PRECISION( cu_cmplx_PRECISION* vec1, cu_cmplx_PRECISION* vec2, \
                                                 schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                 int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                 int num_latt_site_var, block_struct* block, int sites_to_dot ){

  //int i, idx, DD_block_id, block_id, cublocks_per_DD_block, cu_block_ID, start;
  int i, idx, DD_block_id, block_id, start;

  idx = threadIdx.x + blockDim.x * blockIdx.x;

  // not really a DD block id, but rather a linear counting of a grouping (per DD block) of CUDA threads
  DD_block_id = idx/nr_threads_per_DD_block;

  // offsetting idx to make it zero at the beginning of the threads living within a DD block
  idx = idx%nr_threads_per_DD_block;

  // this int will be the ACTUAL DD block ID, in the sense of accessing data from e.g. block_struct* block
  block_id = DD_blocks_to_compute[DD_block_id];

  // in the case of computing the dot product, it'll be only 1
  // NOTE: cublocks_per_DD_block = nr_threads_per_DD_block/blockDim.x;

  // This serves as a substitute of blockIdx.x, to have a more
  // local and DD-block treatment more independent of the other DD blocks
  // NOTE (from NOTE): there is a 1-to-1 correspondence between CUDA blocks and DD blocks, for
  //                   this computation of the dot product
  //cu_block_ID = blockIdx.x%cublocks_per_DD_block;
  //cu_block_ID = blockIdx.x;

  // this is the DD-block start of the spinors (phi, r, latest_iter and temporary ones)
  start = block[block_id].start * num_latt_site_var;

  vec1 += start;
  vec2 += start;

  int nr_block_even_sites, nr_block_odd_sites;
  nr_block_even_sites = s->num_block_even_sites;
  nr_block_odd_sites = s->num_block_odd_sites;

  // length of the vectors to <dot>
  int N;
  if( sites_to_dot==_EVEN_SITES ){
    N = nr_block_even_sites;
  }
  else if( sites_to_dot==_ODD_SITES ){
    N = nr_block_odd_sites;
  }
  else if( sites_to_dot==_FULL_SYSTEM ){
    N = nr_block_even_sites + nr_block_odd_sites;
  }
  N = N*12;

  // IMPORTANT: for this kernel, from here-on it was taken from:
  //            https://github.com/jiekebo/CUDA-By-Example/blob/master/5-dotproduct.cu

  // buffer in shared memory to store the partial sums of the dot product
  extern __shared__ cu_cmplx_PRECISION cache[];
  cu_cmplx_PRECISION *cache1 = (cu_cmplx_PRECISION*)cache;
  cu_cmplx_PRECISION *cache2 = cache1 + blockDim.x;

  //int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  int cache_index = idx;
	
  //float temp = 0;
  cu_cmplx_PRECISION temp1 = make_cu_cmplx_PRECISION(0.0,0.0);
  cu_cmplx_PRECISION temp2 = make_cu_cmplx_PRECISION(0.0,0.0);

  while (tid < N){
    //temp += vec1[tid] * vec2[tid];
    temp1 = cu_cadd_PRECISION( temp1, cu_cmul_PRECISION( cu_conj_PRECISION(vec1[tid]), vec2[tid] ) );
    temp2 = cu_cadd_PRECISION( temp2, cu_cmul_PRECISION( cu_conj_PRECISION(vec1[tid]), vec1[tid] ) );
    //tid += blockDim.x * gridDim.x;
    tid += blockDim.x;
  }

  // set the cache values
  cache1[cache_index] = temp1;
  cache2[cache_index] = temp2;

  // synchronize threads in this block
  __syncthreads();

  // for reductions, threadsPerBlock must be a power of 2
  // because of the following code
  i = blockDim.x/2;
  while (i != 0){
    if (cache_index < i){
      //cache[cache_index] += cache[cache_index + i];
      cache1[cache_index] = cu_cadd_PRECISION( cache1[cache_index], cache1[cache_index + i] );
      cache2[cache_index] = cu_cadd_PRECISION( cache2[cache_index], cache2[cache_index + i] );
    }
    __syncthreads();
    i /= 2;
  }

  cu_cmplx_PRECISION alpha = cu_cdiv_PRECISION( cache1[0], cache2[0] );

  if (cache_index == 0){
    //c[blockIdx.x] = cache[0];
    //printf("block_id=%d, idx=%d, tid=%d, alpha = %f+%f\n", block_id, idx, tid, cu_creal_PRECISION(alpha), cu_cimag_PRECISION(alpha));
  }

}

































__global__ void cuda_block_oe_vector_PRECISION_saxpy_6threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in1, cu_cmplx_PRECISION* in2, cu_cmplx_PRECISION alpha, \
                                                                   schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                   int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                                   int num_latt_site_var, block_struct* block, int sites_to_add ){

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

  out += start;
  in1 += start;
  in2 += start;

  int nr_block_even_sites, nr_block_odd_sites;
  nr_block_even_sites = s->num_block_even_sites;
  nr_block_odd_sites = s->num_block_odd_sites;

  PRECISION buf_real, buf_imag;

  if( sites_to_add==_EVEN_SITES ){
    // even
    if(idx < 6*nr_block_even_sites){
      for(i=0; i<2; i++){

        buf_real = cu_creal_PRECISION(( in1 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                   cu_creal_PRECISION(alpha) * cu_creal_PRECISION(( in2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) - \
                   cu_cimag_PRECISION(alpha) * cu_cimag_PRECISION(( in2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]);
        buf_imag = cu_cimag_PRECISION(( in1 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                   cu_creal_PRECISION(alpha) * cu_cimag_PRECISION(( in2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                   cu_cimag_PRECISION(alpha) * cu_creal_PRECISION(( in2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]);

        ( out + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = make_cu_cmplx_PRECISION( buf_real, buf_imag );

      }
    }
  }
  else if( sites_to_add==_ODD_SITES ){
    // odd
    if(idx < 6*nr_block_odd_sites){
      for(i=0; i<2; i++){

        buf_real = cu_creal_PRECISION(( in1 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                   cu_creal_PRECISION(alpha) * cu_creal_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) - \
                   cu_cimag_PRECISION(alpha) * cu_cimag_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]);
        buf_imag = cu_cimag_PRECISION(( in1 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                   cu_creal_PRECISION(alpha) * cu_cimag_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                   cu_cimag_PRECISION(alpha) * cu_creal_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]);

        ( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = make_cu_cmplx_PRECISION( buf_real, buf_imag );

      }
    }
  }
  else if( sites_to_add==_FULL_SYSTEM ){
    // even
    if(idx < 6*nr_block_even_sites){
      for(i=0; i<2; i++){

        buf_real = cu_creal_PRECISION(( in1 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                   cu_creal_PRECISION(alpha) * cu_creal_PRECISION(( in2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) - \
                   cu_cimag_PRECISION(alpha) * cu_cimag_PRECISION(( in2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]);
        buf_imag = cu_cimag_PRECISION(( in1 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                   cu_creal_PRECISION(alpha) * cu_cimag_PRECISION(( in2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                   cu_cimag_PRECISION(alpha) * cu_creal_PRECISION(( in2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]);

        ( out + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = make_cu_cmplx_PRECISION( buf_real, buf_imag );

      }
    }
    // odd
    if(idx < 6*nr_block_odd_sites){
      for(i=0; i<2; i++){

        buf_real = cu_creal_PRECISION(( in1 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                   cu_creal_PRECISION(alpha) * cu_creal_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) - \
                   cu_cimag_PRECISION(alpha) * cu_cimag_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]);
        buf_imag = cu_cimag_PRECISION(( in1 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                   cu_creal_PRECISION(alpha) * cu_cimag_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                   cu_cimag_PRECISION(alpha) * cu_creal_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]);

        ( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = make_cu_cmplx_PRECISION( buf_real, buf_imag );

      }
    }
  }

}



__global__ void cuda_block_diag_oo_inv_PRECISION_6threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                                               schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                               int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                               int num_latt_site_var, block_struct* block ){

  int i, idx, DD_block_id, block_id, cublocks_per_DD_block, cu_block_ID, size_D_oeclov, start, nr_D_dumps;

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
  //size_D_oeclov = (csw!=0) ? 72 : 12;
  size_D_oeclov = (csw!=0) ? 42 : 12;

  // this is the DD-block start of the spinors (phi, r, latest_iter and temporary ones)
  start = block[block_id].start * num_latt_site_var;

  out += start;
  in += start;

  // this operator is stored in column form!
  //cu_config_PRECISION *op_oe_vect = s->op.oe_clover_vectorized;
  cu_config_PRECISION *op_oe_vect = s->op.oe_clover_gpustorg;
  // FIXME: instead of 12, use num_latt_site_var
  op_oe_vect += (start/12)*size_D_oeclov;

  extern __shared__ cu_cmplx_PRECISION shared_data[];

  //TODO: can we trim the use of shared memory ... ?

  cu_cmplx_PRECISION *shared_data_loc = shared_data;
  cu_cmplx_PRECISION *in_o, *out_o;
  cu_config_PRECISION *clov_vect_b_o;

  in_o = shared_data_loc + 0*(2*blockDim.x);
  out_o = shared_data_loc + 1*(2*blockDim.x);

  clov_vect_b_o = (cu_cmplx_PRECISION*)((cu_cmplx_PRECISION*)shared_data_loc + 2*(2*blockDim.x));

  // TODO: change this partial summary to the new shared memory required
  // partial summary so far:
  //    ** phi_b_e has a memory reservation of size 2*6*(blockDim.x/6)
  //    ** same for r_b_e, tmp_2_e, tmp_3_e
  //    ** clov_vect_b_e has a memory reservation of size size_D_oeclov*(blockDim.x/6)
  //    ** equivalently, we can say the same about the *_o variables

  // odd
  if(idx < 6*nr_block_odd_sites){
    for(i=0; i<2; i++){
      in_o[blockDim.x*i + threadIdx.x] = ( in + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
    }

    // the factor of 12 comes from (the factor of 16 comes from the nr of lattice sites per CUDA block: 96/6 = 16): 72*16=1152 ----> 1152/96=12. This implies 12 dumps of data into shared_memory
    nr_D_dumps = (blockDim.x/6)*size_D_oeclov/blockDim.x; // = 7 always
    //nr_D_dumps /= nr_D_dumps;
    for(i=0; i<nr_D_dumps; i++){
      clov_vect_b_o[blockDim.x*i + threadIdx.x] = ( op_oe_vect + 42*nr_block_even_sites + cu_block_ID*blockDim.x*nr_D_dumps + blockDim.x*i + threadIdx.x )[0];
    }
  }

  __syncthreads();

  // FUNCTION: chi = D_{oo}^{-1} * eta_{0}
  if(idx < 6*nr_block_odd_sites){
    _cuda_block_diag_oo_inv_PRECISION_6threads_opt(out_o, in_o, start, s, idx, clov_vect_b_o, csw);
  }

  __syncthreads();

  // update tmp2 and tmp3
  // odd
  if(idx < 6*nr_block_odd_sites){
    for(i=0; i<2; i++){
      ( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = out_o[blockDim.x*i + threadIdx.x];
    }
  }

}



__forceinline__ __device__ void _cuda_block_diag_oo_inv_PRECISION_6threads_opt(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, \
                                                                               schwarz_PRECISION_struct_on_gpu *s, int idx, cu_config_PRECISION *op_clov_vect, int csw){

  // FIXME: extend code to include case csw==0

  int local_idx = idx%6;
  // this offset is per site and within each CUDA block
  int site_offset = (threadIdx.x/6)*12;

  cu_cmplx_PRECISION* eta_site = eta + site_offset;
  cu_cmplx_PRECISION* phi_site = phi + site_offset;
  cu_config_PRECISION* op_clov_vect_site = op_clov_vect + (site_offset/12)*42;

  eta_site[ local_idx ] = make_cu_cmplx_PRECISION(0.0,0.0);
  eta_site[ 6 + local_idx ] = make_cu_cmplx_PRECISION(0.0,0.0);

  int matrx_indx;

  if( csw!=0 ){
    // Run over all the sites within this block, and perform: eta = op_clov_vect * phi
    // There are 16 sites per CUDA block, in the case of choosing blockDimx.x equal to 96

    // first compute upper half of vector for each site, then lower half
    for( int i=0; i<2; i++ ){
      // outter loop for matrix*vector double loop unrolled
      for( int j=0; j<6; j++ ){

        // The following index is a mapping from the usual full matrix to the reduced form (i.e. the compressed due to being Hermitian)
        // Usually, in the full form: i*21 + j*6 + local_idx
        if( local_idx>j ){
          matrx_indx = 21 - (5-j+1)*((5-j+1)+1)/2 + (local_idx-j);
        }
        else if( local_idx==j ){
          matrx_indx = 21 - (5-j+1)*((5-j+1)+1)/2;
        }
        else{
          matrx_indx = 21 - (5-local_idx+1)*((5-local_idx+1)+1)/2 + (j-local_idx);
        }

        if( local_idx>j || local_idx==j ){
          eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ], cu_cmul_PRECISION( (op_clov_vect_site + i*21)[matrx_indx], phi_site[j + 6*i] ) );
        }
        else{
          eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ], cu_cmul_PRECISION( cu_conj_PRECISION((op_clov_vect_site + i*21)[matrx_indx]), phi_site[j + 6*i] ) );
        }
      }
    }
  }
}



__global__ void cuda_block_diag_oo_inv_PRECISION_2threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                                               schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                               int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                               int num_latt_site_var, block_struct* block ){

  int i, idx, DD_block_id, block_id, cublocks_per_DD_block, cu_block_ID, size_D_oeclov, start, nr_D_dumps;

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
  //size_D_oeclov = (csw!=0) ? 72 : 12;
  size_D_oeclov = (csw!=0) ? 42 : 12;

  // this is the DD-block start of the spinors (phi, r, latest_iter and temporary ones)
  start = block[block_id].start * num_latt_site_var;

  out += start;
  in += start;

  // this operator is stored in column form!
  //cu_config_PRECISION *op_oe_vect = s->op.oe_clover_vectorized;
  cu_config_PRECISION *op_oe_vect = s->op.oe_clover_gpustorg;
  // FIXME: instead of 12, use num_latt_site_var
  op_oe_vect += (start/12)*size_D_oeclov;

  extern __shared__ cu_cmplx_PRECISION shared_data[];

  //TODO: can we trim the use of shared memory ... ?

  cu_cmplx_PRECISION *shared_data_loc = shared_data;
  cu_cmplx_PRECISION *in_o, *out_o;
  cu_config_PRECISION *clov_vect_b_o;

  in_o = shared_data_loc + 0*(6*blockDim.x);
  out_o = shared_data_loc + 1*(6*blockDim.x);

  clov_vect_b_o = (cu_cmplx_PRECISION*)((cu_cmplx_PRECISION*)shared_data_loc + 2*(6*blockDim.x));

  // TODO: change this partial summary to the new shared memory required
  // partial summary so far:
  //    ** phi_b_e has a memory reservation of size 2*6*(blockDim.x/6)
  //    ** same for r_b_e, tmp_2_e, tmp_3_e
  //    ** clov_vect_b_e has a memory reservation of size size_D_oeclov*(blockDim.x/6)
  //    ** equivalently, we can say the same about the *_o variables

  // odd
  if(idx < 2*nr_block_odd_sites){
    for(i=0; i<6; i++){
      in_o[blockDim.x*i + threadIdx.x] = ( in + 12*nr_block_even_sites + cu_block_ID*blockDim.x*6 + blockDim.x*i + threadIdx.x )[0];
    }

    // the factor of 12 comes from (the factor of 16 comes from the nr of lattice sites per CUDA block: 96/6 = 16): 72*16=1152 ----> 1152/96=12. This implies 12 dumps of data into shared_memory
    nr_D_dumps = (blockDim.x/2)*size_D_oeclov/blockDim.x; // = 21 always
    //nr_D_dumps /= nr_D_dumps;
    for(i=0; i<nr_D_dumps; i++){
      clov_vect_b_o[blockDim.x*i + threadIdx.x] = ( op_oe_vect + 42*nr_block_even_sites + cu_block_ID*blockDim.x*nr_D_dumps + blockDim.x*i + threadIdx.x )[0];
    }
  }

  //__syncthreads();

  // FUNCTION: chi = D_{oo}^{-1} * eta_{0}
  if(idx < 2*nr_block_odd_sites){
    _cuda_block_diag_oo_inv_PRECISION_2threads_opt(out_o, in_o, start, s, idx, clov_vect_b_o, csw);
  }

  //__syncthreads();

  // update tmp2 and tmp3
  // odd
  if(idx < 2*nr_block_odd_sites){
    for(i=0; i<6; i++){
      ( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x*6 + blockDim.x*i + threadIdx.x )[0] = out_o[blockDim.x*i + threadIdx.x];
    }
  }

}



__forceinline__ __device__ void _cuda_block_diag_oo_inv_PRECISION_2threads_opt(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, \
                                                                               schwarz_PRECISION_struct_on_gpu *s, int idx, cu_config_PRECISION *op_clov_vect, int csw){

  // FIXME: extend code to include case csw==0

  //int local_idx = idx%2;
  int local_idx;
  // this offset is per site and within each CUDA block
  int site_offset = (threadIdx.x/2)*12;

  cu_cmplx_PRECISION* eta_site = eta + site_offset;
  cu_cmplx_PRECISION* phi_site = phi + site_offset;
  cu_config_PRECISION* op_clov_vect_site = op_clov_vect + (site_offset/12)*42;

  int k;

  for( k=0; k<3; k++ ){

    local_idx = (idx%2)*3 + k;

    eta_site[ local_idx ] = make_cu_cmplx_PRECISION(0.0,0.0);
    eta_site[ 6 + local_idx ] = make_cu_cmplx_PRECISION(0.0,0.0);

    int matrx_indx;

    if( csw!=0 ){
      // Run over all the sites within this block, and perform: eta = op_clov_vect * phi
      // There are 16 sites per CUDA block, in the case of choosing blockDimx.x equal to 96

      // first compute upper half of vector for each site, then lower half
      for( int i=0; i<2; i++ ){
        // outter loop for matrix*vector double loop unrolled
        for( int j=0; j<6; j++ ){

          // The following index is a mapping from the usual full matrix to the reduced form (i.e. the compressed due to being Hermitian)
          // Usually, in the full form: i*21 + j*6 + local_idx
          if( local_idx>j ){
            matrx_indx = 21 - (5-j+1)*((5-j+1)+1)/2 + (local_idx-j);
          }
          else if( local_idx==j ){
            matrx_indx = 21 - (5-j+1)*((5-j+1)+1)/2;
          }
          else{
            matrx_indx = 21 - (5-local_idx+1)*((5-local_idx+1)+1)/2 + (j-local_idx);
          }

          if( local_idx>j || local_idx==j ){
            eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ], cu_cmul_PRECISION( (op_clov_vect_site + i*21)[matrx_indx], phi_site[j + 6*i] ) );
          }
          else{
            eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ], cu_cmul_PRECISION( cu_conj_PRECISION((op_clov_vect_site + i*21)[matrx_indx]), phi_site[j + 6*i] ) );
          }
        }
      }
    }
  }
}



__global__ void cuda_block_diag_ee_PRECISION_6threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                                           schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                           int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                           int num_latt_site_var, block_struct* block ){

  int i, idx, DD_block_id, block_id, cublocks_per_DD_block, cu_block_ID, size_D_oeclov, start, nr_D_dumps;

  //int nr_block_even_sites, nr_block_odd_sites;
  int nr_block_odd_sites;
  //nr_block_even_sites = s->num_block_even_sites;
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
  //size_D_oeclov = (csw!=0) ? 72 : 12;
  size_D_oeclov = (csw!=0) ? 42 : 12;

  // this is the DD-block start of the spinors (phi, r, latest_iter and temporary ones)
  start = block[block_id].start * num_latt_site_var;

  out += start;
  in += start;

  // this operator is stored in column form!
  //cu_config_PRECISION *op_oe_vect = s->op.oe_clover_vectorized;
  cu_config_PRECISION *op_oe_vect = s->op.oe_clover_gpustorg;
  // FIXME: instead of 12, use num_latt_site_var
  op_oe_vect += (start/12)*size_D_oeclov;

  extern __shared__ cu_cmplx_PRECISION shared_data[];

  //TODO: can we trim the use of shared memory ... ?

  cu_cmplx_PRECISION *shared_data_loc = shared_data;
  cu_cmplx_PRECISION *in_o, *out_o;
  cu_config_PRECISION *clov_vect_b_o;

  in_o = shared_data_loc + 0*(2*blockDim.x);
  out_o = shared_data_loc + 1*(2*blockDim.x);

  clov_vect_b_o = (cu_cmplx_PRECISION*)((cu_cmplx_PRECISION*)shared_data_loc + 2*(2*blockDim.x));

  // TODO: change this partial summary to the new shared memory required
  // partial summary so far:
  //    ** phi_b_e has a memory reservation of size 2*6*(blockDim.x/6)
  //    ** same for r_b_e, tmp_2_e, tmp_3_e
  //    ** clov_vect_b_e has a memory reservation of size size_D_oeclov*(blockDim.x/6)
  //    ** equivalently, we can say the same about the *_o variables

  // odd
  if(idx < 6*nr_block_odd_sites){
    for(i=0; i<2; i++){
      //in_o[blockDim.x*i + threadIdx.x] = ( in + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
      in_o[blockDim.x*i + threadIdx.x] = ( in + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
    }

    // the factor of 12 comes from (the factor of 16 comes from the nr of lattice sites per CUDA block: 96/6 = 16): 72*16=1152 ----> 1152/96=12. This implies 12 dumps of data into shared_memory
    nr_D_dumps = (blockDim.x/6)*size_D_oeclov/blockDim.x; // = 7 always
    //nr_D_dumps /= nr_D_dumps;
    for(i=0; i<nr_D_dumps; i++){
      //clov_vect_b_o[blockDim.x*i + threadIdx.x] = ( op_oe_vect + 42*nr_block_even_sites + cu_block_ID*blockDim.x*nr_D_dumps + blockDim.x*i + threadIdx.x )[0];
      clov_vect_b_o[blockDim.x*i + threadIdx.x] = ( op_oe_vect + cu_block_ID*blockDim.x*nr_D_dumps + blockDim.x*i + threadIdx.x )[0];
    }
  }

  __syncthreads();

  // FUNCTION: chi = D_{oo}^{-1} * eta_{0}
  if(idx < 6*nr_block_odd_sites){
    _cuda_block_diag_ee_PRECISION_6threads_opt(out_o, in_o, start, s, idx, clov_vect_b_o, csw);
  }

  __syncthreads();

  // update tmp2 and tmp3
  // odd
  if(idx < 6*nr_block_odd_sites){
    for(i=0; i<2; i++){
      //( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = out_o[blockDim.x*i + threadIdx.x];
      ( out + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = out_o[blockDim.x*i + threadIdx.x];
    }
  }

}



__forceinline__ __device__ void _cuda_block_diag_ee_PRECISION_6threads_opt(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, \
                                                                           schwarz_PRECISION_struct_on_gpu *s, int idx, cu_config_PRECISION *op_clov_vect, int csw){

  // FIXME: extend code to include case csw==0

  int local_idx = idx%6;
  // this offset is per site and within each CUDA block
  int site_offset = (threadIdx.x/6)*12;

  cu_cmplx_PRECISION* eta_site = eta + site_offset;
  cu_cmplx_PRECISION* phi_site = phi + site_offset;
  cu_config_PRECISION* op_clov_vect_site = op_clov_vect + (site_offset/12)*42;

  eta_site[ local_idx ] = make_cu_cmplx_PRECISION(0.0,0.0);
  eta_site[ 6 + local_idx ] = make_cu_cmplx_PRECISION(0.0,0.0);

  int matrx_indx;

  if( csw!=0 ){
    // Run over all the sites within this block, and perform: eta = op_clov_vect * phi
    // There are 16 sites per CUDA block, in the case of choosing blockDimx.x equal to 96

    // first compute upper half of vector for each site, then lower half
    for( int i=0; i<2; i++ ){
      // outter loop for matrix*vector double loop unrolled
      for( int j=0; j<6; j++ ){

        // The following index is a mapping from the usual full matrix to the reduced form (i.e. the compressed due to being Hermitian)
        // Usually, in the full form: i*21 + j*6 + local_idx
        if( local_idx>j ){
          matrx_indx = 21 - (5-j+1)*((5-j+1)+1)/2 + (local_idx-j);
        }
        else if( local_idx==j ){
          matrx_indx = 21 - (5-j+1)*((5-j+1)+1)/2;
        }
        else{
          matrx_indx = 21 - (5-local_idx+1)*((5-local_idx+1)+1)/2 + (j-local_idx);
        }

        if( local_idx>j || local_idx==j ){
          eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ], cu_cmul_PRECISION( (op_clov_vect_site + i*21)[matrx_indx], phi_site[j + 6*i] ) );
        }
        else{
          eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ], cu_cmul_PRECISION( cu_conj_PRECISION((op_clov_vect_site + i*21)[matrx_indx]), phi_site[j + 6*i] ) );
        }
      }
    }
  }
}



__global__ void cuda_block_diag_ee_PRECISION_2threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                                           schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                           int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                           int num_latt_site_var, block_struct* block ){

  int i, idx, DD_block_id, block_id, cublocks_per_DD_block, cu_block_ID, size_D_oeclov, start, nr_D_dumps;

  //int nr_block_even_sites, nr_block_odd_sites;
  int nr_block_odd_sites;
  //nr_block_even_sites = s->num_block_even_sites;
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
  //size_D_oeclov = (csw!=0) ? 72 : 12;
  size_D_oeclov = (csw!=0) ? 42 : 12;

  // this is the DD-block start of the spinors (phi, r, latest_iter and temporary ones)
  start = block[block_id].start * num_latt_site_var;

  out += start;
  in += start;

  // this operator is stored in column form!
  //cu_config_PRECISION *op_oe_vect = s->op.oe_clover_vectorized;
  cu_config_PRECISION *op_oe_vect = s->op.oe_clover_gpustorg;
  // FIXME: instead of 12, use num_latt_site_var
  op_oe_vect += (start/12)*size_D_oeclov;

  extern __shared__ cu_cmplx_PRECISION shared_data[];

  //TODO: can we trim the use of shared memory ... ?

  cu_cmplx_PRECISION *shared_data_loc = shared_data;
  cu_cmplx_PRECISION *in_o, *out_o;
  cu_config_PRECISION *clov_vect_b_o;

  in_o = shared_data_loc + 0*(6*blockDim.x);
  out_o = shared_data_loc + 1*(6*blockDim.x);

  clov_vect_b_o = (cu_cmplx_PRECISION*)((cu_cmplx_PRECISION*)shared_data_loc + 2*(6*blockDim.x));

  // TODO: change this partial summary to the new shared memory required
  // partial summary so far:
  //    ** phi_b_e has a memory reservation of size 2*6*(blockDim.x/6)
  //    ** same for r_b_e, tmp_2_e, tmp_3_e
  //    ** clov_vect_b_e has a memory reservation of size size_D_oeclov*(blockDim.x/6)
  //    ** equivalently, we can say the same about the *_o variables

  // odd
  if(idx < 2*nr_block_odd_sites){
    for(i=0; i<6; i++){
      //in_o[blockDim.x*i + threadIdx.x] = ( in + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
      in_o[blockDim.x*i + threadIdx.x] = ( in + cu_block_ID*blockDim.x*6 + blockDim.x*i + threadIdx.x )[0];
    }

    // the factor of 12 comes from (the factor of 16 comes from the nr of lattice sites per CUDA block: 96/6 = 16): 72*16=1152 ----> 1152/96=12. This implies 12 dumps of data into shared_memory
    nr_D_dumps = (blockDim.x/2)*size_D_oeclov/blockDim.x; // = 7 always
    //nr_D_dumps /= nr_D_dumps;
    for(i=0; i<nr_D_dumps; i++){
      //clov_vect_b_o[blockDim.x*i + threadIdx.x] = ( op_oe_vect + 42*nr_block_even_sites + cu_block_ID*blockDim.x*nr_D_dumps + blockDim.x*i + threadIdx.x )[0];
      clov_vect_b_o[blockDim.x*i + threadIdx.x] = ( op_oe_vect + cu_block_ID*blockDim.x*nr_D_dumps + blockDim.x*i + threadIdx.x )[0];
    }
  }

  //__syncthreads();

  // FUNCTION: chi = D_{oo}^{-1} * eta_{0}
  if(idx < 2*nr_block_odd_sites){
    _cuda_block_diag_ee_PRECISION_2threads_opt(out_o, in_o, start, s, idx, clov_vect_b_o, csw);
  }

  //__syncthreads();

  // update tmp2 and tmp3
  // odd
  if(idx < 2*nr_block_odd_sites){
    for(i=0; i<6; i++){
      //( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = out_o[blockDim.x*i + threadIdx.x];
      ( out + cu_block_ID*blockDim.x*6 + blockDim.x*i + threadIdx.x )[0] = out_o[blockDim.x*i + threadIdx.x];
    }
  }

}



__forceinline__ __device__ void _cuda_block_diag_ee_PRECISION_2threads_opt(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, \
                                                                           schwarz_PRECISION_struct_on_gpu *s, int idx, cu_config_PRECISION *op_clov_vect, int csw){

  // FIXME: extend code to include case csw==0

  //int local_idx = idx%6;
  int local_idx;
  // this offset is per site and within each CUDA block
  int site_offset = (threadIdx.x/2)*12;

  cu_cmplx_PRECISION* eta_site = eta + site_offset;
  cu_cmplx_PRECISION* phi_site = phi + site_offset;
  cu_config_PRECISION* op_clov_vect_site = op_clov_vect + (site_offset/12)*42;

  int k;

  for( k=0; k<3; k++ ){

    local_idx = (idx%2)*3 + k;

    eta_site[ local_idx ] = make_cu_cmplx_PRECISION(0.0,0.0);
    eta_site[ 6 + local_idx ] = make_cu_cmplx_PRECISION(0.0,0.0);

    int matrx_indx;

    if( csw!=0 ){
      // Run over all the sites within this block, and perform: eta = op_clov_vect * phi
      // There are 16 sites per CUDA block, in the case of choosing blockDimx.x equal to 96

      // first compute upper half of vector for each site, then lower half
      for( int i=0; i<2; i++ ){
        // outter loop for matrix*vector double loop unrolled
        for( int j=0; j<6; j++ ){

          // The following index is a mapping from the usual full matrix to the reduced form (i.e. the compressed due to being Hermitian)
          // Usually, in the full form: i*21 + j*6 + local_idx
          if( local_idx>j ){
            matrx_indx = 21 - (5-j+1)*((5-j+1)+1)/2 + (local_idx-j);
          }
          else if( local_idx==j ){
            matrx_indx = 21 - (5-j+1)*((5-j+1)+1)/2;
          }
          else{
            matrx_indx = 21 - (5-local_idx+1)*((5-local_idx+1)+1)/2 + (j-local_idx);
          }

          if( local_idx>j || local_idx==j ){
            eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ], cu_cmul_PRECISION( (op_clov_vect_site + i*21)[matrx_indx], phi_site[j + 6*i] ) );
          }
          else{
            eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ], cu_cmul_PRECISION( cu_conj_PRECISION((op_clov_vect_site + i*21)[matrx_indx]), phi_site[j + 6*i] ) );
          }
        }
      }
    }
  }
}



__forceinline__ __device__ int get_index_in_shared(int *x_indxs, int x){

  int length_of_array = blockDim.x/6;
  int lookup_indx;
  int i;

  for( i=0; i<length_of_array; i++ ){
    if( x_indxs[i]==x ){
      lookup_indx = i;
    }
  }

  return lookup_indx;

}



// gamma_val and gamma_coo are both loaded into shared memory for this kernel
__global__ void cuda_block_n_hopping_term_PRECISION_plus_6threads_naive( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                                                         schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                         int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                                         int num_latt_site_var, block_struct* block, int ext_dir, int amount ){

  int idx;
  idx = threadIdx.x + blockDim.x * blockIdx.x;

  int* gamma_coo;

  cu_cmplx_PRECISION* gamma_val;

  int DD_block_id, block_id, start;

  //int nr_block_even_sites;
  //nr_block_even_sites = s->num_block_even_sites;

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

  extern __shared__ cu_cmplx_PRECISION shared_data_bare[];
  cu_cmplx_PRECISION* shared_data = shared_data_bare;

  //int loc_ind=idx%6;

  //int *j_indxs = (int*)shared_data;
  //shared_data = (cu_cmplx_PRECISION*)((int*)shared_data + blockDim.x/6);

  //int *k_indxs = (int*)shared_data;
  //shared_data = (cu_cmplx_PRECISION*)((int*)shared_data + blockDim.x/6);

  gamma_coo = (int*)shared_data;
  shared_data = (cu_cmplx_PRECISION*)((int*)shared_data + 16);

  gamma_val = shared_data;
  shared_data = shared_data + 16;

  cu_cmplx_PRECISION *tmp_loc;
  tmp_loc = shared_data;
  shared_data = shared_data + 2*blockDim.x;

  //cu_cmplx_PRECISION *in_loc;
  //in_loc = shared_data;
  //shared_data = shared_data + 2*blockDim.x;

  //cu_cmplx_PRECISION *out_loc;
  //out_loc = shared_data;
  //shared_data = shared_data + 2*blockDim.x;

  // loading gamma coordinates into shared memory
  if( threadIdx.x<16 ){
    gamma_coo[threadIdx.x] = s->gamma_info_coo[threadIdx.x];
  }

  // loading gamma values into shared memory
  if( threadIdx.x<16 ){
    gamma_val[threadIdx.x] = s->gamma_info_vals[threadIdx.x];
  }

  // initializing to zero a local buffer for temporary computations
  //if(idx < 6*nr_block_even_sites){
  if(idx < 6*( (amount==_EVEN_SITES)?(s->dir_length_even[ext_dir]):(s->dir_length_odd[ext_dir]) )){
    tmp_loc[threadIdx.x] = make_cu_cmplx_PRECISION(0.0,0.0);
    tmp_loc[threadIdx.x + blockDim.x] = make_cu_cmplx_PRECISION(0.0,0.0);
  }

  // loading the 'in' and 'out' spinors into shared memory
  /*

  int a1, i, k, j;

  if( amount==_EVEN_SITES ){
    a1=0; //n1=s->dir_length_even[ext_dir];
  }
  else if( amount==_ODD_SITES ){
    a1=s->dir_length_even[ext_dir]; //n1=a1+s->dir_length_odd[ext_dir];
  }
  else{
    //TODO
  }

  i = idx/6 + a1;
  k = s->oe_index[ext_dir][i];
  j = s->op.neighbor_table[4*k+ext_dir];

  //if( threadIdx.x==0 ){
  //  printf( "k=%d, j=%d\n", k, j );
  //}

  if( loc_ind==0 ){
    j_indxs[ threadIdx.x/6 ] = j; // phi ---> in
    k_indxs[ threadIdx.x/6 ] = k; // eta ---> out
  }

  __syncthreads();

  int half_warp_loc_indx, half_warp_indx;

  int copy_unwnds = blockDim.x/6 / (blockDim.x/16);
  int copy_unwnds_rest = blockDim.x/6 - copy_unwnds*(blockDim.x/16);

  half_warp_loc_indx = threadIdx.x%16;
  half_warp_indx = threadIdx.x/16;

  // in, copy unwnds
  //for( i=0; i<copy_unwnds; i++ ){
  //  // each half-warp copies a lattice site
  //  if( half_warp_loc_indx<12 ){
  //    ( in_loc + (half_warp_indx + i*blockDim.x/16)*12 )[half_warp_loc_indx] = ( in + 12*j_indxs[half_warp_indx + i*blockDim.x/16] )[half_warp_loc_indx];
  //  }
  //}
  // in, copy the rest
  //if( half_warp_indx<copy_unwnds_rest ){
  //  // each half-warp copies a lattice site
  //  if( half_warp_loc_indx<12 ){
  //    ( in_loc + (half_warp_indx + copy_unwnds*blockDim.x/16)*12 )[half_warp_loc_indx] = ( in + 12*j_indxs[half_warp_indx + copy_unwnds*blockDim.x/16] )[half_warp_loc_indx];
  //  }
  //}

  // out, copy unwnds
  //for( i=0; i<copy_unwnds; i++ ){
  //  // each half-warp copies a lattice site
  //  if( half_warp_loc_indx<12 ){
  //    ( out_loc + (half_warp_indx + i*blockDim.x/16)*12 )[half_warp_loc_indx] = ( out + 12*k_indxs[half_warp_indx + i*blockDim.x/16] )[half_warp_loc_indx];
  //  }
  //}
  // out, copy the rest
  //if( half_warp_indx<copy_unwnds_rest ){
  //  // each half-warp copies a lattice site
  //  if( half_warp_loc_indx<12 ){
  //    ( out_loc + (half_warp_indx + copy_unwnds*blockDim.x/16)*12 )[half_warp_loc_indx] = ( out + 12*k_indxs[half_warp_indx + copy_unwnds*blockDim.x/16] )[half_warp_loc_indx];
  //  }
  //}

  */

  // test:

  //if( k==85 ){
    //printf("k = %d, j = %d, in[%d][loc_ind] = %f + i%f, in_loc[][loc_ind] = %f + i%f\n", k, j, j, \
    //       cu_creal_PRECISION( (in+12*j)[threadIdx.x%6] ), cu_cimag_PRECISION( (in+12*j)[threadIdx.x%6] ), \
    //       cu_creal_PRECISION( (in_loc+12*get_index_in_shared(j_indxs, j))[threadIdx.x%6] ), cu_cimag_PRECISION( (in_loc+12*get_index_in_shared(j_indxs, j))[threadIdx.x%6] ) );
  //}

  __syncthreads();

  //_cuda_block_n_hopping_term_PRECISION_plus_6threads_naive(out_loc, in_loc, start, amount, s, idx, tmp_loc, ext_dir, gamma_coo, gamma_val, j_indxs, k_indxs);
  //_cuda_block_n_hopping_term_PRECISION_plus_6threads_naive(out, in, start, amount, s, idx, tmp_loc, ext_dir, gamma_coo, gamma_val, j_indxs, k_indxs);
  _cuda_block_n_hopping_term_PRECISION_plus_6threads_naive(out, in, start, amount, s, idx, tmp_loc, ext_dir, gamma_coo, gamma_val);

  //__syncthreads();

  // out, copy unwnds
  //for( i=0; i<copy_unwnds; i++ ){
  //  // each half-warp copies a lattice site
  //  if( half_warp_loc_indx<12 ){
  //    //( in_loc + (half_warp_indx + i*blockDim.x/16)*12 )[half_warp_loc_indx] = ( in + 12*j_indxs[half_warp_indx + i*blockDim.x/16] )[half_warp_loc_indx];
  //    ( out + 12*k_indxs[half_warp_indx + i*blockDim.x/16] )[half_warp_loc_indx] = ( out_loc + (half_warp_indx + i*blockDim.x/16)*12 )[half_warp_loc_indx];
  //  }
  //}
  // out, copy the rest
  //if( half_warp_indx<copy_unwnds_rest ){
  //  // each half-warp copies a lattice site
  //  if( half_warp_loc_indx<12 ){
  //    //( in_loc + (half_warp_indx + copy_unwnds*blockDim.x/16)*12 )[half_warp_loc_indx] = ( in + 12*j_indxs[half_warp_indx + copy_unwnds*blockDim.x/16] )[half_warp_loc_indx];
  //    ( out + 12*k_indxs[half_warp_indx + copy_unwnds*blockDim.x/16] )[half_warp_loc_indx] = ( out_loc + (half_warp_indx + copy_unwnds*blockDim.x/16)*12 )[half_warp_loc_indx];
  //  }
  //}

}


__forceinline__ __device__ void _cuda_block_n_hopping_term_PRECISION_plus_6threads_naive(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, int amount, \
                                                                                         schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf, \
                                                                                         int ext_dir, int* gamma_coo, cu_cmplx_PRECISION* gamma_val){
  //if amount==0 then even sites, if amount==1 then odd sites, else no oddeven split

  //int dir; // dir '=' {0,1,2,3} = {T,Z,Y,X}
  //int dir, a1, a2, n1, n2, k=0, j=0, i=0, **index = s->oe_index, *ind, *neighbor = s->op.neighbor_table, loc_ind=idx%6, spin, w, *gamma_coo, idx_in_cublock = idx%blockDim.x;
  int dir, a1, n1, k=0, j=0, i=0, **index = s->oe_index, *ind, *neighbor = s->op.neighbor_table, loc_ind=idx%6, spin, w, idx_in_cublock = idx%blockDim.x;
  //cu_cmplx_PRECISION *gamma_val;

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
    a1 = 0;
    n1 = s->dir_length[dir];
  }

  //a1 = 0; n1 = length_even[mu]+length_odd[mu];
  //a2 = 0; n2 = n1;

  ind = index[dir];

  // less threads in charge of this portion of execution, compute contribution due to even sites
  if( idx<6*(n1-a1) ){

    //printf("n1-a1 = %d\n", n1-a1);

    i = idx/6 + a1;
    k = ind[i];
    j = neighbor[4*k+dir];
    D_pt = D + 36*k + 9*dir;

    lphi = phi + 12*j;
    //lphi = phi + 12*get_index_in_shared(j_indxs, j);

    leta = eta + 12*k;
    //leta = eta + 12*get_index_in_shared(k_indxs, k);

    spin = (loc_ind/3)*2;
    //with this setup, gamma_val[0] gives spins 0 and 1, and gamma_val[1] spins 2 and 3
    //gamma_val = s->gamma_info_vals + dir*4 + spin;
    //gamma_coo = s->gamma_info_coo  + dir*4 + spin;

    // prp_T_PRECISION(...)
    buf1[ loc_ind ] = cu_csub_PRECISION( lphi[ loc_ind ], cu_cmul_PRECISION( (gamma_val + dir*4 + spin)[0], lphi[ 3*(gamma_coo + dir*4 + spin)[0] + loc_ind%3 ] ) );
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
    leta[ 6 + loc_ind ] = cu_cadd_PRECISION( leta[ 6 + loc_ind ], cu_cmul_PRECISION( (gamma_val + dir*4 + spin)[1], buf2[ 3*(gamma_coo + dir*4 + spin)[1] + loc_ind%3 ] ) );
  }

  // FIXME: is this sync necessary ?
  //__syncthreads();
}


// gamma_val and gamma_coo are both loaded into shared memory for this kernel
__global__ void cuda_block_n_hopping_term_PRECISION_minus_6threads_naive( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                                                          schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                          int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                                          int num_latt_site_var, block_struct* block, int ext_dir, int amount ){

  int idx;
  idx = threadIdx.x + blockDim.x * blockIdx.x;

  int DD_block_id, block_id, start;

  int nr_block_even_sites;
  nr_block_even_sites = s->num_block_even_sites;

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
  cu_cmplx_PRECISION *shared_data_loc = shared_data;
  cu_cmplx_PRECISION *tmp_loc;

  //the following are 'bare' values, i.e. with respect to the 0th element within a CUDA block
  // EVEN
  tmp_loc = shared_data_loc;

  if(idx < 6*nr_block_even_sites){
    tmp_loc[threadIdx.x] = make_cu_cmplx_PRECISION(0.0,0.0);
    tmp_loc[threadIdx.x + blockDim.x] = make_cu_cmplx_PRECISION(0.0,0.0);
  } //even


  _cuda_block_n_hopping_term_PRECISION_minus_6threads_naive(out, in, start, amount, s, idx, tmp_loc, ext_dir);
}


__forceinline__ __device__ void _cuda_block_n_hopping_term_PRECISION_minus_6threads_naive(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, int amount, schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf, int ext_dir){

  //if amount==0 then even sites, if amount==1 then odd sites, else no oddeven split

  //int dir; // dir '=' {0,1,2,3} = {T,Z,Y,X}
  //int dir, a1, a2, n1, n2, k=0, j=0, i=0, **index = s->oe_index, *ind, *neighbor = s->op.neighbor_table, loc_ind=idx%6, spin, w, *gamma_coo, idx_in_cublock = idx%blockDim.x;
  int dir, a1, n1, a2, n2, k=0, j=0, i=0, **index = s->oe_index, *ind, *neighbor = s->op.neighbor_table, loc_ind=idx%6, spin, w, *gamma_coo, idx_in_cublock = idx%blockDim.x;
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
    a2=n1; n2=a2+s->dir_length_odd[dir]; //for the - part
  }
  else if( amount==_ODD_SITES ){
    a1=s->dir_length_even[dir]; n1=a1+s->dir_length_odd[dir];
    a2=0; n2=a1;
  }
  else{
    a2 = 0;
    n2 = s->dir_length[dir];
  }

  //a1 = 0; n1 = length_even[mu]+length_odd[mu];
  //a2 = 0; n2 = n1;

  ind = index[dir];

  //and now, compute the contribution due to odd sites
  if( idx<6*(n2-a2) ){
    i = idx/6 + a2;
    k = ind[i];
    j = neighbor[4*k+dir];
    D_pt = D + 36*k + 9*dir;

    lphi = phi + 12*k;
    leta = eta + 12*j;

    spin = (loc_ind/3)*2;
    //with this setup, gamma_val[0] gives spins 0 and 1, and gamma_val[1] spins 2 and 3
    gamma_val = s->gamma_info_vals + dir*4 + spin;
    gamma_coo = s->gamma_info_coo  + dir*4 + spin;

    // prn_T_PRECISION(...)
    buf1[ loc_ind ] = cu_cadd_PRECISION( lphi[ loc_ind ], cu_cmul_PRECISION( gamma_val[0], lphi[ 3*gamma_coo[0] + loc_ind%3 ] ) );

  }

  __syncthreads();

  if( idx<6*(n2-a2) ){
    // nmvmh_PRECISION(...), twice
    buf2[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
    for( w=0; w<3; w++ ){
      buf2[ loc_ind ] = cu_csub_PRECISION( buf2[ loc_ind ], cu_cmul_PRECISION( cu_conj_PRECISION(D_pt[ loc_ind%3 + w*3 ]), buf1[ (loc_ind/3)*3 + w ] ) );
    }
  }

  __syncthreads();

  if( idx<6*(n2-a2) ){
    // pbn_su3_T_PRECISION(...)
    leta[ loc_ind ] = cu_csub_PRECISION( leta[ loc_ind ], buf2[ loc_ind ] );
    leta[ 6 + loc_ind ] = cu_csub_PRECISION( leta[ 6 + loc_ind ], cu_cmul_PRECISION( gamma_val[1], buf2[ 3*gamma_coo[1] + loc_ind%3 ] ) );
  }

  //__syncthreads();

}











































//********************************************************************************************************************************************************************

// gamma_val and gamma_coo are both loaded into shared memory for this kernel
__global__ void cuda_block_hopping_term_PRECISION_plus_6threads_naive( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                                                         schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                         int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                                         int num_latt_site_var, block_struct* block, int ext_dir, int amount ){

  int idx;
  idx = threadIdx.x + blockDim.x * blockIdx.x;

  int* gamma_coo;

  cu_cmplx_PRECISION* gamma_val;

  int DD_block_id, block_id, start;

  //int nr_block_even_sites;
  //nr_block_even_sites = s->num_block_even_sites;

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

  extern __shared__ cu_cmplx_PRECISION shared_data_bare[];
  cu_cmplx_PRECISION* shared_data = shared_data_bare;

  gamma_coo = (int*)shared_data;
  shared_data = (cu_cmplx_PRECISION*)((int*)shared_data + 16);

  gamma_val = shared_data;
  shared_data = shared_data + 16;

  cu_cmplx_PRECISION *tmp_loc;
  tmp_loc = shared_data;
  shared_data = shared_data + 2*blockDim.x;

  // loading gamma coordinates into shared memory
  if( threadIdx.x<16 ){
    gamma_coo[threadIdx.x] = s->gamma_info_coo[threadIdx.x];
  }

  // loading gamma values into shared memory
  if( threadIdx.x<16 ){
    gamma_val[threadIdx.x] = s->gamma_info_vals[threadIdx.x];
  }

  // initializing to zero a local buffer for temporary computations
  //if(idx < 6*nr_block_even_sites){
  if(idx < 6*( (amount==_EVEN_SITES)?(s->dir_length_even[ext_dir]):(s->dir_length_odd[ext_dir]) )){
    tmp_loc[threadIdx.x] = make_cu_cmplx_PRECISION(0.0,0.0);
    tmp_loc[threadIdx.x + blockDim.x] = make_cu_cmplx_PRECISION(0.0,0.0);
  }

  __syncthreads();

  _cuda_block_hopping_term_PRECISION_plus_6threads_naive(out, in, start, amount, s, idx, tmp_loc, ext_dir, gamma_coo, gamma_val);

}


__forceinline__ __device__ void _cuda_block_hopping_term_PRECISION_plus_6threads_naive(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, int amount, \
                                                                                         schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf, \
                                                                                         int ext_dir, int* gamma_coo, cu_cmplx_PRECISION* gamma_val){
  //if amount==0 then even sites, if amount==1 then odd sites, else no oddeven split

  //int dir; // dir '=' {0,1,2,3} = {T,Z,Y,X}
  //int dir, a1, a2, n1, n2, k=0, j=0, i=0, **index = s->oe_index, *ind, *neighbor = s->op.neighbor_table, loc_ind=idx%6, spin, w, *gamma_coo, idx_in_cublock = idx%blockDim.x;
  int dir, a1, n1, k=0, j=0, i=0, **index = s->oe_index, *ind, *neighbor = s->op.neighbor_table, loc_ind=idx%6, spin, w, idx_in_cublock = idx%blockDim.x;
  //cu_cmplx_PRECISION *gamma_val;

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
    a1 = 0;
    n1 = s->dir_length[dir];
  }

  ind = index[dir];

  // less threads in charge of this portion of execution, compute contribution due to even sites
  if( idx<6*(n1-a1) ){

    //printf("n1-a1 = %d\n", n1-a1);

    i = idx/6 + a1;
    k = ind[i];
    j = neighbor[4*k+dir];
    D_pt = D + 36*k + 9*dir;

    lphi = phi + 12*j;
    //lphi = phi + 12*get_index_in_shared(j_indxs, j);

    leta = eta + 12*k;
    //leta = eta + 12*get_index_in_shared(k_indxs, k);

    spin = (loc_ind/3)*2;
    //with this setup, gamma_val[0] gives spins 0 and 1, and gamma_val[1] spins 2 and 3
    //gamma_val = s->gamma_info_vals + dir*4 + spin;
    //gamma_coo = s->gamma_info_coo  + dir*4 + spin;

    // prp_T_PRECISION(...)
    buf1[ loc_ind ] = cu_csub_PRECISION( lphi[ loc_ind ], cu_cmul_PRECISION( (gamma_val + dir*4 + spin)[0], lphi[ 3*(gamma_coo + dir*4 + spin)[0] + loc_ind%3 ] ) );
  }

  __syncthreads();

  if( idx<6*(n1-a1) ){
    // mvm_PRECISION(...), twice
    buf2[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
    for( w=0; w<3; w++ ){
      buf2[ loc_ind ] = cu_cadd_PRECISION( buf2[ loc_ind ], cu_cmul_PRECISION( D_pt[ (loc_ind*3)%9 + w ], buf1[ (loc_ind/3)*3 + w ] ) );
    }
  }

  __syncthreads();

  if( idx<6*(n1-a1) ){
    // pbp_su3_T_PRECISION(...)
    leta[ loc_ind ] = cu_csub_PRECISION( leta[ loc_ind ], buf2[ loc_ind ] );
    leta[ 6 + loc_ind ] = cu_cadd_PRECISION( leta[ 6 + loc_ind ], cu_cmul_PRECISION( (gamma_val + dir*4 + spin)[1], buf2[ 3*(gamma_coo + dir*4 + spin)[1] + loc_ind%3 ] ) );
  }

  // FIXME: is this sync necessary ?
  //__syncthreads();
}


// gamma_val and gamma_coo are both loaded into shared memory for this kernel
__global__ void cuda_block_hopping_term_PRECISION_minus_6threads_naive( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                                                          schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                          int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                                          int num_latt_site_var, block_struct* block, int ext_dir, int amount ){

  int idx;
  idx = threadIdx.x + blockDim.x * blockIdx.x;

  int DD_block_id, block_id, start;

  int nr_block_even_sites;
  nr_block_even_sites = s->num_block_even_sites;

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
  cu_cmplx_PRECISION *shared_data_loc = shared_data;
  cu_cmplx_PRECISION *tmp_loc;

  //the following are 'bare' values, i.e. with respect to the 0th element within a CUDA block
  // EVEN
  tmp_loc = shared_data_loc;

  if(idx < 6*nr_block_even_sites){
    tmp_loc[threadIdx.x] = make_cu_cmplx_PRECISION(0.0,0.0);
    tmp_loc[threadIdx.x + blockDim.x] = make_cu_cmplx_PRECISION(0.0,0.0);
  } //even


  _cuda_block_hopping_term_PRECISION_minus_6threads_naive(out, in, start, amount, s, idx, tmp_loc, ext_dir);
}


__forceinline__ __device__ void _cuda_block_hopping_term_PRECISION_minus_6threads_naive(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, int amount, schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf, int ext_dir){

  //if amount==0 then even sites, if amount==1 then odd sites, else no oddeven split

  //int dir; // dir '=' {0,1,2,3} = {T,Z,Y,X}
  //int dir, a1, a2, n1, n2, k=0, j=0, i=0, **index = s->oe_index, *ind, *neighbor = s->op.neighbor_table, loc_ind=idx%6, spin, w, *gamma_coo, idx_in_cublock = idx%blockDim.x;
  int dir, a1, n1, a2, n2, k=0, j=0, i=0, **index = s->oe_index, *ind, *neighbor = s->op.neighbor_table, loc_ind=idx%6, spin, w, *gamma_coo, idx_in_cublock = idx%blockDim.x;
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
    a2=n1; n2=a2+s->dir_length_odd[dir]; //for the - part
  }
  else if( amount==_ODD_SITES ){
    a1=s->dir_length_even[dir]; n1=a1+s->dir_length_odd[dir];
    a2=0; n2=a1;
  }
  else{
    a2 = 0;
    n2 = s->dir_length[dir];
  }

  ind = index[dir];

  //and now, compute the contribution due to odd sites
  if( idx<6*(n2-a2) ){
    i = idx/6 + a2;
    k = ind[i];
    j = neighbor[4*k+dir];
    D_pt = D + 36*k + 9*dir;

    lphi = phi + 12*k;
    leta = eta + 12*j;

    spin = (loc_ind/3)*2;
    //with this setup, gamma_val[0] gives spins 0 and 1, and gamma_val[1] spins 2 and 3
    gamma_val = s->gamma_info_vals + dir*4 + spin;
    gamma_coo = s->gamma_info_coo  + dir*4 + spin;

    // prn_T_PRECISION(...)
    buf1[ loc_ind ] = cu_cadd_PRECISION( lphi[ loc_ind ], cu_cmul_PRECISION( gamma_val[0], lphi[ 3*gamma_coo[0] + loc_ind%3 ] ) );

  }

  __syncthreads();

  if( idx<6*(n2-a2) ){
    // mvmh_PRECISION(...), twice
    buf2[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
    for( w=0; w<3; w++ ){
      buf2[ loc_ind ] = cu_cadd_PRECISION( buf2[ loc_ind ], cu_cmul_PRECISION( cu_conj_PRECISION(D_pt[ loc_ind%3 + w*3 ]), buf1[ (loc_ind/3)*3 + w ] ) );
    }
  }

  __syncthreads();

  if( idx<6*(n2-a2) ){
    // pbn_su3_T_PRECISION(...)
    leta[ loc_ind ] = cu_csub_PRECISION( leta[ loc_ind ], buf2[ loc_ind ] );
    leta[ 6 + loc_ind ] = cu_csub_PRECISION( leta[ 6 + loc_ind ], cu_cmul_PRECISION( gamma_val[1], buf2[ 3*gamma_coo[1] + loc_ind%3 ] ) );
  }

}

//********************************************************************************************************************************************************************

































__global__ void cuda_block_solve_update_6threads_opt( cu_cmplx_PRECISION* phi, cu_cmplx_PRECISION* r, cu_cmplx_PRECISION* latest_iter, \
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
    }
  }
  // odd
  if(idx < 6*nr_block_odd_sites){
    for(i=0; i<2; i++){
      ( latest_iter + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = ( tmp2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
    }
  }
  // even
  if(idx < 6*nr_block_even_sites){
    for(i=0; i<2; i++){
      ( phi + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = make_cu_cmplx_PRECISION( cu_creal_PRECISION(( phi + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) +\
                                                                                                   cu_creal_PRECISION(( tmp2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]),
                                                                                                   cu_cimag_PRECISION(( phi + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                                                                                                   cu_cimag_PRECISION(( tmp2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) );
    }
  }
  // odd
  if(idx < 6*nr_block_odd_sites){
    for(i=0; i<2; i++){
      ( phi + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = make_cu_cmplx_PRECISION( 
                                                                     cu_creal_PRECISION(( phi + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                                                                     cu_creal_PRECISION(( tmp2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]),
                                                                     cu_cimag_PRECISION(( phi + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + \
                                                                     cu_cimag_PRECISION(( tmp2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) );
    }
  }
  // even
  if(idx < 6*nr_block_even_sites){
    for(i=0; i<2; i++){
      ( r + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = ( tmp3 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
    }
  }
  // odd
  if(idx < 6*nr_block_odd_sites){
    for(i=0; i<2; i++){
      ( r + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = make_cu_cmplx_PRECISION(0.0,0.0);
    }
  }

}



// WARNING: the use of this function may lead to performance reduction
__global__ void cuda_block_n_hopping_term_PRECISION( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                                     schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                     int csw, int nr_DD_blocks_to_compute, int* DD_blocks_to_compute, \
                                                     int num_latt_site_var, block_struct* block, int sites_to_compute ){

  int dir;

  int threads_per_cublock, nr_threads, nr_threads_per_DD_block;
  size_t tot_shared_mem;

  // we choose here a multiple of 96 due to being the smallest nr divisible by 32, but also divisible by 6
  threads_per_cublock = 96;

  nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites;
  nr_threads = nr_threads*(12/2);
  nr_threads = nr_threads*nr_DD_blocks_to_compute;

  nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

  if( threadIdx.x==0 ){

    // hopping term, even sites
    //tot_shared_mem = 2*4*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 2*size_D_oeclov*(threads_per_cublock/6)*sizeof(cu_config_PRECISION);
    tot_shared_mem = 1*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION);
    for( dir=0; dir<4; dir++ ){
      cuda_block_n_hopping_term_PRECISION_plus_6threads_naive<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem >>> \
                                                             (out, in, s, thread_id, csw, nr_threads_per_DD_block, DD_blocks_to_compute, num_latt_site_var, block, dir, sites_to_compute);
      cuda_block_n_hopping_term_PRECISION_minus_6threads_naive<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem >>> \
                                                              (out, in, s, thread_id, csw, nr_threads_per_DD_block, DD_blocks_to_compute, num_latt_site_var, block, dir, sites_to_compute);
    }
  }

}



extern "C" void cuda_apply_block_schur_complement_PRECISION( cuda_vector_PRECISION out, cuda_vector_PRECISION in,
                                                             schwarz_PRECISION_struct *s, level_struct *l, int nr_DD_blocks_to_compute,
                                                             int* DD_blocks_to_compute, cudaStream_t *streams, int stream_id, int sites_to_solve ){

  int dir, size_D_oeclov;

  cu_cmplx_PRECISION **tmp, *tmp0, *tmp1;
  tmp = (s->s_on_gpu_cpubuff).oe_buf;
  tmp0 = tmp[0];
  tmp1 = tmp[1];

  int threads_per_cublock_diagops = 32;

  int threads_per_cublock, nr_threads, nr_threads_per_DD_block;
  size_t tot_shared_mem;

  // this is the size of the local matrix, i.e. per lattice site. 12^2=144, but ... (??)
  //size_D_oeclov = (g.csw!=0) ? 72 : 12;
  size_D_oeclov = (g.csw!=0) ? 42 : 12;

  // block_diag_ee_PRECISION( out, in, start, s, l, threading );
  nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
  nr_threads = nr_threads*(12/6); // threads per site
  nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
  nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
  tot_shared_mem = 2*(6*threads_per_cublock_diagops)*sizeof(cu_cmplx_PRECISION) + 1*size_D_oeclov*(threads_per_cublock_diagops/2)*sizeof(cu_config_PRECISION);
  cuda_block_diag_ee_PRECISION_2threads_opt<<< nr_threads/threads_per_cublock_diagops, threads_per_cublock_diagops, tot_shared_mem, streams[stream_id] >>> \
                                           (out, in, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block);

  //vector_PRECISION_define( tmp[0], 0, start + 12*s->num_block_even_sites, start + s->block_vector_size, l );
  nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites;
  nr_threads = nr_threads*(12/2);
  nr_threads = nr_threads*nr_DD_blocks_to_compute;
  nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
  threads_per_cublock = 96;
  cuda_block_oe_vector_PRECISION_define_6threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                                    (tmp0, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, \
                                                    l->num_lattice_site_var, (s->cu_s).block, _ODD_SITES, make_cu_cmplx_PRECISION(0.0,0.0));

  //block_hopping_term_PRECISION( tmp[0], in, start, _ODD_SITES, s, l, threading );
  threads_per_cublock = 96;
  tot_shared_mem = 1*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 16*sizeof(cu_cmplx_PRECISION) + 16*sizeof(int);
  //tot_shared_mem = (2+1)*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION); // to load in, out and allocate tmp_loc (this last one to use as buf1 and buf2)
  //tot_shared_mem += 1*36*(threads_per_cublock/6)*sizeof(cu_config_PRECISION); // to load s->op.D
  //tot_shared_mem += 16*sizeof(cu_cmplx_PRECISION) + 16*sizeof(int); // to load info related to Gamma matrices
  //tot_shared_mem += 2*(threads_per_cublock/6)*sizeof(int); // to store indices of sites to compute and neighbors
  for( dir=0; dir<4; dir++ ){

    nr_threads = (sites_to_solve==_EVEN_SITES)?s->dir_length_even[dir]:s->dir_length_odd[dir];
    nr_threads = nr_threads*(12/2);
    nr_threads = nr_threads*nr_DD_blocks_to_compute;
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

    cuda_block_hopping_term_PRECISION_plus_6threads_naive<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                                         (tmp0, in, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, dir, _ODD_SITES);
    cuda_block_hopping_term_PRECISION_minus_6threads_naive<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                                          (tmp0, in, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, dir, _ODD_SITES);
  }

  // TODO: remove the following hopping computation
  //block_n_hopping_term_PRECISION( out, tmp[1], start, _EVEN_SITES, s, l, threading );
  threads_per_cublock = 96;
  tot_shared_mem = 1*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 16*sizeof(cu_cmplx_PRECISION) + 16*sizeof(int);
  //tot_shared_mem = (2+1)*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION); // to load in, out and allocate tmp_loc (this last one to use as buf1 and buf2)
  //tot_shared_mem += 1*36*(threads_per_cublock/6)*sizeof(cu_config_PRECISION); // to load s->op.D
  //tot_shared_mem += 16*sizeof(cu_cmplx_PRECISION) + 16*sizeof(int); // to load info related to Gamma matrices
  //tot_shared_mem += 2*(threads_per_cublock/6)*sizeof(int); // to store indices of sites to compute and neighbors
  for( dir=0; dir<4; dir++ ){

    nr_threads = (sites_to_solve==_EVEN_SITES)?s->dir_length_even[dir]:s->dir_length_odd[dir];
    nr_threads = nr_threads*(12/2);
    nr_threads = nr_threads*nr_DD_blocks_to_compute;
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

    //cuda_block_n_hopping_term_PRECISION_plus_6threads_naive<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
    //                                                       (tmp0, in, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, dir, _ODD_SITES);
    //cuda_block_n_hopping_term_PRECISION_minus_6threads_naive<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
    //                                                        (tmp0, in, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, dir, _ODD_SITES);
  }

  //block_diag_oo_inv_PRECISION( tmp[1], tmp[0], start, s, l, threading );
  // diag_oo inv
  nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
  nr_threads = nr_threads*(12/6); // threads per site
  nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
  nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
  tot_shared_mem = 2*(6*threads_per_cublock_diagops)*sizeof(cu_cmplx_PRECISION) + 1*size_D_oeclov*(threads_per_cublock_diagops/2)*sizeof(cu_config_PRECISION);
  cuda_block_diag_oo_inv_PRECISION_2threads_opt<<< nr_threads/threads_per_cublock_diagops, threads_per_cublock_diagops, tot_shared_mem, streams[stream_id] >>> \
                                               (tmp1, tmp0, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block);

  //block_n_hopping_term_PRECISION( out, tmp[1], start, _EVEN_SITES, s, l, threading );
  // hopping term, even sites
  threads_per_cublock = 96;
  tot_shared_mem = 1*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 16*sizeof(cu_cmplx_PRECISION) + 16*sizeof(int);
  //tot_shared_mem = (2+1)*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION); // to load in, out and allocate tmp_loc (this last one to use as buf1 and buf2)
  //tot_shared_mem += 1*36*(threads_per_cublock/6)*sizeof(cu_config_PRECISION); // to load s->op.D
  //tot_shared_mem += 16*sizeof(cu_cmplx_PRECISION) + 16*sizeof(int); // to load info related to Gamma matrices
  //tot_shared_mem += 2*(threads_per_cublock/6)*sizeof(int); // to store indices of sites to compute and neighbors
  for( dir=0; dir<4; dir++ ){

    nr_threads = (sites_to_solve==_EVEN_SITES)?s->dir_length_even[dir]:s->dir_length_odd[dir];
    nr_threads = nr_threads*(12/2);
    nr_threads = nr_threads*nr_DD_blocks_to_compute;
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

    cuda_block_n_hopping_term_PRECISION_plus_6threads_naive<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                                           (out, tmp1, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, dir, _EVEN_SITES);
    cuda_block_n_hopping_term_PRECISION_minus_6threads_naive<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                                            (out, tmp1, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, dir, _EVEN_SITES);
  }

}



// sites_to_solve = {_EVEN_SITES, _ODD_SITES, _FULL_SYSTEM}
extern "C" void cuda_local_minres_PRECISION( cuda_vector_PRECISION phi, cuda_vector_PRECISION eta, cuda_vector_PRECISION latest_iter,
                                             schwarz_PRECISION_struct *s, level_struct *l, int nr_DD_blocks_to_compute,
                                             int* DD_blocks_to_compute, cudaStream_t *streams, int stream_id, int sites_to_solve ) {

  // This local_minres performs an inversion on EVEN sites only

  int i, n = l->block_iter;
  cuda_vector_PRECISION Dr = (s->cu_s).local_minres_buffer[0];
  cuda_vector_PRECISION r = (s->cu_s).local_minres_buffer[1];
  cuda_vector_PRECISION lphi = (s->cu_s).local_minres_buffer[2];
  cu_cmplx_PRECISION alpha;
  //void (*block_op)() = (l->depth==0)?(g.odd_even?apply_block_schur_complement_PRECISION:block_d_plus_clover_PRECISION)
  //                                  :coarse_block_operator_PRECISION;

  int threads_per_cublock, nr_threads, nr_threads_per_DD_block;
  size_t tot_shared_mem;

  // copy: r <----- eta
  // the use of _EVEN_SITES comes from the CPU code: end = (g.odd_even&&l->depth==0)?start+12*s->num_block_even_sites:start+s->block_vector_size
  //vector_PRECISION_copy( r, eta, start, end, l );
  nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
  nr_threads = nr_threads*(12/2); // threads per site
  nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
  nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
  threads_per_cublock = 96;
  cuda_block_oe_vector_PRECISION_copy_6threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                                  (r, eta, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, sites_to_solve);

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
    cuda_apply_block_schur_complement_PRECISION( Dr, r, s, l, nr_DD_blocks_to_compute, DD_blocks_to_compute, streams, stream_id, _EVEN_SITES );

    // TODO: remove the following line
    alpha = make_cu_cmplx_PRECISION(1.0,3.0);

    //printf("i=%d (within for loop of cuda_local_min_res)\n", i);

//    // alpha = <Dr,r>/<Dr,Dr>
//    //alpha = local_xy_over_xx_PRECISION( Dr, r, start, end, l );
//    // To be able to call the current implementation of the dot product,
//    // threads_per_cublock has to be a power of 2
    threads_per_cublock = 64;
    nr_threads = threads_per_cublock;
    nr_threads = nr_threads*nr_DD_blocks_to_compute;
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
    // buffer to store partial sums of the overall-per-DD-block dot product
    tot_shared_mem = 2*(threads_per_cublock)*sizeof(cu_cmplx_PRECISION);
    cuda_local_xy_over_xx_PRECISION<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>>
                                   ( Dr, r, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, sites_to_solve );

    // TODO: remove the following line
    //cuda_safe_call( cudaDeviceSynchronize() );

    // phi += alpha * r
    //vector_PRECISION_saxpy( lphi, lphi, r, alpha, start, end, l );
    nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
    nr_threads = nr_threads*(12/2); // threads per site
    nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
    threads_per_cublock = 96;
    cuda_block_oe_vector_PRECISION_saxpy_6threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                                     (lphi, lphi, r, alpha, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, sites_to_solve);

    // r -= alpha * Dr
    // vector_PRECISION_saxpy( r, r, Dr, -alpha, start, end, l );
    nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
    nr_threads = nr_threads*(12/2); // threads per site
    nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
    threads_per_cublock = 96;
    cuda_block_oe_vector_PRECISION_saxpy_6threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                                     (r, r, Dr, cu_csub_PRECISION(make_cu_cmplx_PRECISION(0.0,0.0), alpha), s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, sites_to_solve);

  }

  //vector_PRECISION_copy( latest_iter, lphi, start, end, l );
  if ( latest_iter != NULL ){
    nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
    nr_threads = nr_threads*(12/2); // threads per site
    nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
    threads_per_cublock = 96;
    cuda_block_oe_vector_PRECISION_copy_6threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                                    (latest_iter, lphi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, sites_to_solve);
  }

  //vector_PRECISION_plus( phi, phi, lphi, start, end, l );
  if ( phi != NULL ){
    nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
    nr_threads = nr_threads*(12/2); // threads per site
    nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
    threads_per_cublock = 96;
    cuda_block_oe_vector_PRECISION_plus_6threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                                    (phi, phi, lphi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, sites_to_solve);
  }

  //vector_PRECISION_copy( eta, r, start, end, l );
  nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
  nr_threads = nr_threads*(12/2); // threads per site
  nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
  nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
  threads_per_cublock = 96;
  cuda_block_oe_vector_PRECISION_copy_6threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                                  (eta, r, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, sites_to_solve);

}



extern "C" void cuda_blocks_vector_copy_noncontig_PRECISION_naive( cuda_vector_PRECISION out, cuda_vector_PRECISION in, int nr_DD_blocks_to_compute,
                                                                   schwarz_PRECISION_struct* s, level_struct *l, int* DD_blocks_to_compute, cudaStream_t *streams){

  int i, b_start;
  for( i=0; i<nr_DD_blocks_to_compute; i++ ){
    b_start = s->block[DD_blocks_to_compute[i]].start * l->num_lattice_site_var;
    cuda_vector_PRECISION_copy((void*)out, (void*)in, b_start, s->block_vector_size, l, _D2D, _CUDA_ASYNC, 0, streams );
  }
}


// Use of dynamic parallelism to make _D2D copies
__global__ void cuda_blocks_vector_copy_noncontig_PRECISION_dyn_dev( cuda_vector_PRECISION out, cuda_vector_PRECISION in, int* DD_blocks_to_compute, \
                                         block_struct* block, int num_latt_site_var, schwarz_PRECISION_struct_on_gpu *s){
                                         
  int block_id, start;

  block_id = DD_blocks_to_compute[blockIdx.x];
  start = block[block_id].start * num_latt_site_var;

  if( threadIdx.x==0 ){
    cudaMemcpyAsync(out + start, in + start, (s->block_vector_size)*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToDevice);
  }

}



// Use of dynamic parallelism
extern "C" void cuda_blocks_vector_copy_noncontig_PRECISION_dyn( cuda_vector_PRECISION out, cuda_vector_PRECISION in, int nr_DD_blocks_to_compute,
                                                                 schwarz_PRECISION_struct* s, level_struct *l, int* DD_blocks_to_compute, cudaStream_t *streams,
                                                                 int stream_id){
  cuda_blocks_vector_copy_noncontig_PRECISION_dyn_dev<<< nr_DD_blocks_to_compute, 32, 0, streams[stream_id] >>>
                                                     (out, in, DD_blocks_to_compute, (s->cu_s).block, l->num_lattice_site_var, s->s_on_gpu);
}



extern "C" void cuda_block_solve_oddeven_PRECISION( cuda_vector_PRECISION phi, cuda_vector_PRECISION r, cuda_vector_PRECISION latest_iter,
                                                    int start, int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s, level_struct *l,
                                                    struct Thread *threading, int stream_id, cudaStream_t *streams, int solve_at_cpu, int color,
                                                    int* DD_blocks_to_compute_gpu, int* DD_blocks_to_compute_cpu ) {

  if(solve_at_cpu){
    block_solve_oddeven_PRECISION( (vector_PRECISION)phi, (vector_PRECISION)r, (vector_PRECISION)latest_iter,
                                   start, s, l, threading );
  } else {

    int threads_per_cublock, nr_threads, size_D_oeclov, nr_threads_per_DD_block, dir;
    size_t tot_shared_mem;

    int threads_per_cublock_diagops = 32;

    // we choose here a multiple of 96 due to being the smallest nr divisible by 32, but also divisible by 6
    //threads_per_cublock = 96;

    // the 'nr_threads' var needed is computed like this: max between num_block_even_sites and num_block_odd_sites, and then
    //                                                   for each lattice site (of those even-odd), we need 12/2 really independent
    //                                                   components, due to gamma5 symmetry. I.e. each thread is in charge of
    //                                                   one site component !

    // this is the size of the local matrix, i.e. per lattice site. 12^2=144, but ... (??)
    //size_D_oeclov = (g.csw!=0) ? 72 : 12;
    size_D_oeclov = (g.csw!=0) ? 42 : 12;

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
    //tot_shared_mem = (2+3)*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 2*size_D_oeclov*(threads_per_cublock/6)*sizeof(cu_config_PRECISION);

    cu_cmplx_PRECISION **tmp, *tmp2, *tmp3;
    tmp = (s->s_on_gpu_cpubuff).oe_buf;
    tmp2 = tmp[2];
    tmp3 = tmp[3];

    nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
    nr_threads = nr_threads*(12/2); // threads per site
    nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
    threads_per_cublock = 96;
    cuda_block_oe_vector_PRECISION_copy_6threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                                    (tmp3, r, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block, _FULL_SYSTEM);
    //cuda_blocks_vector_copy_noncontig_PRECISION_naive(tmp3, r, nr_DD_blocks_to_compute, s, l, DD_blocks_to_compute_cpu, streams);
    //cuda_blocks_vector_copy_noncontig_PRECISION_dyn(tmp3, r, nr_DD_blocks_to_compute, s, l, DD_blocks_to_compute_gpu, streams, 0);

    // diag_oo inv
    nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
    nr_threads = nr_threads*(12/6); // threads per site
    nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
    tot_shared_mem = 2*(6*threads_per_cublock_diagops)*sizeof(cu_cmplx_PRECISION) + 1*size_D_oeclov*(threads_per_cublock_diagops/2)*sizeof(cu_config_PRECISION);
    cuda_block_diag_oo_inv_PRECISION_2threads_opt<<< nr_threads/threads_per_cublock_diagops, threads_per_cublock_diagops, tot_shared_mem, streams[stream_id] >>> \
                                                 (tmp2, tmp3, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block);

    // hopping term, even sites
    threads_per_cublock = 96;
    tot_shared_mem = 1*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 16*sizeof(cu_cmplx_PRECISION) + 16*sizeof(int);
    //tot_shared_mem = (2+1)*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION); // to load in, out and allocate tmp_loc (this last one to use as buf1 and buf2)
    //tot_shared_mem += 1*36*(threads_per_cublock/6)*sizeof(cu_config_PRECISION); // to load s->op.D
    //tot_shared_mem += 16*sizeof(cu_cmplx_PRECISION) + 16*sizeof(int); // to load info related to Gamma matrices
    //tot_shared_mem += 2*(threads_per_cublock/6)*sizeof(int); // to store indices of sites to compute and neighbors
    for( dir=0; dir<4; dir++ ){

      //nr_threads = (sites_to_solve==_EVEN_SITES)?s->dir_length_even[dir]:s->dir_length_odd[dir];
      nr_threads = s->dir_length_even[dir];
      nr_threads = nr_threads*(12/2);
      nr_threads = nr_threads*nr_DD_blocks_to_compute;
      nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

      cuda_block_n_hopping_term_PRECISION_plus_6threads_naive<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                                             (tmp3, tmp2, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block, dir, _EVEN_SITES);
      cuda_block_n_hopping_term_PRECISION_minus_6threads_naive<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                                              (tmp3, tmp2, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block, dir, _EVEN_SITES);
    }
    //cuda_block_n_hopping_term_PRECISION<<< 1, 32, 0, streams[stream_id] >>> \
    //                                   (tmp3, tmp2, s->s_on_gpu, g.my_rank, g.csw, nr_DD_blocks_to_compute, DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block, _EVEN_SITES);

    cuda_local_minres_PRECISION( NULL, tmp3, tmp2, s, l, nr_DD_blocks_to_compute, DD_blocks_to_compute_gpu, streams, stream_id, _EVEN_SITES );

    // hopping term, odd sites
    threads_per_cublock = 96;
    tot_shared_mem = 1*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 16*sizeof(cu_cmplx_PRECISION) + 16*sizeof(int);
    //tot_shared_mem = (2+1)*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION); // to load in, out and allocate tmp_loc (this last one to use as buf1 and buf2)
    //tot_shared_mem += 1*36*(threads_per_cublock/6)*sizeof(cu_config_PRECISION); // to load s->op.D
    //tot_shared_mem += 16*sizeof(cu_cmplx_PRECISION) + 16*sizeof(int); // to load info related to Gamma matrices
    //tot_shared_mem += 2*(threads_per_cublock/6)*sizeof(int); // to store indices of sites to compute and neighbors
    for( dir=0; dir<4; dir++ ){

      //nr_threads = (sites_to_solve==_EVEN_SITES)?s->dir_length_even[dir]:s->dir_length_odd[dir];
      nr_threads = s->dir_length_odd[dir];
      nr_threads = nr_threads*(12/2);
      nr_threads = nr_threads*nr_DD_blocks_to_compute;
      nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

      cuda_block_n_hopping_term_PRECISION_plus_6threads_naive<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                                             (tmp3, tmp2, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block, dir, _ODD_SITES);
      cuda_block_n_hopping_term_PRECISION_minus_6threads_naive<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                                              (tmp3, tmp2, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block, dir, _ODD_SITES);
    }

    // diag_oo inv
    nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
    nr_threads = nr_threads*(12/6); // threads per site
    nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
    tot_shared_mem = 2*(6*threads_per_cublock_diagops)*sizeof(cu_cmplx_PRECISION) + 1*size_D_oeclov*(threads_per_cublock_diagops/2)*sizeof(cu_config_PRECISION);
    cuda_block_diag_oo_inv_PRECISION_2threads_opt<<< nr_threads/threads_per_cublock_diagops, threads_per_cublock_diagops, tot_shared_mem, streams[stream_id] >>> \
                                                 (tmp2, tmp3, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block);

    // update phi and latest_iter, and r
    nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
    nr_threads = nr_threads*(12/2); // threads per site
    nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
    threads_per_cublock = 96;
    cuda_block_solve_update_6threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>>
                                        (phi, r, latest_iter, s->s_on_gpu, g.my_rank, g.csw, 0, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block);

    // TODO: eventually, remove this line
    //cuda_check_error( _HARD_CHECK );
    //exit(1);

  }
}






//************************************************************************************************************


  //_cuda_block_PRECISION_boundary_op_naive(out, in, start, amount, s, idx, tmp_loc, ext_dir);

__forceinline__ __device__ void _cuda_block_PRECISION_boundary_op_plus_naive(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int block_id, schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf, int ext_dir, block_struct* block){

  int dir, i, loc_ind=idx%6, spin, w, *gamma_coo, idx_in_cublock = idx%blockDim.x, index, neighbor_index, *bbl = s->block_boundary_length;
  cu_cmplx_PRECISION *gamma_val;

  cu_cmplx_PRECISION *buf1, *buf2;
  buf += (idx_in_cublock/6)*12;
  buf1 = buf;
  buf2 = buf + 6;

  //TODO: implement a dynamic way of loading s->op.D into __shared__

  //cu_config_PRECISION* D = s->op.D + (start/12)*36;
  cu_config_PRECISION* D = s->op.D;
  cu_config_PRECISION* D_pt;

  cu_cmplx_PRECISION *eta_pt, *phi_pt; //eta and phi are already shifted by 'start'

  // TODO: is there a reason for this not to be integrated with extra_dir ?
  dir = ext_dir;

  i = idx/6;
  i *= 2;
  i += bbl[ 2*ext_dir ];

  index = block[block_id].bt_on_gpu[i];
  neighbor_index = block[block_id].bt_on_gpu[i+1];

  if( block_id==619 && loc_ind==0 ){
    //printf("(block_id=%d) index=%d, neighbor_index=%d, idx=%d, i=%d, ext_dir=%d, bbl[ 2*ext_dir ]=%d\n", block_id, index, neighbor_index, idx, i, ext_dir, bbl[ 2*ext_dir ]);
  }

  D_pt = D + 36*index + 9*ext_dir;

  phi_pt = phi + 12*neighbor_index;
  eta_pt = eta + 12*index;

  spin = (loc_ind/3)*2;
  //with this setup, gamma_val[0] gives spins 0 and 1, and gamma_val[1] spins 2 and 3
  gamma_val = s->gamma_info_vals + dir*4 + spin;
  gamma_coo = s->gamma_info_coo  + dir*4 + spin;

  // prp_T_PRECISION(...)
  buf1[ loc_ind ] = cu_csub_PRECISION( phi_pt[ loc_ind ], cu_cmul_PRECISION( gamma_val[0], phi_pt[ 3*gamma_coo[0] + loc_ind%3 ] ) );

  __syncthreads();

  // mvm_PRECISION(...), twice
  buf2[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
  for( w=0; w<3; w++ ){
    buf2[ loc_ind ] = cu_cadd_PRECISION( buf2[ loc_ind ], cu_cmul_PRECISION( D_pt[ (loc_ind*3)%9 + w ], buf1[ (loc_ind/3)*3 + w ] ) );
  }

  __syncthreads();

  //pbp_su3_T_PRECISION( buf2, eta_pt );
  eta_pt[ loc_ind ] = cu_csub_PRECISION( eta_pt[ loc_ind ], buf2[ loc_ind ] );
  eta_pt[ 6 + loc_ind ] = cu_cadd_PRECISION( eta_pt[ 6 + loc_ind ], cu_cmul_PRECISION( gamma_val[1], buf2[ 3*gamma_coo[1] + loc_ind%3 ] ) );
}







__global__ void cuda_block_PRECISION_boundary_op_plus_naive( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                                             schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                             int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                             int num_latt_site_var, block_struct* block, int ext_dir ){

  int idx;
  idx = threadIdx.x + blockDim.x * blockIdx.x;

  //int DD_block_id, block_id, start;
  int DD_block_id, block_id;

  // not really a DD block id, but rather a linear counting of a grouping (per DD block) of CUDA threads
  DD_block_id = idx/nr_threads_per_DD_block;

  // offsetting idx to make it zero at the beginning of the threads living within a DD block
  idx = idx%nr_threads_per_DD_block;

  // this int will be the ACTUAL DD block ID, in the sense of accessing data from e.g. block_struct* block
  block_id = DD_blocks_to_compute[DD_block_id];

  extern __shared__ cu_cmplx_PRECISION shared_data[];

  // a part of shared_memory is dedicated to even sites, the rest to odd sites
  cu_cmplx_PRECISION *shared_data_loc = shared_data;
  cu_cmplx_PRECISION *tmp_loc;

  //the following are 'bare' values, i.e. with respect to the 0th element within a CUDA block
  // EVEN
  tmp_loc = shared_data_loc;

  _cuda_block_PRECISION_boundary_op_plus_naive(out, in, block_id, s, idx, tmp_loc, ext_dir, block);

}



__forceinline__ __device__ void _cuda_block_PRECISION_boundary_op_minus_naive(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int block_id, schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf, int ext_dir, block_struct* block){

  int dir, i, loc_ind=idx%6, spin, w, *gamma_coo, idx_in_cublock = idx%blockDim.x, index, neighbor_index, *bbl = s->block_boundary_length;
  cu_cmplx_PRECISION *gamma_val;

  cu_cmplx_PRECISION *buf1, *buf2;
  buf += (idx_in_cublock/6)*12;
  buf1 = buf;
  buf2 = buf + 6;

  //TODO: implement a dynamic way of loading s->op.D into __shared__

  //cu_config_PRECISION* D = s->op.D + (start/12)*36;
  cu_config_PRECISION* D = s->op.D;
  cu_config_PRECISION* D_pt;

  cu_cmplx_PRECISION *eta_pt, *phi_pt; //eta and phi are already shifted by 'start'

  // TODO: is there a reason for this not to be integrated with extra_dir ?
  dir = ext_dir;

  i = idx/6;
  i *= 2;
  i += bbl[ 2*ext_dir + 1 ];

  index = block[block_id].bt_on_gpu[i];
  neighbor_index = block[block_id].bt_on_gpu[i+1];

  if( block_id==619 && loc_ind==0 ){
    //printf("(block_id=%d) index=%d, neighbor_index=%d, idx=%d, i=%d, ext_dir=%d, bbl[ 2*ext_dir ]=%d\n", block_id, index, neighbor_index, idx, i, ext_dir, bbl[ 2*ext_dir ]);
  }

  D_pt = D + 36*neighbor_index + 9*ext_dir;

  phi_pt = phi + 12*neighbor_index;
  eta_pt = eta + 12*index;

  spin = (loc_ind/3)*2;
  //with this setup, gamma_val[0] gives spins 0 and 1, and gamma_val[1] spins 2 and 3
  gamma_val = s->gamma_info_vals + dir*4 + spin;
  gamma_coo = s->gamma_info_coo  + dir*4 + spin;

  // prn_T_PRECISION(...)
  buf1[ loc_ind ] = cu_cadd_PRECISION( phi_pt[ loc_ind ], cu_cmul_PRECISION( gamma_val[0], phi_pt[ 3*gamma_coo[0] + loc_ind%3 ] ) );

  __syncthreads();

  // mvmh_PRECISION(...), twice
  buf2[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
  for( w=0; w<3; w++ ){
    buf2[ loc_ind ] = cu_cadd_PRECISION( buf2[ loc_ind ], cu_cmul_PRECISION( cu_conj_PRECISION(D_pt[ loc_ind%3 + w*3 ]), buf1[ (loc_ind/3)*3 + w ] ) );
  }

  __syncthreads();

  //pbn_su3_T_PRECISION( buf2, eta_pt );
  eta_pt[ loc_ind ] = cu_csub_PRECISION( eta_pt[ loc_ind ], buf2[ loc_ind ] );
  eta_pt[ 6 + loc_ind ] = cu_csub_PRECISION( eta_pt[ 6 + loc_ind ], cu_cmul_PRECISION( gamma_val[1], buf2[ 3*gamma_coo[1] + loc_ind%3 ] ) );
}



__global__ void cuda_block_PRECISION_boundary_op_minus_naive( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                                              schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                              int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                              int num_latt_site_var, block_struct* block, int ext_dir ){

  int idx;
  idx = threadIdx.x + blockDim.x * blockIdx.x;

  //int DD_block_id, block_id, start;
  int DD_block_id, block_id;

  // not really a DD block id, but rather a linear counting of a grouping (per DD block) of CUDA threads
  DD_block_id = idx/nr_threads_per_DD_block;

  // offsetting idx to make it zero at the beginning of the threads living within a DD block
  idx = idx%nr_threads_per_DD_block;

  // this int will be the ACTUAL DD block ID, in the sense of accessing data from e.g. block_struct* block
  block_id = DD_blocks_to_compute[DD_block_id];

  extern __shared__ cu_cmplx_PRECISION shared_data[];

  // a part of shared_memory is dedicated to even sites, the rest to odd sites
  cu_cmplx_PRECISION *shared_data_loc = shared_data;
  cu_cmplx_PRECISION *tmp_loc;

  //the following are 'bare' values, i.e. with respect to the 0th element within a CUDA block
  // EVEN
  tmp_loc = shared_data_loc;

  _cuda_block_PRECISION_boundary_op_minus_naive(out, in, block_id, s, idx, tmp_loc, ext_dir, block);

}




extern "C" void cuda_block_PRECISION_boundary_op( cuda_vector_PRECISION eta, cuda_vector_PRECISION phi,
                                                  int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s, level_struct *l,
                                                  struct Thread *threading, int stream_id, cudaStream_t *streams, int color,
                                                  int* DD_blocks_to_compute_gpu, int* DD_blocks_to_compute_cpu ) {

  int dir, nr_threads, nr_threads_per_DD_block, threads_per_cublock, tot_shared_mem;

  threads_per_cublock = 96;
  tot_shared_mem = 1*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 16*sizeof(cu_cmplx_PRECISION) + 16*sizeof(int);

  for( dir=0; dir<4; dir++ ){

    //printf("s->num_boundary_sites[%d] = %d\n", dir, s->num_boundary_sites[dir*2]);

    //printf("index[619][516]=%d, neighbor_index[619][516]=%d\n", s->block[619].bt[516], s->block[619].bt[517]);

    // both directions (+ and -) are independent of each other... <<NOT AS IN THE HOPPING TERM>>
    //nr_threads = s->num_boundary_sites[dir*2] + s->num_boundary_sites[dir*2+1];
    nr_threads = s->num_boundary_sites[dir*2];
    nr_threads = nr_threads*(12/2);
    nr_threads = nr_threads*nr_DD_blocks_to_compute;
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

    cuda_block_PRECISION_boundary_op_plus_naive<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                               (eta, phi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, \
                                               l->num_lattice_site_var, (s->cu_s).block, dir);

    cuda_block_PRECISION_boundary_op_minus_naive<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                                (eta, phi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, \
                                                l->num_lattice_site_var, (s->cu_s).block, dir);

    //cuda_safe_call( cudaDeviceSynchronize() );
  }

  //printf("s->dir_length_odd[0] = %d\n", s->dir_length_odd[0]);

}



//************************************************************************************************************







































//************************************************************************************************************

__forceinline__ __device__ void _cuda_n_block_PRECISION_boundary_op_plus_naive(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int block_id, schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf, int ext_dir, block_struct* block){

  int dir, i, loc_ind=idx%6, spin, w, *gamma_coo, idx_in_cublock = idx%blockDim.x, index, neighbor_index, *bbl = s->block_boundary_length;
  cu_cmplx_PRECISION *gamma_val;

  cu_cmplx_PRECISION *buf1, *buf2;
  buf += (idx_in_cublock/6)*12;
  buf1 = buf;
  buf2 = buf + 6;

  //TODO: implement a dynamic way of loading s->op.D into __shared__

  //cu_config_PRECISION* D = s->op.D + (start/12)*36;
  cu_config_PRECISION* D = s->op.D;
  cu_config_PRECISION* D_pt;

  cu_cmplx_PRECISION *eta_pt, *phi_pt; //eta and phi are already shifted by 'start'

  // TODO: is there a reason for this not to be integrated with extra_dir ?
  dir = ext_dir;

  i = idx/6;
  i *= 2;
  i += bbl[ 2*ext_dir ];

  index = block[block_id].bt_on_gpu[i];
  neighbor_index = block[block_id].bt_on_gpu[i+1];

  if( block_id==619 && loc_ind==0 ){
    //printf("(block_id=%d) index=%d, neighbor_index=%d, idx=%d, i=%d, ext_dir=%d, bbl[ 2*ext_dir ]=%d\n", block_id, index, neighbor_index, idx, i, ext_dir, bbl[ 2*ext_dir ]);
  }

  D_pt = D + 36*index + 9*ext_dir;

  phi_pt = phi + 12*neighbor_index;
  eta_pt = eta + 12*index;

  spin = (loc_ind/3)*2;
  //with this setup, gamma_val[0] gives spins 0 and 1, and gamma_val[1] spins 2 and 3
  gamma_val = s->gamma_info_vals + dir*4 + spin;
  gamma_coo = s->gamma_info_coo  + dir*4 + spin;

  // prp_T_PRECISION(...)
  buf1[ loc_ind ] = cu_csub_PRECISION( phi_pt[ loc_ind ], cu_cmul_PRECISION( gamma_val[0], phi_pt[ 3*gamma_coo[0] + loc_ind%3 ] ) );

  __syncthreads();

  // nmvm_PRECISION(...), twice
  buf2[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
  for( w=0; w<3; w++ ){
    //buf2[ loc_ind ] = cu_cadd_PRECISION( buf2[ loc_ind ], cu_cmul_PRECISION( D_pt[ (loc_ind*3)%9 + w ], buf1[ (loc_ind/3)*3 + w ] ) );
    buf2[ loc_ind ] = cu_csub_PRECISION( buf2[ loc_ind ], cu_cmul_PRECISION( D_pt[ (loc_ind*3)%9 + w ], buf1[ (loc_ind/3)*3 + w ] ) );
  }

  __syncthreads();

  //pbp_su3_T_PRECISION( buf2, eta_pt );
  eta_pt[ loc_ind ] = cu_csub_PRECISION( eta_pt[ loc_ind ], buf2[ loc_ind ] );
  eta_pt[ 6 + loc_ind ] = cu_cadd_PRECISION( eta_pt[ 6 + loc_ind ], cu_cmul_PRECISION( gamma_val[1], buf2[ 3*gamma_coo[1] + loc_ind%3 ] ) );
}







__global__ void cuda_n_block_PRECISION_boundary_op_plus_naive( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                                               schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                               int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                               int num_latt_site_var, block_struct* block, int ext_dir ){

  int idx;
  idx = threadIdx.x + blockDim.x * blockIdx.x;

  //int DD_block_id, block_id, start;
  int DD_block_id, block_id;

  // not really a DD block id, but rather a linear counting of a grouping (per DD block) of CUDA threads
  DD_block_id = idx/nr_threads_per_DD_block;

  // offsetting idx to make it zero at the beginning of the threads living within a DD block
  idx = idx%nr_threads_per_DD_block;

  // this int will be the ACTUAL DD block ID, in the sense of accessing data from e.g. block_struct* block
  block_id = DD_blocks_to_compute[DD_block_id];

  extern __shared__ cu_cmplx_PRECISION shared_data[];

  // a part of shared_memory is dedicated to even sites, the rest to odd sites
  cu_cmplx_PRECISION *shared_data_loc = shared_data;
  cu_cmplx_PRECISION *tmp_loc;

  //the following are 'bare' values, i.e. with respect to the 0th element within a CUDA block
  // EVEN
  tmp_loc = shared_data_loc;

  _cuda_n_block_PRECISION_boundary_op_plus_naive(out, in, block_id, s, idx, tmp_loc, ext_dir, block);

}



__forceinline__ __device__ void _cuda_n_block_PRECISION_boundary_op_minus_naive(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int block_id, schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf, int ext_dir, block_struct* block){

  int dir, i, loc_ind=idx%6, spin, w, *gamma_coo, idx_in_cublock = idx%blockDim.x, index, neighbor_index, *bbl = s->block_boundary_length;
  cu_cmplx_PRECISION *gamma_val;

  cu_cmplx_PRECISION *buf1, *buf2;
  buf += (idx_in_cublock/6)*12;
  buf1 = buf;
  buf2 = buf + 6;

  //TODO: implement a dynamic way of loading s->op.D into __shared__

  //cu_config_PRECISION* D = s->op.D + (start/12)*36;
  cu_config_PRECISION* D = s->op.D;
  cu_config_PRECISION* D_pt;

  cu_cmplx_PRECISION *eta_pt, *phi_pt; //eta and phi are already shifted by 'start'

  // TODO: is there a reason for this not to be integrated with extra_dir ?
  dir = ext_dir;

  i = idx/6;
  i *= 2;
  i += bbl[ 2*ext_dir + 1 ];

  index = block[block_id].bt_on_gpu[i];
  neighbor_index = block[block_id].bt_on_gpu[i+1];

  if( block_id==619 && loc_ind==0 ){
    //printf("(block_id=%d) index=%d, neighbor_index=%d, idx=%d, i=%d, ext_dir=%d, bbl[ 2*ext_dir ]=%d\n", block_id, index, neighbor_index, idx, i, ext_dir, bbl[ 2*ext_dir ]);
  }

  D_pt = D + 36*neighbor_index + 9*ext_dir;

  phi_pt = phi + 12*neighbor_index;
  eta_pt = eta + 12*index;

  spin = (loc_ind/3)*2;
  //with this setup, gamma_val[0] gives spins 0 and 1, and gamma_val[1] spins 2 and 3
  gamma_val = s->gamma_info_vals + dir*4 + spin;
  gamma_coo = s->gamma_info_coo  + dir*4 + spin;

  // prn_T_PRECISION(...)
  buf1[ loc_ind ] = cu_cadd_PRECISION( phi_pt[ loc_ind ], cu_cmul_PRECISION( gamma_val[0], phi_pt[ 3*gamma_coo[0] + loc_ind%3 ] ) );

  __syncthreads();

  // nmvmh_PRECISION(...), twice
  buf2[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
  for( w=0; w<3; w++ ){
    //buf2[ loc_ind ] = cu_cadd_PRECISION( buf2[ loc_ind ], cu_cmul_PRECISION( cu_conj_PRECISION(D_pt[ loc_ind%3 + w*3 ]), buf1[ (loc_ind/3)*3 + w ] ) );
    buf2[ loc_ind ] = cu_csub_PRECISION( buf2[ loc_ind ], cu_cmul_PRECISION( cu_conj_PRECISION(D_pt[ loc_ind%3 + w*3 ]), buf1[ (loc_ind/3)*3 + w ] ) );
  }

  __syncthreads();

  //pbn_su3_T_PRECISION( buf2, eta_pt );
  eta_pt[ loc_ind ] = cu_csub_PRECISION( eta_pt[ loc_ind ], buf2[ loc_ind ] );
  eta_pt[ 6 + loc_ind ] = cu_csub_PRECISION( eta_pt[ 6 + loc_ind ], cu_cmul_PRECISION( gamma_val[1], buf2[ 3*gamma_coo[1] + loc_ind%3 ] ) );
}



__global__ void cuda_n_block_PRECISION_boundary_op_minus_naive( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                                int num_latt_site_var, block_struct* block, int ext_dir ){

  int idx;
  idx = threadIdx.x + blockDim.x * blockIdx.x;

  //int DD_block_id, block_id, start;
  int DD_block_id, block_id;

  // not really a DD block id, but rather a linear counting of a grouping (per DD block) of CUDA threads
  DD_block_id = idx/nr_threads_per_DD_block;

  // offsetting idx to make it zero at the beginning of the threads living within a DD block
  idx = idx%nr_threads_per_DD_block;

  // this int will be the ACTUAL DD block ID, in the sense of accessing data from e.g. block_struct* block
  block_id = DD_blocks_to_compute[DD_block_id];

  extern __shared__ cu_cmplx_PRECISION shared_data[];

  // a part of shared_memory is dedicated to even sites, the rest to odd sites
  cu_cmplx_PRECISION *shared_data_loc = shared_data;
  cu_cmplx_PRECISION *tmp_loc;

  //the following are 'bare' values, i.e. with respect to the 0th element within a CUDA block
  // EVEN
  tmp_loc = shared_data_loc;

  _cuda_n_block_PRECISION_boundary_op_minus_naive(out, in, block_id, s, idx, tmp_loc, ext_dir, block);

}




extern "C" void cuda_n_block_PRECISION_boundary_op( cuda_vector_PRECISION eta, cuda_vector_PRECISION phi,
                                                    int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s, level_struct *l,
                                                    struct Thread *threading, int stream_id, cudaStream_t *streams, int color,
                                                    int* DD_blocks_to_compute_gpu, int* DD_blocks_to_compute_cpu ) {

  int dir, nr_threads, nr_threads_per_DD_block, threads_per_cublock, tot_shared_mem;

  threads_per_cublock = 96;
  tot_shared_mem = 1*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 16*sizeof(cu_cmplx_PRECISION) + 16*sizeof(int);

  for( dir=0; dir<4; dir++ ){

    //printf("s->num_boundary_sites[%d] = %d\n", dir, s->num_boundary_sites[dir*2]);

    //printf("index[619][516]=%d, neighbor_index[619][516]=%d\n", s->block[619].bt[516], s->block[619].bt[517]);

    // both directions (+ and -) are independent of each other... <<NOT AS IN THE HOPPING TERM>>
    //nr_threads = s->num_boundary_sites[dir*2] + s->num_boundary_sites[dir*2+1];
    nr_threads = s->num_boundary_sites[dir*2];
    nr_threads = nr_threads*(12/2);
    nr_threads = nr_threads*nr_DD_blocks_to_compute;
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

    cuda_n_block_PRECISION_boundary_op_plus_naive<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                                 (eta, phi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, \
                                                 l->num_lattice_site_var, (s->cu_s).block, dir);

    cuda_n_block_PRECISION_boundary_op_minus_naive<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                                  (eta, phi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, \
                                                  l->num_lattice_site_var, (s->cu_s).block, dir);

    //cuda_safe_call( cudaDeviceSynchronize() );
  }

  //printf("s->dir_length_odd[0] = %d\n", s->dir_length_odd[0]);

}



//************************************************************************************************************












__forceinline__ __device__ void _cuda_block_d_plus_clover_PRECISION_6threads_naive(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf, int ext_dir){

  //if amount==0 then even sites, if amount==1 then odd sites, else no oddeven split

  //int dir; // dir '=' {0,1,2,3} = {T,Z,Y,X}
  //int dir, a1, a2, n1, n2, k=0, j=0, i=0, **index = s->oe_index, *ind, *neighbor = s->op.neighbor_table, loc_ind=idx%6, spin, w, *gamma_coo, idx_in_cublock = idx%blockDim.x;
  //int dir, a1, n1, a2, n2, k=0, j=0, i=0, **index = s->oe_index, *ind, *neighbor = s->op.neighbor_table, loc_ind=idx%6, spin, w, *gamma_coo, idx_in_cublock = idx%blockDim.x;
  int dir, k=0, j=0, i=0, **index = s->index, *ind, *neighbor = s->op.neighbor_table, loc_ind=idx%6, spin, w, *gamma_coo, idx_in_cublock = idx%blockDim.x;
  cu_cmplx_PRECISION *gamma_val;

  cu_cmplx_PRECISION *buf1, *buf2, *buf3, *buf4;
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

  //lphi = phi + 12*k;
  //leta = eta + 12*j;

  // already added <start> to the original input spinors
  lphi = phi;
  leta = eta;

  spin = (loc_ind/3)*2;
  //with this setup, gamma_val[0] gives spins 0 and 1, and gamma_val[1] spins 2 and 3
  gamma_val = s->gamma_info_vals + dir*4 + spin;
  gamma_coo = s->gamma_info_coo  + dir*4 + spin;

  // prn_T_PRECISION(...)
  buf1[ loc_ind ] = cu_cadd_PRECISION( (lphi + 12*k)[ loc_ind ], cu_cmul_PRECISION( gamma_val[0], (lphi + 12*k)[ 3*gamma_coo[0] + loc_ind%3 ] ) );

  // prp_T_PRECISION(...)
  buf2[ loc_ind ] = cu_csub_PRECISION( (lphi + 12*j)[ loc_ind ], cu_cmul_PRECISION( gamma_val[0], (lphi + 12*j)[ 3*gamma_coo[0] + loc_ind%3 ] ) );

  __syncthreads();

  // mvmh_PRECISION(...), twice
  buf3[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
  for( w=0; w<3; w++ ){
    buf3[ loc_ind ] = cu_cadd_PRECISION( buf3[ loc_ind ], cu_cmul_PRECISION( cu_conj_PRECISION(D_pt[ loc_ind%3 + w*3 ]), buf1[ (loc_ind/3)*3 + w ] ) );
  }

  // mvm_PRECISION(...), twice
  buf4[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
  for( w=0; w<3; w++ ){
    buf4[ loc_ind ] = cu_cadd_PRECISION( buf4[ loc_ind ], cu_cmul_PRECISION( D_pt[ (loc_ind*3)%9 + w ], buf2[ (loc_ind/3)*3 + w ] ) );
  }

  // mvmh_PRECISION(...), twice
  //buf2[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
  //for( w=0; w<3; w++ ){
  //  buf2[ loc_ind ] = cu_cadd_PRECISION( buf2[ loc_ind ], cu_cmul_PRECISION( cu_conj_PRECISION(D_pt[ loc_ind%3 + w*3 ]), buf1[ (loc_ind/3)*3 + w ] ) );
  //}

  __syncthreads();

  // pbn_su3_T_PRECISION(...)
  (leta + 12*j)[ loc_ind ] = cu_csub_PRECISION( (leta + 12*j)[ loc_ind ], buf3[ loc_ind ] );
  (leta + 12*j)[ 6 + loc_ind ] = cu_csub_PRECISION( (leta + 12*j)[ 6 + loc_ind ], cu_cmul_PRECISION( gamma_val[1], buf3[ 3*gamma_coo[1] + loc_ind%3 ] ) );

  // pbp_su3_T_PRECISION(...);
  (leta + 12*k)[ loc_ind ] = cu_csub_PRECISION( (leta + 12*k)[ loc_ind ], buf4[ loc_ind ] );
  (leta + 12*k)[ 6 + loc_ind ] = cu_cadd_PRECISION( (leta + 12*k)[ 6 + loc_ind ], cu_cmul_PRECISION( gamma_val[1], buf4[ 3*gamma_coo[1] + loc_ind%3 ] ) );

}




// gamma_val and gamma_coo are both loaded into shared memory for this kernel
__global__ void cuda_block_d_plus_clover_PRECISION_6threads_naive( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                                                   schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                   int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                                   int num_latt_site_var, block_struct* block, int ext_dir ){

  int idx;
  idx = threadIdx.x + blockDim.x * blockIdx.x;

  int DD_block_id, block_id, start;

  //int nr_block_even_sites;
  //nr_block_even_sites = s->num_block_even_sites;

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
  cu_cmplx_PRECISION *shared_data_loc = shared_data;
  cu_cmplx_PRECISION *tmp_loc;

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














//********************************************************************************************************************************





















__forceinline__ __device__ void _cuda_site_clover_PRECISION( cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, \
                                                             schwarz_PRECISION_struct_on_gpu *s, int idx, cu_config_PRECISION *op_clov, int csw ){

  // FIXME: extend code to include case csw==0

  int local_idx = idx%6;
  //int local_idx;
  // this offset is per site and within each CUDA block
  int site_offset = (threadIdx.x/6)*12;

  cu_cmplx_PRECISION* eta_site = eta + site_offset;
  cu_cmplx_PRECISION* phi_site = phi + site_offset;
  cu_config_PRECISION* op_clov_site = op_clov + (site_offset/12)*42;

  int i, k, matrx_indx;

  eta_site[ local_idx ] = make_cu_cmplx_PRECISION(0.0,0.0);
  eta_site[ 6 + local_idx ] = make_cu_cmplx_PRECISION(0.0,0.0);

  for( k=0; k<2; k++ ){

    for( i=0; i<6; i++ ){

      if( (i+k*6)>local_idx ){

        matrx_indx = 12 +15*k + (14 - ( (5-local_idx%6-1)*(5-local_idx%6-1 + 1)/2 + (5-i) ));

        eta_site[local_idx] = cu_cadd_PRECISION( eta_site[local_idx], cu_cmul_PRECISION( op_clov_site[matrx_indx], phi_site[i + k*6] ) ) ;

      }
      else if( (i+k*6)<local_idx ){
      //else if( i<local_idx ){

        matrx_indx = 12 +15*k + (14 - ( (5-i-1)*(5-i-1 + 1)/2 + (5-local_idx%6) ));

        eta_site[local_idx] = cu_cadd_PRECISION( eta_site[local_idx], cu_cmul_PRECISION( cu_conj_PRECISION( op_clov_site[matrx_indx] ), phi_site[i + k*6] ) ) ;

      }
      else{ // i==local_idx

        //matrx_indx = 12 +15*k + (14 - ( (5-i-1)*(5-i-1 + 1)/2 + (5-local_idx%6) ));
        //eta_site[local_idx] = cu_cadd_PRECISION( eta_site[local_idx], cu_cmul_PRECISION( cu_conj_PRECISION( op_clov_site[matrx_indx] ), phi_site[i + k*6] ) ) ;
        //if(k==0) 
        eta_site[local_idx] = cu_cadd_PRECISION( eta_site[local_idx], cu_cmul_PRECISION( op_clov_site[local_idx], phi_site[local_idx] ) ) ;

      }

    }

    local_idx += 6;

  }

}




__global__ void cuda_site_clover_PRECISION( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                            schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                            int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                            int num_latt_site_var, block_struct* block ){

  // IMPORTANT: each component of each site is independent, and so, 12 threads could be used per
  //            lattice site, but here we use 6 due to loading s->op.clover into shared memory
  //            collaboratively by all the threads

  int i, idx, DD_block_id, block_id, cublocks_per_DD_block, cu_block_ID, size_D_clov, start, nr_D_dumps;

  //return;

  //int nr_block_even_sites, nr_block_odd_sites;
  //nr_block_even_sites = s->num_block_even_sites;
  //nr_block_odd_sites = s->num_block_odd_sites;

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
  size_D_clov = (csw!=0) ? 42 : 12;

  // this is the DD-block start of the spinors (phi, r, latest_iter and temporary ones)
  start = block[block_id].start * num_latt_site_var;

  out += start;
  in += start;

  cu_config_PRECISION *op = s->op.clover_gpustorg;
  // FIXME: instead of 12, use num_latt_site_var
  op += (start/12)*size_D_clov;

  extern __shared__ cu_cmplx_PRECISION shared_data[];

  //TODO: can we trim the use of shared memory ... ?

  cu_cmplx_PRECISION *shared_data_loc = shared_data;
  cu_cmplx_PRECISION *in_o, *out_o;
  cu_config_PRECISION *clov_o;

  in_o = shared_data_loc + 0*(2*blockDim.x);
  out_o = shared_data_loc + 1*(2*blockDim.x);

  clov_o = (cu_cmplx_PRECISION*)((cu_cmplx_PRECISION*)shared_data_loc + 2*(2*blockDim.x));

  // TODO: change this partial summary to the new shared memory required
  // partial summary so far:
  //    ** phi_b_e has a memory reservation of size 2*6*(blockDim.x/6)
  //    ** same for r_b_e, tmp_2_e, tmp_3_e
  //    ** clov_vect_b_e has a memory reservation of size size_D_oeclov*(blockDim.x/6)
  //    ** equivalently, we can say the same about the *_o variables

  // odd
  //if(idx < 2*nr_block_odd_sites){
    for(i=0; i<2; i++){
      in_o[blockDim.x*i + threadIdx.x] = ( in + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
    }

    for(i=0; i<2; i++){
      //out_o[blockDim.x*i + threadIdx.x] = ( out + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
    }

    // the factor of 12 comes from (the factor of 16 comes from the nr of lattice sites per CUDA block: 96/6 = 16): 72*16=1152 ----> 1152/96=12. This implies 12 dumps of data into shared_memory
    nr_D_dumps = (blockDim.x/6)*size_D_clov/blockDim.x; // = 7 always
    //nr_D_dumps /= nr_D_dumps;
    for(i=0; i<nr_D_dumps; i++){
      clov_o[blockDim.x*i + threadIdx.x] = ( op + cu_block_ID*blockDim.x*nr_D_dumps + blockDim.x*i + threadIdx.x )[0];
    }
  //}

  __syncthreads();

  i = idx/6;

  //if( block_id==621 && i==77 ){
  //if( i==77 ){

  //  int site_offset = threadIdx.x/6 * 42;
  //  printf("(block=%d)(site=77) clov[%d] = %f+i%f\n", block_id, idx%6, cu_creal_PRECISION(clov_o[site_offset + idx%6]), cu_cimag_PRECISION(clov_o[site_offset + idx%6]));
  //  printf("(block=%d)(site=77) eta[%d] = %f+i%f\n", block_id, idx%6, cu_creal_PRECISION(in_o[site_offset/42*12 + idx%6]), cu_cimag_PRECISION(in_o[site_offset/42*12 + idx%6]));
  //  printf("(block=%d)(site=77) (bef)out[%d] = %f+i%f\n", block_id, idx%6, cu_creal_PRECISION(out_o[site_offset/42*12 + idx%6]), cu_cimag_PRECISION(out_o[site_offset/42*12 + idx%6]));

  //  printf("(block=%d)(site=77) clov[%d] = %f+i%f\n", block_id, idx%6+6, cu_creal_PRECISION(clov_o[site_offset + idx%6+6]), cu_cimag_PRECISION(clov_o[site_offset + idx%6+6]));
  //  printf("(block=%d)(site=77) eta[%d] = %f+i%f\n", block_id, idx%6+6, cu_creal_PRECISION(in_o[site_offset/42*12 + idx%6+6]), cu_cimag_PRECISION(in_o[site_offset/42*12 + idx%6+6]));
  //  printf("(block=%d)(site=77) (bef)out[%d] = %f+i%f\n", block_id, idx%6+6, cu_creal_PRECISION(out_o[site_offset/42*12 + idx%6+6]), cu_cimag_PRECISION(out_o[site_offset/42*12 + idx%6+6]));

  //}

  _cuda_site_clover_PRECISION(out_o, in_o, start, s, idx, clov_o, csw);

  __syncthreads();

  // update tmp2 and tmp3
  // odd
  //if(idx < 2*nr_block_odd_sites){
    for(i=0; i<2; i++){
      ( out + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = out_o[blockDim.x*i + threadIdx.x];
    }
  //}

  __syncthreads();

  i = idx/6;

  //if( block_id==621 && i==77 ){
  //if( i==77 ){

    //int site_offset = threadIdx.x/6 * 42;
    //printf("(block=%d)(site=77) (aft)out[%d] = %f+i%f\n", block_id, idx%6, cu_creal_PRECISION(out_o[site_offset/42*12 + idx%6]), cu_cimag_PRECISION(out_o[site_offset/42*12 + idx%6]));
    ////printf("(block=%d)(site=77) clov[%d] = %f+i%f\n", block_id, idx%6, cu_creal_PRECISION(clov_o[site_offset + idx%6]), cu_cimag_PRECISION(clov_o[site_offset + idx%6]));
    //printf("(block=%d)(site=77) (aft)out[%d] = %f+i%f\n", block_id, idx%6+6, cu_creal_PRECISION(out_o[site_offset/42*12 + idx%6+6]), cu_cimag_PRECISION(out_o[site_offset/42*12 + idx%6+6]));

  //}

}




















extern "C" void cuda_block_d_plus_clover_PRECISION( cuda_vector_PRECISION eta, cuda_vector_PRECISION phi,
                                                    int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s, level_struct *l,
                                                    struct Thread *threading, int stream_id, cudaStream_t *streams, int color,
                                                    int* DD_blocks_to_compute_gpu, int* DD_blocks_to_compute_cpu ){

  //int *length = s->dir_length;
  //for( int dir=0; dir<4; dir++ ){
  //  printf("length[%d]=%d\n", dir, length[dir]);
  //}



  //int threads_per_cublock, nr_threads, size_D_oeclov, nr_threads_per_DD_block, dir;
  int threads_per_cublock, nr_threads, nr_threads_per_DD_block, dir, n = s->num_block_sites;
  size_t tot_shared_mem;

  //size_D_oeclov = (g.csw!=0) ? 42 : 12;

  // clover term
  if ( g.csw == 0.0 ) {
    //clover_PRECISION( leta, lphi, clover, 12*n, l, threading ); 
    //TODO
  } else {
    //for ( i=0; i<n; i++ ) {
    //  //site_clover_PRECISION( leta+12*i, lphi+12*i, clover+42*i );
    //}

    //cuda_site_clover_PRECISION<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>>
    //                          ( leta+12*i, lphi+12*i, clover+42*i );

    threads_per_cublock = 96;

    // diag_oo inv
    nr_threads = n; // nr sites per DD block
    nr_threads = nr_threads*(12/2); // threads per site
    nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
    //tot_shared_mem = 2*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 1*size_D_oeclov*(threads_per_cublock/6)*sizeof(cu_config_PRECISION);
    tot_shared_mem = 2*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 1*42*(threads_per_cublock/6)*sizeof(cu_config_PRECISION);

    cuda_site_clover_PRECISION<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                              (eta, phi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block);

  }

  //return;

  threads_per_cublock = 96;
  tot_shared_mem = 1*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 16*sizeof(cu_cmplx_PRECISION) + 16*sizeof(int);
  for( dir=0; dir<4; dir++ ){

    //nr_threads = (sites_to_solve==_EVEN_SITES)?s->dir_length_even[dir]:s->dir_length_odd[dir];
    //nr_threads = s->dir_length_even[dir];
    nr_threads = s->dir_length[dir];
    nr_threads = nr_threads*(12/2);
    nr_threads = nr_threads*nr_DD_blocks_to_compute;
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

    //cuda_block_d_plus_clover_PRECISION_6threads_naive<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
    //                                                     (eta, phi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, \
    //                                                     l->num_lattice_site_var, (s->cu_s).block, dir);



    cuda_block_hopping_term_PRECISION_plus_6threads_naive<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                                           (eta, phi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, \
                                                           l->num_lattice_site_var, (s->cu_s).block, dir, _FULL_SYSTEM);
    cuda_block_hopping_term_PRECISION_minus_6threads_naive<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                                            (eta, phi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, \
                                                            l->num_lattice_site_var, (s->cu_s).block, dir, _FULL_SYSTEM);




  }


}
































extern "C" void cuda_vector_PRECISION_minus( cuda_vector_PRECISION out, cuda_vector_PRECISION in1, cuda_vector_PRECISION in2,
                                             int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s, level_struct *l,
                                             struct Thread *threading, int stream_id, cudaStream_t *streams, int color,
                                             int* DD_blocks_to_compute_gpu, int* DD_blocks_to_compute_cpu ) {

  int nr_threads, nr_threads_per_DD_block, threads_per_cublock;

  nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
  nr_threads = nr_threads*(12/2); // threads per site
  nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
  nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

  threads_per_cublock = 96;

  cuda_block_oe_vector_PRECISION_minus_6threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                                      (out, in1, in2, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, \
                                                      l->num_lattice_site_var, (s->cu_s).block, _FULL_SYSTEM);

}









#endif
