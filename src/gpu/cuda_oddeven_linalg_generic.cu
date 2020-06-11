#include <mpi.h>

extern "C"{

  #define IMPORT_FROM_EXTERN_C
  #include "main.h"
  #undef IMPORT_FROM_EXTERN_C

}

#ifdef CUDA_OPT

__global__ void cuda_block_oe_vector_PRECISION_copy_6threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                                                  schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                  double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
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


__global__ void cuda_block_oe_vector_PRECISION_copy_12threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                                                  schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                  double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                                  int num_latt_site_var, block_struct* block, int sites_to_copy ){

  int idx, DD_block_id, block_id, cublocks_per_DD_block, cu_block_ID, start;

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
    if(idx < 12*nr_block_even_sites){
      ( out + cu_block_ID*blockDim.x + threadIdx.x )[0] = ( in + cu_block_ID*blockDim.x + threadIdx.x )[0];
    }
  }
  else if( sites_to_copy==_ODD_SITES ){
    // odd
    if(idx < 12*nr_block_odd_sites){
      ( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0] = ( in + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0];
    }
  }
  else if( sites_to_copy==_FULL_SYSTEM ){
    // even
    if(idx < 12*nr_block_even_sites){
      ( out + cu_block_ID*blockDim.x + threadIdx.x )[0] = ( in + cu_block_ID*blockDim.x + threadIdx.x )[0];
    }
    // odd
    if(idx < 12*nr_block_odd_sites){
      ( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0] = ( in + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0];
    }
  }

}


__global__ void cuda_block_oe_vector_PRECISION_define_12threads_opt( cu_cmplx_PRECISION* spinor,\
                                                                    schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                    double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                                    int num_latt_site_var, block_struct* block, int sites_to_define,
                                                                    cu_cmplx_PRECISION val_to_assign ){

  int idx, DD_block_id, block_id, cublocks_per_DD_block, cu_block_ID, start;

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
    if(idx < 12*nr_block_even_sites){
      //for(i=0; i<2; i++){
      ( spinor + cu_block_ID*blockDim.x + threadIdx.x )[0] = val_to_assign;
      //}
    }
  }
  else if( sites_to_define==_ODD_SITES ){
    // odd
    if(idx < 12*nr_block_odd_sites){
      //for(i=0; i<2; i++){
      ( spinor + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0] = val_to_assign;
      //}
    }
  }
  else if( sites_to_define==_FULL_SYSTEM ){
    // even
    if(idx < 12*nr_block_even_sites){
      //for(i=0; i<2; i++){
      ( spinor + cu_block_ID*blockDim.x + threadIdx.x )[0] = val_to_assign;
      //}
    }
    // odd
    if(idx < 12*nr_block_odd_sites){
      //for(i=0; i<2; i++){
      ( spinor + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0] = val_to_assign;
      //}
    }
  }

}


__global__ void cuda_block_oe_vector_PRECISION_define_6threads_opt( cu_cmplx_PRECISION* spinor,\
                                                                    schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                    double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
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




__global__ void cuda_block_oe_vector_PRECISION_plus_12threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in1, cu_cmplx_PRECISION* in2, \
                                                                   schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                   double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                                   int num_latt_site_var, block_struct* block, int sites_to_add ){

  int idx, DD_block_id, block_id, cublocks_per_DD_block, cu_block_ID, start;

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

  if( sites_to_add==_EVEN_SITES ){
    // even
    if(idx < 12*nr_block_even_sites){

      ( out + cu_block_ID*blockDim.x + threadIdx.x )[0] = cu_cadd_PRECISION( ( in1 + cu_block_ID*blockDim.x + threadIdx.x )[0], \
                                                                             ( in2 + cu_block_ID*blockDim.x + threadIdx.x )[0] );

    }
  }
  else if( sites_to_add==_ODD_SITES ){
    // odd
    if(idx < 12*nr_block_odd_sites){

      ( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0] = \
                                             cu_cadd_PRECISION( ( in1 + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0], \
                                                                ( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0] );

    }
  }
  else if( sites_to_add==_FULL_SYSTEM ){
    // even
    if(idx < 12*nr_block_even_sites){

      ( out + cu_block_ID*blockDim.x + threadIdx.x )[0] = cu_cadd_PRECISION( ( in1 + cu_block_ID*blockDim.x + threadIdx.x )[0], \
                                                                             ( in2 + cu_block_ID*blockDim.x + threadIdx.x )[0] );

    }
    // odd
    if(idx < 12*nr_block_odd_sites){

      ( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0] = \
                                             cu_cadd_PRECISION( ( in1 + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0], \
                                                                ( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0] );

    }
  }

}


__global__ void cuda_block_oe_vector_PRECISION_plus_6threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in1, cu_cmplx_PRECISION* in2, \
                                                                  schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                  double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
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
                                                                   double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
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
                                                 double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
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

  if( idx==0 ){
    s->alphas[block_id] = alpha;
  }

}


__global__ void cuda_block_oe_vector_PRECISION_saxpy_6threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in1, cu_cmplx_PRECISION* in2, cu_cmplx_PRECISION alpha, \
                                                                   schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                   double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
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


__global__ void cuda_block_oe_vector_PRECISION_saxpy_12threads_opt_onchip( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in1, cu_cmplx_PRECISION* in2, \
                                                                          PRECISION prefctr_alpha, cu_cmplx_PRECISION *alphas, \
                                                                          schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                          double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                                          int num_latt_site_var, block_struct* block, int sites_to_add ){

  int idx, DD_block_id, block_id, cublocks_per_DD_block, cu_block_ID, start;

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

  cu_cmplx_PRECISION alpha = alphas[block_id];
  // add pre-factor multiplication
  alpha.x *= prefctr_alpha;
  alpha.y *= prefctr_alpha;
  if( sites_to_add==_EVEN_SITES ){
    // even
    if(idx < 12*nr_block_even_sites){

      buf_real = cu_creal_PRECISION(( in1 + cu_block_ID*blockDim.x + threadIdx.x )[0]) + \
                 cu_creal_PRECISION(alpha) * cu_creal_PRECISION(( in2 + cu_block_ID*blockDim.x + threadIdx.x )[0]) - \
                 cu_cimag_PRECISION(alpha) * cu_cimag_PRECISION(( in2 + cu_block_ID*blockDim.x + threadIdx.x )[0]);
      buf_imag = cu_cimag_PRECISION(( in1 + cu_block_ID*blockDim.x + threadIdx.x )[0]) + \
                 cu_creal_PRECISION(alpha) * cu_cimag_PRECISION(( in2 + cu_block_ID*blockDim.x + threadIdx.x )[0]) + \
                 cu_cimag_PRECISION(alpha) * cu_creal_PRECISION(( in2 + cu_block_ID*blockDim.x + threadIdx.x )[0]);
      ( out + cu_block_ID*blockDim.x + threadIdx.x )[0] = make_cu_cmplx_PRECISION( buf_real, buf_imag );

    }
  }
  else if( sites_to_add==_ODD_SITES ){
    // odd
    if(idx < 12*nr_block_odd_sites){

      buf_real = cu_creal_PRECISION(( in1 + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0]) + \
                 cu_creal_PRECISION(alpha) * cu_creal_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0]) - \
                 cu_cimag_PRECISION(alpha) * cu_cimag_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0]);
      buf_imag = cu_cimag_PRECISION(( in1 + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0]) + \
                 cu_creal_PRECISION(alpha) * cu_cimag_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0]) + \
                 cu_cimag_PRECISION(alpha) * cu_creal_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0]);

      ( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0] = make_cu_cmplx_PRECISION( buf_real, buf_imag );

    }
  }
  else if( sites_to_add==_FULL_SYSTEM ){
    // even
    if(idx < 12*nr_block_even_sites){

      buf_real = cu_creal_PRECISION(( in1 + cu_block_ID*blockDim.x + threadIdx.x )[0]) + \
                 cu_creal_PRECISION(alpha) * cu_creal_PRECISION(( in2 + cu_block_ID*blockDim.x + threadIdx.x )[0]) - \
                 cu_cimag_PRECISION(alpha) * cu_cimag_PRECISION(( in2 + cu_block_ID*blockDim.x + threadIdx.x )[0]);
      buf_imag = cu_cimag_PRECISION(( in1 + cu_block_ID*blockDim.x + threadIdx.x )[0]) + \
                 cu_creal_PRECISION(alpha) * cu_cimag_PRECISION(( in2 + cu_block_ID*blockDim.x + threadIdx.x )[0]) + \
                 cu_cimag_PRECISION(alpha) * cu_creal_PRECISION(( in2 + cu_block_ID*blockDim.x + threadIdx.x )[0]);

      ( out + cu_block_ID*blockDim.x + threadIdx.x )[0] = make_cu_cmplx_PRECISION( buf_real, buf_imag );

    }
    // odd
    if(idx < 12*nr_block_odd_sites){

      buf_real = cu_creal_PRECISION(( in1 + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0]) + \
                 cu_creal_PRECISION(alpha) * cu_creal_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0]) - \
                 cu_cimag_PRECISION(alpha) * cu_cimag_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0]);
      buf_imag = cu_cimag_PRECISION(( in1 + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0]) + \
                 cu_creal_PRECISION(alpha) * cu_cimag_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0]) + \
                 cu_cimag_PRECISION(alpha) * cu_creal_PRECISION(( in2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0]);

      ( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0] = make_cu_cmplx_PRECISION( buf_real, buf_imag );

    }
  }

}


__global__ void cuda_block_oe_vector_PRECISION_saxpy_6threads_opt_onchip( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in1, cu_cmplx_PRECISION* in2, PRECISION prefctr_alpha, cu_cmplx_PRECISION *alphas, \
                                                                          schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                          double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
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

  cu_cmplx_PRECISION alpha = alphas[block_id];
  // add pre-factor multiplication
  alpha.x *= prefctr_alpha;
  alpha.y *= prefctr_alpha;

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


extern "C" void cuda_block_vector_PRECISION_minus( cuda_vector_PRECISION out, cuda_vector_PRECISION in1, cuda_vector_PRECISION in2,
                                                   int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s, level_struct *l,
                                                   struct Thread *threading, int stream_id, cudaStream_t *streams, int color,
                                                   int* DD_blocks_to_compute_gpu, int* DD_blocks_to_compute_cpu ) {

  if( nr_DD_blocks_to_compute==0 ){ return; }

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


// TODO: generalize to receive a CUDA stream as parameter

__global__ void _cuda_vector_PRECISION_define( cuda_vector_PRECISION phi, cu_cmplx_PRECISION scalar ){

  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  phi[idx] = scalar;

}


extern "C" void cuda_vector_PRECISION_define( cuda_vector_PRECISION phi, cu_cmplx_PRECISION scalar, int start,
                                              int length, level_struct *l, int sync_type, int stream_id, cudaStream_t *streams ){

  int nr_threads = length;
  int threads_per_cublock = 32;

  _cuda_vector_PRECISION_define<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>>
                               ( phi+start, scalar );

  if( sync_type == _CUDA_SYNC ){
    cuda_safe_call( cudaDeviceSynchronize() );
  }

}


// TODO: generalize to receive a CUDA stream as parameter
__global__ void _cuda_vector_PRECISION_scale( cuda_vector_PRECISION phi1, cuda_vector_PRECISION phi2, cu_cmplx_PRECISION scalar ){

  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  //phi[idx] = cu_cmul_PRECISION( phi[idx], scalar );
  phi1[idx] = cu_cmul_PRECISION( phi2[idx], scalar );

}


extern "C" void cuda_vector_PRECISION_scale( cuda_vector_PRECISION phi1, cuda_vector_PRECISION phi2, cu_cmplx_PRECISION scalar, int start,
                                             int length, level_struct *l, int sync_type, int stream_id, cudaStream_t *streams ){

  int nr_threads = length;
  int threads_per_cublock = 32;

  _cuda_vector_PRECISION_scale<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>>
                               ( phi1+start, phi2+start, scalar );

  if( sync_type == _CUDA_SYNC ){
    cuda_safe_call( cudaDeviceSynchronize() );
  }

}


__global__ void cuda_block_oe_vector_PRECISION_scale_6threads_opt( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, cu_cmplx_PRECISION scalar, \
                                                                   schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                   double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
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
  in += start;
  //in2 += start;

  int nr_block_even_sites, nr_block_odd_sites;
  nr_block_even_sites = s->num_block_even_sites;
  nr_block_odd_sites = s->num_block_odd_sites;

  // TODO: change the following additions to use cu_csub_PRECISION(...)

  if( sites_to_add==_EVEN_SITES ){
    // even
    if(idx < 6*nr_block_even_sites){
      for(i=0; i<2; i++){

        ( out + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = cu_cmul_PRECISION( ( in + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0], scalar );

      }
    }
  }
  else if( sites_to_add==_ODD_SITES ){
    // odd
    if(idx < 6*nr_block_odd_sites){
      for(i=0; i<2; i++){

        ( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = cu_cmul_PRECISION( ( in + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0], scalar );

      }
    }
  }
  else if( sites_to_add==_FULL_SYSTEM ){
    // even
    if(idx < 6*nr_block_even_sites){
      for(i=0; i<2; i++){

        ( out + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = cu_cmul_PRECISION( ( in + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0], scalar );

      }
    }
    // odd
    if(idx < 6*nr_block_odd_sites){
      for(i=0; i<2; i++){

        ( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = cu_cmul_PRECISION( ( in + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0], scalar );

      }
    }
  }

}


extern "C" void cuda_block_vector_PRECISION_scale( cuda_vector_PRECISION out, cuda_vector_PRECISION in, cu_cmplx_PRECISION scalar,
                                                   int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s, level_struct *l,
                                                   struct Thread *threading, int stream_id, cudaStream_t *streams, int color,
                                                   int* DD_blocks_to_compute_gpu, int* DD_blocks_to_compute_cpu ) {

  if( nr_DD_blocks_to_compute==0 ){ return; }

  int nr_threads, nr_threads_per_DD_block, threads_per_cublock;

  nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites; // nr sites per DD block
  nr_threads = nr_threads*(12/2); // threads per site
  nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
  nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

  threads_per_cublock = 96;

  cuda_block_oe_vector_PRECISION_scale_6threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                                       (out, in, scalar, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, \
                                                       l->num_lattice_site_var, (s->cu_s).block, _FULL_SYSTEM);
}

#endif
