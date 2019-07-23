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
__forceinline__ __device__ void _cuda_block_diag_ee_PRECISION(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, schwarz_PRECISION_struct_on_gpu *s, int idx, cu_config_PRECISION *clov_vect, int csw);
__forceinline__ __device__ void _cuda_block_n_hopping_term_PRECISION_plus_6threads(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, int amount, schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf, int ext_dir, int* gamma_coo, cu_cmplx_PRECISION* gamma_val);
__forceinline__ __device__ void _cuda_block_n_hopping_term_PRECISION_minus(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, int amount, schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf, int ext_dir);



__global__ void cuda_block_oe_vector_PRECISION_copy( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
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



__global__ void cuda_block_oe_vector_PRECISION_define( cu_cmplx_PRECISION* spinor,\
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



__global__ void cuda_block_oe_vector_PRECISION_plus( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in1, cu_cmplx_PRECISION* in2, \
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



__global__ void cuda_block_oe_vector_PRECISION_saxpy( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in1, cu_cmplx_PRECISION* in2, cu_cmplx_PRECISION alpha, \
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



__global__ void cuda_block_diag_oo_inv_PRECISION( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
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
    _cuda_block_diag_oo_inv_PRECISION(out_o, in_o, start, s, idx, clov_vect_b_o, csw);
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



__forceinline__ __device__ void _cuda_block_diag_oo_inv_PRECISION(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, \
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

        //if( (threadIdx.x + blockDim.x * blockIdx.x)==0 ){
        //  printf("local_idx=%d, j=%d\n", local_idx, j);
        //}

        if( local_idx>j || local_idx==j ){

          eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ], cu_cmul_PRECISION( (op_clov_vect_site + i*21)[matrx_indx], phi_site[j + 6*i] ) );

          //printf("%f+i%f\n", cu_creal_PRECISION((op_clov_vect_site + i*21)[matrx_indx]), cu_cimag_PRECISION((op_clov_vect_site + i*21)[matrx_indx]));

          //matrx_indx = 21 - (5-j+1)*((5-j+1)+1)/2 + (local_idx-j);
        }
        else{

          eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ], cu_cmul_PRECISION( cu_conj_PRECISION((op_clov_vect_site + i*21)[matrx_indx]), phi_site[j + 6*i] ) );

          //printf("%f+i%f\n", cu_creal_PRECISION(cu_conj_PRECISION((op_clov_vect_site + i*21)[matrx_indx])), cu_cimag_PRECISION(cu_conj_PRECISION((op_clov_vect_site + i*21)[matrx_indx])));

          //matrx_indx = 21 - (5-local_idx+1)*((5-local_idx+1)+1)/2 + (j-local_idx);
        }



      }
    }

  }
}













__global__ void cuda_block_diag_ee_PRECISION( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
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
    _cuda_block_diag_ee_PRECISION(out_o, in_o, start, s, idx, clov_vect_b_o, csw);
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



__forceinline__ __device__ void _cuda_block_diag_ee_PRECISION(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, \
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

        //if( (threadIdx.x + blockDim.x * blockIdx.x)==0 ){
        //  printf("local_idx=%d, j=%d\n", local_idx, j);
        //}

        if( local_idx>j || local_idx==j ){

          eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ], cu_cmul_PRECISION( (op_clov_vect_site + i*21)[matrx_indx], phi_site[j + 6*i] ) );

          //printf("%f+i%f\n", cu_creal_PRECISION((op_clov_vect_site + i*21)[matrx_indx]), cu_cimag_PRECISION((op_clov_vect_site + i*21)[matrx_indx]));

          //matrx_indx = 21 - (5-j+1)*((5-j+1)+1)/2 + (local_idx-j);
        }
        else{

          eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ], cu_cmul_PRECISION( cu_conj_PRECISION((op_clov_vect_site + i*21)[matrx_indx]), phi_site[j + 6*i] ) );

          //printf("%f+i%f\n", cu_creal_PRECISION(cu_conj_PRECISION((op_clov_vect_site + i*21)[matrx_indx])), cu_cimag_PRECISION(cu_conj_PRECISION((op_clov_vect_site + i*21)[matrx_indx])));

          //matrx_indx = 21 - (5-local_idx+1)*((5-local_idx+1)+1)/2 + (j-local_idx);
        }



      }
    }

  }
}

















// gamma_val and gamma_coo are both loaded into shared memory for this kernel
__global__ void cuda_block_n_hopping_term_PRECISION_plus_6threads( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                                                   schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                   int csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                                   int num_latt_site_var, block_struct* block, int ext_dir, int amount ){

  int idx;
  idx = threadIdx.x + blockDim.x * blockIdx.x;

  int* gamma_coo;

  cu_cmplx_PRECISION* gamma_val;

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

  extern __shared__ cu_cmplx_PRECISION shared_data_bare[];
  cu_cmplx_PRECISION* shared_data = shared_data_bare;

  gamma_coo = (int*)shared_data;
  shared_data = (cu_cmplx_PRECISION*)((int*)shared_data + 16);

  gamma_val = shared_data;
  shared_data = shared_data + 16;

  cu_cmplx_PRECISION *tmp_loc;
  tmp_loc = shared_data;
  shared_data = shared_data + 2*blockIdx.x;

  // loading gamma coordinates into shared memory
  if( threadIdx.x<16 ){
    gamma_coo[threadIdx.x] = s->gamma_info_coo[threadIdx.x];
  }

  // loading gamma values into shared memory
  if( threadIdx.x<16 ){
    gamma_val[threadIdx.x] = s->gamma_info_vals[threadIdx.x];
  }

  // initializing to zero a local buffer for temporary computations
  if(idx < 6*nr_block_even_sites){
    tmp_loc[threadIdx.x] = make_cu_cmplx_PRECISION(0.0,0.0);
    tmp_loc[threadIdx.x + blockDim.x] = make_cu_cmplx_PRECISION(0.0,0.0);
  }

  __syncthreads();

  _cuda_block_n_hopping_term_PRECISION_plus_6threads(out, in, start, amount, s, idx, tmp_loc, ext_dir, gamma_coo, gamma_val);
}


__forceinline__ __device__ void _cuda_block_n_hopping_term_PRECISION_plus_6threads(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, int amount, \
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
    //TODO
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
    leta = eta + 12*k;

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
__global__ void cuda_block_n_hopping_term_PRECISION_minus( cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
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


  _cuda_block_n_hopping_term_PRECISION_minus(out, in, start, amount, s, idx, tmp_loc, ext_dir);
}


__forceinline__ __device__ void _cuda_block_n_hopping_term_PRECISION_minus(cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start, int amount, schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf, int ext_dir){

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
    //TODO
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
      cuda_block_n_hopping_term_PRECISION_plus_6threads<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem >>> \
                                                       (out, in, s, thread_id, csw, nr_threads_per_DD_block, DD_blocks_to_compute, num_latt_site_var, block, dir, sites_to_compute);
      cuda_block_n_hopping_term_PRECISION_minus<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem >>> \
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

  int threads_per_cublock, nr_threads, nr_threads_per_DD_block;
  size_t tot_shared_mem;

  // this is the size of the local matrix, i.e. per lattice site. 12^2=144, but ... (??)
  //size_D_oeclov = (g.csw!=0) ? 72 : 12;
  size_D_oeclov = (g.csw!=0) ? 42 : 12;

  // we choose here a multiple of 96 due to being the smallest nr divisible by 32, but also divisible by 6
  threads_per_cublock = 96;

  nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites;
  nr_threads = nr_threads*(12/2);
  nr_threads = nr_threads*nr_DD_blocks_to_compute;

  nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

  // block_diag_ee_PRECISION( out, in, start, s, l, threading );
  //threads_per_cublock = 96;
  tot_shared_mem = 2*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 1*size_D_oeclov*(threads_per_cublock/6)*sizeof(cu_config_PRECISION);
  cuda_block_diag_ee_PRECISION<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                              (out, in, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block);

  //vector_PRECISION_define( tmp[0], 0, start + 12*s->num_block_even_sites, start + s->block_vector_size, l );
  cuda_block_oe_vector_PRECISION_define<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                       (tmp0, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, \
                                       l->num_lattice_site_var, (s->cu_s).block, _ODD_SITES, make_cu_cmplx_PRECISION(0.0,0.0));

  // TODO: implement block_hopping_term_PRECISION(...) with CUDA
  //block_hopping_term_PRECISION( tmp[0], in, start, _ODD_SITES, s, l, threading );

  // TODO: remove the following hopping computation
  //block_n_hopping_term_PRECISION( out, tmp[1], start, _EVEN_SITES, s, l, threading );
  //tot_shared_mem = 2*4*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 2*size_D_oeclov*(threads_per_cublock/6)*sizeof(cu_config_PRECISION);
  //tot_shared_mem = 1*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION);
  tot_shared_mem = (2+1)*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 1*36*(threads_per_cublock/6)*sizeof(cu_config_PRECISION) + 16*sizeof(cu_cmplx_PRECISION) + 16*sizeof(int);
  tot_shared_mem *= 3;
  //printf("tot_shared_mem = %d\n", tot_shared_mem);
  for( dir=0; dir<4; dir++ ){
    cuda_block_n_hopping_term_PRECISION_plus_6threads<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                                     (tmp0, in, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, dir, _ODD_SITES);
    cuda_block_n_hopping_term_PRECISION_minus<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                             (tmp0, in, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, dir, _ODD_SITES);
  }

  //block_diag_oo_inv_PRECISION( tmp[1], tmp[0], start, s, l, threading );
  // diag_oo inv
  threads_per_cublock = 96;
  tot_shared_mem = 2*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 1*size_D_oeclov*(threads_per_cublock/6)*sizeof(cu_config_PRECISION);
  cuda_block_diag_oo_inv_PRECISION<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                    (tmp1, tmp0, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block);

  //block_n_hopping_term_PRECISION( out, tmp[1], start, _EVEN_SITES, s, l, threading );
  // hopping term, even sites
  //tot_shared_mem = 2*4*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 2*size_D_oeclov*(threads_per_cublock/6)*sizeof(cu_config_PRECISION);
  //tot_shared_mem = 1*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION);
  tot_shared_mem = (2+1)*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 1*36*(threads_per_cublock/6)*sizeof(cu_config_PRECISION) + 16*sizeof(cu_cmplx_PRECISION) + 16*sizeof(int);
  for( dir=0; dir<4; dir++ ){
    cuda_block_n_hopping_term_PRECISION_plus_6threads<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                                     (out, tmp1, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, dir, _EVEN_SITES);
    cuda_block_n_hopping_term_PRECISION_minus<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                             (out, tmp1, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, dir, _EVEN_SITES);
  }

}



// sites_to_solve = {_EVEN_SITES, _ODD_SITES, _FULL_SYSTEM}
extern "C" void cuda_local_minres_PRECISION( cuda_vector_PRECISION phi, cuda_vector_PRECISION eta, cuda_vector_PRECISION latest_iter,
                                             schwarz_PRECISION_struct *s, level_struct *l, int nr_DD_blocks_to_compute,
                                             int* DD_blocks_to_compute, cudaStream_t *streams, int stream_id, int sites_to_solve ) {

  // This local_minres performs an inversion on EVEN sites only

  // TODO: remove the following commented lines !
  int i, n = l->block_iter;
  //int i, end = (g.odd_even&&l->depth==0)?start+12*s->num_block_even_sites:start+s->block_vector_size, n = l->block_iter;
  cuda_vector_PRECISION Dr = (s->cu_s).local_minres_buffer[0];
  cuda_vector_PRECISION r = (s->cu_s).local_minres_buffer[1];
  cuda_vector_PRECISION lphi = (s->cu_s).local_minres_buffer[2];
  cu_cmplx_PRECISION alpha;
  //void (*block_op)() = (l->depth==0)?(g.odd_even?apply_block_schur_complement_PRECISION:block_d_plus_clover_PRECISION)
  //                                  :coarse_block_operator_PRECISION;

  int threads_per_cublock, nr_threads, nr_threads_per_DD_block;
  //size_t tot_shared_mem;

  // we choose here a multiple of 96 due to being the smallest nr divisible by 32, but also divisible by 6
  threads_per_cublock = 96;

  // the nr of threads needed is computed like this: max between num_block_even_sites and num_block_odd_sites, and then
  //                                                 for each lattice site (of those even-odd), we need 12/2 really independent
  //                                                 components, due to gamma5 symmetry. I.e. each thread is in charge of
  //                                                 one site component !
  nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites;
  nr_threads = nr_threads*(12/2);
  nr_threads = nr_threads*nr_DD_blocks_to_compute;

  // this is the size of the local matrix, i.e. per lattice site. 12^2=144, but ... (??)
  //size_D_oeclov = (g.csw!=0) ? 72 : 12;

  nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

  // copy: r <----- eta
  // the use of _EVEN_SITES comes from the CPU code: end = (g.odd_even&&l->depth==0)?start+12*s->num_block_even_sites:start+s->block_vector_size
  //vector_PRECISION_copy( r, eta, start, end, l );
  cuda_block_oe_vector_PRECISION_copy<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                     (r, eta, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, sites_to_solve);

  //vector_PRECISION_define( lphi, 0, start, end, l );
  cuda_block_oe_vector_PRECISION_define<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                       (lphi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, \
                                       l->num_lattice_site_var, (s->cu_s).block, sites_to_solve, make_cu_cmplx_PRECISION(0.0,0.0));

  for ( i=0; i<n; i++ ) {
    // Dr = blockD*r
    //block_op( Dr, r, start, s, l, no_threading );
    cuda_apply_block_schur_complement_PRECISION( Dr, r, s, l, nr_DD_blocks_to_compute, DD_blocks_to_compute, streams, stream_id, _EVEN_SITES );

    // TODO: implement local_xy_over_xx_PRECISION(...) with CUDA
  //  // alpha = <Dr,r>/<Dr,Dr>
  //  alpha = local_xy_over_xx_PRECISION( Dr, r, start, end, l );

    // TODO: remove the following line
    alpha = make_cu_cmplx_PRECISION(1.0,3.0);

    // phi += alpha * r
    //vector_PRECISION_saxpy( lphi, lphi, r, alpha, start, end, l );
    cuda_block_oe_vector_PRECISION_saxpy<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                        (lphi, lphi, r, alpha, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, sites_to_solve);

    // r -= alpha * Dr
    // vector_PRECISION_saxpy( r, r, Dr, -alpha, start, end, l );
    cuda_block_oe_vector_PRECISION_saxpy<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                        (r, r, Dr, cu_csub_PRECISION(make_cu_cmplx_PRECISION(0.0,0.0), alpha), s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, sites_to_solve);

  }

  //vector_PRECISION_copy( latest_iter, lphi, start, end, l );
  if ( latest_iter != NULL ){
    cuda_block_oe_vector_PRECISION_copy<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                       (latest_iter, lphi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, sites_to_solve);
  }

  //vector_PRECISION_plus( phi, phi, lphi, start, end, l );
  if ( phi != NULL ){
    cuda_block_oe_vector_PRECISION_plus<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                       (phi, phi, lphi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block, sites_to_solve);
  }

  //vector_PRECISION_copy( eta, r, start, end, l );
  cuda_block_oe_vector_PRECISION_copy<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
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
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

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

    cuda_block_oe_vector_PRECISION_copy<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>> \
                                       (tmp3, r, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block, _FULL_SYSTEM);
    //cuda_blocks_vector_copy_noncontig_PRECISION_naive(tmp3, r, nr_DD_blocks_to_compute, s, l, DD_blocks_to_compute_cpu, streams);
    //cuda_blocks_vector_copy_noncontig_PRECISION_dyn(tmp3, r, nr_DD_blocks_to_compute, s, l, DD_blocks_to_compute_gpu, streams, 0);

    // diag_oo inv
    threads_per_cublock = 96;
    tot_shared_mem = 2*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 1*size_D_oeclov*(threads_per_cublock/6)*sizeof(cu_config_PRECISION);
    cuda_block_diag_oo_inv_PRECISION<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                    (tmp2, tmp3, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block);

    // hopping term, even sites
    //tot_shared_mem = 2*4*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 2*size_D_oeclov*(threads_per_cublock/6)*sizeof(cu_config_PRECISION);
    //tot_shared_mem = 1*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION);
    //for( dir=0; dir<4; dir++ ){
    //  cuda_block_n_hopping_term_PRECISION_plus_6threads<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
    //                                                   (tmp3, tmp2, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block, dir, _EVEN_SITES);
    //  cuda_block_n_hopping_term_PRECISION_minus<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
    //                                           (tmp3, tmp2, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block, dir, _EVEN_SITES);
    //}
    //cuda_block_n_hopping_term_PRECISION<<< 1, 32, 0, streams[stream_id] >>> \
    //                                   (tmp3, tmp2, s->s_on_gpu, g.my_rank, g.csw, nr_DD_blocks_to_compute, DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block, _EVEN_SITES);

    cuda_local_minres_PRECISION( NULL, tmp3, tmp2, s, l, nr_DD_blocks_to_compute, DD_blocks_to_compute_gpu, streams, stream_id, _EVEN_SITES );

    // hopping term, odd sites
    //tot_shared_mem = 2*4*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 2*size_D_oeclov*(threads_per_cublock/6)*sizeof(cu_config_PRECISION);
    //tot_shared_mem = 1*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION);
    //for( dir=0; dir<4; dir++ ){
    //  cuda_block_n_hopping_term_PRECISION_plus_6threads<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
    //                                                   (tmp3, tmp2, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block, dir, _ODD_SITES);
    //  cuda_block_n_hopping_term_PRECISION_minus<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
    //                                           (tmp3, tmp2, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block, dir, _ODD_SITES);
    //}
    //cuda_block_n_hopping_term_PRECISION<<< 1, 32, 0, streams[stream_id] >>> \
    //                                   (tmp3, tmp2, s->s_on_gpu, g.my_rank, g.csw, nr_DD_blocks_to_compute, DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block, _ODD_SITES);

    // diag_oo inv
    threads_per_cublock = 96;
    tot_shared_mem = 2*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) + 1*size_D_oeclov*(threads_per_cublock/6)*sizeof(cu_config_PRECISION);
    cuda_block_diag_oo_inv_PRECISION<<< nr_threads/threads_per_cublock, threads_per_cublock, tot_shared_mem, streams[stream_id] >>> \
                                    (tmp2, tmp3, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block);

    // update phi and latest_iter, and r
    cuda_block_solve_update<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>>
                           (phi, r, latest_iter, s->s_on_gpu, g.my_rank, g.csw, 0, nr_threads_per_DD_block, DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block);

    // TODO: eventually, remove this line
    cuda_check_error( _HARD_CHECK );
    //exit(1);

  }
}


#endif
