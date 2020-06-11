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

#ifdef CUDA_OPT

// Pre-definitions of CUDA functions to be called from the CUDA kernels, force inlining on some device functions

// 6 threads, naive

__forceinline__ __device__ void
_cuda_block_n_hopping_term_PRECISION_plus_6threads_naive(	cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
								cu_cmplx_PRECISION *buf, int ext_dir, int* gamma_coo, 
								cu_cmplx_PRECISION* gamma_val );
__forceinline__ __device__ void
_cuda_block_n_hopping_term_PRECISION_minus_6threads_naive(	cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
								cu_cmplx_PRECISION *buf, int ext_dir );
__global__ void
cuda_block_n_hopping_term_PRECISION_plus_6threads_naive(	cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
								schwarz_PRECISION_struct_on_gpu *s, int thread_id,
								double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
								int num_latt_site_var, block_struct* block, int ext_dir,
								int amount );
__global__ void
cuda_block_n_hopping_term_PRECISION_minus_6threads_naive(	cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
								schwarz_PRECISION_struct_on_gpu *s, int thread_id,
								double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
								int num_latt_site_var, block_struct* block, int ext_dir,
								int amount );
__forceinline__ __device__ void
_cuda_block_hopping_term_PRECISION_plus_6threads_naive(		cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
								cu_cmplx_PRECISION *buf, int ext_dir, int* gamma_coo,
								cu_cmplx_PRECISION* gamma_val );
__forceinline__ __device__ void
_cuda_block_hopping_term_PRECISION_minus_6threads_naive(	cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
								cu_cmplx_PRECISION *buf, int ext_dir );
__global__ void
cuda_block_hopping_term_PRECISION_plus_6threads_naive(		cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
								schwarz_PRECISION_struct_on_gpu *s, int thread_id, double csw,
								int nr_threads_per_DD_block, int* DD_blocks_to_compute,
								int num_latt_site_var, block_struct* block, int ext_dir,
								int amount );
__global__ void
cuda_block_hopping_term_PRECISION_minus_6threads_naive(		cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
								schwarz_PRECISION_struct_on_gpu *s, int thread_id, double csw,
								int nr_threads_per_DD_block, int* DD_blocks_to_compute,
								int num_latt_site_var, block_struct* block, int ext_dir,
								int amount );

// 6 threads, optimized

__forceinline__ __device__ void
_cuda_block_n_hopping_term_PRECISION_plus_6threads_opt(		cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
								cu_cmplx_PRECISION *buf, int ext_dir, cu_cmplx_PRECISION *Dgpu,
								int* gamma_coo, cu_cmplx_PRECISION* gamma_val );
__forceinline__ __device__ void
_cuda_block_n_hopping_term_PRECISION_minus_6threads_opt(	cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
								cu_cmplx_PRECISION *buf, int ext_dir, cu_cmplx_PRECISION *Dgpu,
								cu_cmplx_PRECISION *gamma_val_loc, int *gamma_coo_loc);
__global__ void
cuda_block_n_hopping_term_PRECISION_plus_6threads_opt(          cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block, int ext_dir,
                                                                int amount );
__global__ void
cuda_block_n_hopping_term_PRECISION_minus_6threads_opt(		cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
								schwarz_PRECISION_struct_on_gpu *s, int thread_id,
								double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
								int num_latt_site_var, block_struct* block, int ext_dir,
								int amount );
__forceinline__ __device__ void
_cuda_block_hopping_term_PRECISION_plus_6threads_opt(		cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
								cu_cmplx_PRECISION *buf, int ext_dir, cu_cmplx_PRECISION *Dgpu,
								int* gamma_coo, cu_cmplx_PRECISION* gamma_val );
__forceinline__ __device__ void
_cuda_block_hopping_term_PRECISION_minus_6threads_opt(		cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
								cu_cmplx_PRECISION *buf, int ext_dir, cu_cmplx_PRECISION *Dgpu,
								int* gamma_coo, cu_cmplx_PRECISION* gamma_val );
__global__ void
cuda_block_hopping_term_PRECISION_plus_6threads_opt(            cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block, int ext_dir,
                                                                int amount );
__global__ void
cuda_block_hopping_term_PRECISION_minus_6threads_opt(           cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block, int ext_dir,
                                                                int amount );
__forceinline__ __device__ void
_cuda_block_diag_oo_inv_PRECISION_6threads_opt(			cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								schwarz_PRECISION_struct_on_gpu *s, int idx,
								cu_config_PRECISION *clov_vect, double csw );
__forceinline__ __device__ void
_cuda_block_diag_ee_PRECISION_6threads_opt(			cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								schwarz_PRECISION_struct_on_gpu *s, int idx,
								cu_config_PRECISION *clov_vect, double csw );
__global__ void
cuda_block_diag_oo_inv_PRECISION_6threads_opt(			cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
								schwarz_PRECISION_struct_on_gpu *s, int thread_id, double csw,
								int nr_threads_per_DD_block, int* DD_blocks_to_compute,
								int num_latt_site_var, block_struct* block );
__global__ void
cuda_block_diag_ee_PRECISION_6threads_opt(			cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
								schwarz_PRECISION_struct_on_gpu *s, int thread_id, double csw,
								int nr_threads_per_DD_block, int* DD_blocks_to_compute,
								int num_latt_site_var, block_struct* block );
__global__ void
cuda_block_oe_vector_PRECISION_saxpy_6threads_opt(		cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in1,
								cu_cmplx_PRECISION* in2, cu_cmplx_PRECISION alpha,
								schwarz_PRECISION_struct_on_gpu *s, int thread_id, double csw,
								int nr_threads_per_DD_block, int* DD_blocks_to_compute,
								int num_latt_site_var, block_struct* block, int sites_to_add );
__global__ void
cuda_block_solve_update_6threads_opt(				cu_cmplx_PRECISION* phi, cu_cmplx_PRECISION* r,
								cu_cmplx_PRECISION* latest_iter, schwarz_PRECISION_struct_on_gpu *s,
								int thread_id, double csw, int kernel_id, int nr_threads_per_DD_block,
								int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block );
__global__ void
cuda_block_solve_update_12threads_opt(                           cu_cmplx_PRECISION* phi, cu_cmplx_PRECISION* r,
                                                                 cu_cmplx_PRECISION* latest_iter, schwarz_PRECISION_struct_on_gpu *s,
                                                                 int thread_id, double csw, int kernel_id, int nr_threads_per_DD_block,
                                                                 int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block );

// 2 threads, optimized

__forceinline__ __device__ void
_cuda_block_diag_oo_inv_PRECISION_2threads_opt(			cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								schwarz_PRECISION_struct_on_gpu *s, int idx, cu_config_PRECISION *clov_vect,
								double csw );
__forceinline__ __device__ void
_cuda_block_diag_ee_PRECISION_2threads_opt(			cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								schwarz_PRECISION_struct_on_gpu *s, int idx, cu_config_PRECISION *clov_vect,
								double csw );
__global__ void
cuda_block_diag_oo_inv_PRECISION_2threads_opt(			cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
								schwarz_PRECISION_struct_on_gpu *s, int thread_id, double csw,
								int nr_threads_per_DD_block, int* DD_blocks_to_compute,
								int num_latt_site_var, block_struct* block );
__global__ void
cuda_block_diag_ee_PRECISION_2threads_opt(			cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
								schwarz_PRECISION_struct_on_gpu *s, int thread_id, double csw,
								int nr_threads_per_DD_block, int* DD_blocks_to_compute,
								int num_latt_site_var, block_struct* block );
__forceinline__ __device__ void
_cuda_block_n_hopping_term_PRECISION_plus_2threads_opt(         cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
                                                                int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
                                                                cu_cmplx_PRECISION *buf, int ext_dir, cu_cmplx_PRECISION *Dgpu,
                                                                int* gamma_coo, cu_cmplx_PRECISION* gamma_val );
__global__ void
cuda_block_n_hopping_term_PRECISION_plus_2threads_opt(          cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block, int ext_dir,
                                                                int amount );
__forceinline__ __device__ void
_cuda_block_n_hopping_term_PRECISION_minus_2threads_opt(        cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
                                                                int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
                                                                cu_cmplx_PRECISION *buf, int ext_dir, cu_cmplx_PRECISION *Dgpu,
                                                                cu_cmplx_PRECISION *gamma_val_loc, int *gamma_coo_loc);
__global__ void
cuda_block_n_hopping_term_PRECISION_minus_2threads_opt(         cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block, int ext_dir,
                                                                int amount );
__forceinline__ __device__ void
_cuda_block_hopping_term_PRECISION_plus_2threads_opt(           cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
                                                                int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
                                                                cu_cmplx_PRECISION *buf, int ext_dir, cu_cmplx_PRECISION *Dgpu,
                                                                int* gamma_coo, cu_cmplx_PRECISION* gamma_val );
__global__ void
cuda_block_hopping_term_PRECISION_plus_2threads_opt(            cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block, int ext_dir,
                                                                int amount );
__forceinline__ __device__ void
_cuda_block_hopping_term_PRECISION_minus_2threads_opt(          cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
                                                                int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
                                                                cu_cmplx_PRECISION *buf, int ext_dir, cu_cmplx_PRECISION *Dgpu,
                                                                int* gamma_coo, cu_cmplx_PRECISION* gamma_val );
__global__ void
cuda_block_hopping_term_PRECISION_minus_2threads_opt(           cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block, int ext_dir,
                                                                int amount );


//------------------------------------------------------------------------------------------------------------------------------------------

// device functions

// this function encapsulates code common to the optimized hopping functions
// (those starting with "_")
__forceinline__ __device__ void
_DD_gateway_hopping(						schwarz_PRECISION_struct_on_gpu *s, int ext_dir,
								int start, int idx, int ***index,
                                                     		int **neighbor, int *loc_ind, int *idx_in_cublock, int *k, int *j, int *i,
                                                     		cu_cmplx_PRECISION **buf, cu_cmplx_PRECISION **buf1,
                                                     		cu_cmplx_PRECISION **buf2, cu_cmplx_PRECISION **Dgpu, int **ind, int tps ){
  *index = s->oe_index;
  *neighbor = s->op.neighbor_table;
  *loc_ind = idx%tps;
  *idx_in_cublock = idx%blockDim.x;
  *k=0;
  *j=0;
  *i=0;
  *buf += (*idx_in_cublock/tps)*12;
  *buf1 = *buf;
  *buf2 = *buf + 6;
  *Dgpu += (*idx_in_cublock/tps)*9;
  *ind = (*index)[ext_dir];
}

// this function encapsulates code common to the optimized hopping functions
__forceinline__ __device__ void
DD_gateway_hopping(						schwarz_PRECISION_struct_on_gpu *s, int *DD_blocks_to_compute,
								int nr_threads_per_DD_block, int num_latt_site_var, int *idx, int *start,
								int *int_offset_Dgpu, int *sites_offset_Dgpu, cu_cmplx_PRECISION **out,
								cu_cmplx_PRECISION **in, int *nr_dumps_Dgpu, int **gamma_coo,
								cu_cmplx_PRECISION **gamma_val, cu_cmplx_PRECISION **tmp_loc,
								cu_cmplx_PRECISION **Dgpu_local, int amount, block_struct *block,
								int ext_dir, int tps ){
  // tps = CUDA threads per site

  int i, DD_block_id, block_id, cublocks_per_DD_block, cu_block_ID;

  // not really a DD block id, but rather a linear counting of a grouping (per DD block) of CUDA threads
  DD_block_id = (*idx)/nr_threads_per_DD_block;
  // offsetting idx to make it zero at the beginning of the threads living within a DD block
  (*idx) = (*idx)%nr_threads_per_DD_block;
  // this int will be the ACTUAL DD block ID, in the sense of accessing data from e.g. block_struct* block
  block_id = DD_blocks_to_compute[DD_block_id];
  cublocks_per_DD_block = nr_threads_per_DD_block/blockDim.x;
  // This serves as a substitute of blockIdx.x, to have a more
  // local and DD-block treatment more independent of the other DD blocks
  cu_block_ID = blockIdx.x%cublocks_per_DD_block;
  // this is the DD-block start of the spinors (phi, r, latest_iter and temporary ones)
  (*start) = block[block_id].start * num_latt_site_var;
  (*int_offset_Dgpu) = ext_dir;
  // This offset is in "units" of nr of lattice sites, and
  // we have to multiply by 9 to get the offset within s->op.Dgpu
  // offset to the beginning of the corresponding DD block

  (*sites_offset_Dgpu) = (nr_threads_per_DD_block/tps)*block_id;

  // offset to the beginning of the CUDA block within that DD block
  (*sites_offset_Dgpu) += (blockDim.x/tps)*cu_block_ID;
  (*out) += (*start);
  (*in) += (*start);

  if( (((blockDim.x/tps)*9)%blockDim.x)==0 ){
    (*nr_dumps_Dgpu) = ((blockDim.x/tps)*9)/blockDim.x;
  }
  else{
    (*nr_dumps_Dgpu) = ((blockDim.x/tps)*9)/blockDim.x + 1;
  }

  extern __shared__ cu_cmplx_PRECISION shared_data_bare[];
  cu_cmplx_PRECISION* shared_data = shared_data_bare;
  //(*gamma_coo) = (int*)shared_data;
  //shared_data = (cu_cmplx_PRECISION*)((int*)shared_data + 16);
  //(*gamma_val) = shared_data;
  //shared_data = shared_data + 16;
  (*tmp_loc) = shared_data;
  shared_data = shared_data + (12/tps)*blockDim.x;
  (*Dgpu_local) = shared_data;

  // loading gamma coordinates into shared memory
  //if( threadIdx.x<16 ){
    //(*gamma_coo)[threadIdx.x] = s->gamma_info_coo[threadIdx.x];
    //(*gamma_coo)[threadIdx.x] = gamma_info_coo_PRECISION[threadIdx.x];
  //}
  // loading gamma values into shared memory
  //if( threadIdx.x<16 ){
    //(*gamma_val)[threadIdx.x] = s->gamma_info_vals[threadIdx.x];
    //(*gamma_val)[threadIdx.x] = gamma_info_vals_PRECISION[threadIdx.x];
  //}

  (*gamma_val) = gamma_info_vals_PRECISION;
  (*gamma_coo) = gamma_info_coo_PRECISION;

  // initializing to zero a local buffer for temporary computations
  //if(idx < 6*nr_block_even_sites){
  if((*idx) < tps*( (amount==_EVEN_SITES)?(s->dir_length_even[ext_dir]):(s->dir_length_odd[ext_dir]) )){

    for( i=0; i<(12/tps); i++ ){
      (*tmp_loc)[threadIdx.x + i*blockDim.x] = make_cu_cmplx_PRECISION(0.0,0.0);
      //(*tmp_loc)[threadIdx.x + blockDim.x] = make_cu_cmplx_PRECISION(0.0,0.0);
    }

  }
}


__forceinline__ __device__ void
_cuda_block_diag_oo_inv_PRECISION_6threads_opt(			cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								schwarz_PRECISION_struct_on_gpu *s, int idx,
								cu_config_PRECISION *op_clov_vect, double csw ){
  int local_idx, site_offset, matrx_indx, i, j;
  cu_cmplx_PRECISION *eta_site, *phi_site;
  cu_config_PRECISION *op_clov_vect_site;

  local_idx = idx%6;
  // this offset is per site and within each CUDA block
  site_offset = (threadIdx.x/6)*12;

  eta_site = eta + site_offset;
  phi_site = phi + site_offset;
  op_clov_vect_site = op_clov_vect + (site_offset/12)*42;

  eta_site[ local_idx ] = make_cu_cmplx_PRECISION(0.0,0.0);
  eta_site[ 6 + local_idx ] = make_cu_cmplx_PRECISION(0.0,0.0);

  if( csw!=0.0 ){
    // Run over all the sites within this block, and perform: eta = op_clov_vect * phi
    // There are 16 sites per CUDA block, in the case of choosing blockDimx.x equal to 96

    // first compute upper half of vector for each site, then lower half
    for( i=0; i<2; i++ ){
      // outter loop for matrix*vector double loop unrolled
      for( j=0; j<6; j++ ){

        // The following index is a mapping from the usual full matrix to the reduced form
        // (i.e. the compressed due to being Hermitian)
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
          eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ],
                                                           cu_cmul_PRECISION(
                                                           (op_clov_vect_site + i*21)[matrx_indx], phi_site[j + 6*i] ) 
                                                          );
        }
        else{
          eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ],
                                                           cu_cmul_PRECISION(
                                                           cu_conj_PRECISION( (op_clov_vect_site + i*21)[matrx_indx]), phi_site[j + 6*i] )
                                                          );
        }
      }
    }
  }
}


__forceinline__ __device__ void
_cuda_block_diag_oo_inv_PRECISION_2threads_opt(			cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								schwarz_PRECISION_struct_on_gpu *s, int idx,
								cu_config_PRECISION *op_clov_vect, double csw ){
  int local_idx, site_offset, k, matrx_indx, i, j;
  cu_cmplx_PRECISION *eta_site, *phi_site, *op_clov_vect_site;

  // this offset is per site and within each CUDA block
  site_offset = (threadIdx.x/2)*12;

  eta_site = eta + site_offset;
  phi_site = phi + site_offset;
  op_clov_vect_site = op_clov_vect + (site_offset/12)*42;

  for( k=0; k<3; k++ ){

    local_idx = (idx%2)*3 + k;

    eta_site[ local_idx ] = make_cu_cmplx_PRECISION(0.0,0.0);
    eta_site[ 6 + local_idx ] = make_cu_cmplx_PRECISION(0.0,0.0);

    if( csw!=0.0 ){
      // Run over all the sites within this block, and perform: eta = op_clov_vect * phi
      // There are 16 sites per CUDA block, in the case of choosing blockDimx.x equal to 96

      // first compute upper half of vector for each site, then lower half
      for( i=0; i<2; i++ ){
        // outter loop for matrix*vector double loop unrolled
        for( j=0; j<6; j++ ){

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
            eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ],
            						     cu_cmul_PRECISION(
            						     (op_clov_vect_site + i*21)[matrx_indx], phi_site[j + 6*i] )
            						    );
          }
          else{
            eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ],
            						     cu_cmul_PRECISION(
            						     cu_conj_PRECISION( (op_clov_vect_site + i*21)[matrx_indx]), phi_site[j + 6*i] )
            						    );
          }
        }
      }
    }
  }
}


__forceinline__ __device__ void
_cuda_block_diag_ee_PRECISION_6threads_opt(			cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								schwarz_PRECISION_struct_on_gpu *s, int idx,
								cu_config_PRECISION *op_clov_vect, double csw ){
  int local_idx, site_offset, matrx_indx, i, j;
  cu_cmplx_PRECISION *eta_site, *phi_site;
  cu_config_PRECISION *op_clov_vect_site;

  local_idx = idx%6;
  // this offset is per site and within each CUDA block
  site_offset = (threadIdx.x/6)*12;

  eta_site = eta + site_offset;
  phi_site = phi + site_offset;
  op_clov_vect_site = op_clov_vect + (site_offset/12)*42;

  eta_site[ local_idx ] = make_cu_cmplx_PRECISION(0.0,0.0);
  eta_site[ 6 + local_idx ] = make_cu_cmplx_PRECISION(0.0,0.0);

  if( csw!=0.0 ){
    // Run over all the sites within this block, and perform: eta = op_clov_vect * phi
    // There are 16 sites per CUDA block, in the case of choosing blockDimx.x equal to 96

    // first compute upper half of vector for each site, then lower half
    for( i=0; i<2; i++ ){
      // outter loop for matrix*vector double loop unrolled
      for( j=0; j<6; j++ ){

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
          eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ],
          					           cu_cmul_PRECISION(
          					           (op_clov_vect_site + i*21)[matrx_indx], phi_site[j + 6*i] )
          					          );
        }
        else{
          eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ],
           						   cu_cmul_PRECISION(
           						   cu_conj_PRECISION( (op_clov_vect_site + i*21)[matrx_indx]), phi_site[j + 6*i] )
           						  );
        }
      }
    }
  }
}


__forceinline__ __device__ void
_cuda_block_diag_ee_PRECISION_2threads_opt(			cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								schwarz_PRECISION_struct_on_gpu *s, int idx,
								cu_config_PRECISION *op_clov_vect, double csw ){

  int local_idx, site_offset, k, matrx_indx, i, j;
  cu_cmplx_PRECISION *eta_site, *phi_site;
  cu_config_PRECISION *op_clov_vect_site;

  // this offset is per site and within each CUDA block
  site_offset = (threadIdx.x/2)*12;

  eta_site = eta + site_offset;
  phi_site = phi + site_offset;
  op_clov_vect_site = op_clov_vect + (site_offset/12)*42;

  for( k=0; k<3; k++ ){

    local_idx = (idx%2)*3 + k;

    eta_site[ local_idx ] = make_cu_cmplx_PRECISION(0.0,0.0);
    eta_site[ 6 + local_idx ] = make_cu_cmplx_PRECISION(0.0,0.0);

    if( csw!=0.0 ){
      // Run over all the sites within this block, and perform: eta = op_clov_vect * phi
      // There are 16 sites per CUDA block, in the case of choosing blockDimx.x equal to 96

      // first compute upper half of vector for each site, then lower half
      for( i=0; i<2; i++ ){
        // outter loop for matrix*vector double loop unrolled
        for( j=0; j<6; j++ ){

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
            eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ],
            						     cu_cmul_PRECISION(
            						     (op_clov_vect_site + i*21)[matrx_indx], phi_site[j + 6*i] )
            						    );
          }
          else{
            eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ],
            						     cu_cmul_PRECISION(
            						     cu_conj_PRECISION( (op_clov_vect_site + i*21)[matrx_indx]), phi_site[j + 6*i] )
            						    );
          }
        }
      }
    }
  }
}


__forceinline__ __device__ void
_cuda_block_n_hopping_term_PRECISION_plus_6threads_naive(	cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
								cu_cmplx_PRECISION *buf, int ext_dir, int* gamma_coo,
								cu_cmplx_PRECISION* gamma_val ){
  //if amount==0 then even sites, if amount==1 then odd sites, else no oddeven split
  // ext_dir '=' {0,1,2,3} = {T,Z,Y,X}
  int a1, n1, k, j, i, **index, *ind, *neighbor, loc_ind, spin, w, idx_in_cublock;
  cu_cmplx_PRECISION *buf1, *buf2, *leta, *lphi; //eta and phi are already shifted by 'start'
  cu_config_PRECISION *D, *D_pt;
  index = s->oe_index;
  neighbor = s->op.neighbor_table;
  loc_ind = idx%6;
  idx_in_cublock = idx%blockDim.x;
  k=0;
  j=0;
  i=0;
  buf += (idx_in_cublock/6)*12;
  buf1 = buf;
  buf2 = buf + 6;
  D = s->op.D + (start/12)*36;
  ind = index[ext_dir];
  if( amount==_EVEN_SITES ){
    a1=0; n1=s->dir_length_even[ext_dir]; //for the + part
  }
  else if( amount==_ODD_SITES ){
    a1=s->dir_length_even[ext_dir]; n1=a1+s->dir_length_odd[ext_dir];
  }
  else{
    a1 = 0;
    n1 = s->dir_length[ext_dir];
  }
  // less threads in charge of this portion of execution, compute contribution due to even sites
  if( idx<6*(n1-a1) ){
    i = idx/6 + a1;
    k = ind[i];
    j = neighbor[4*k+ext_dir];
    D_pt = D + 36*k + 9*ext_dir;

    lphi = phi + 12*j;
    leta = eta + 12*k;

    spin = (loc_ind/3)*2;

    // prp_T_PRECISION(...)
    buf1[ loc_ind ] = cu_csub_PRECISION( lphi[ loc_ind ],
    					 cu_cmul_PRECISION(
    					 (gamma_val + ext_dir*4 + spin)[0], lphi[ 3*(gamma_coo + ext_dir*4 + spin)[0] + loc_ind%3 ] )
    					);
  }
  __syncthreads();
  if( idx<6*(n1-a1) ){
    // nmvm_PRECISION(...), twice
    buf2[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
    for( w=0; w<3; w++ ){
      buf2[ loc_ind ] = cu_csub_PRECISION( buf2[ loc_ind ],
      					   cu_cmul_PRECISION(
      					   D_pt[ (loc_ind*3)%9 + w ], buf1[ (loc_ind/3)*3 + w ] )
      					  );
    }
  }
  __syncthreads();
  if( idx<6*(n1-a1) ){
    // pbp_su3_T_PRECISION(...)
    leta[ loc_ind ] = cu_csub_PRECISION( leta[ loc_ind ], buf2[ loc_ind ] );
    leta[ 6 + loc_ind ] = cu_cadd_PRECISION( leta[ 6 + loc_ind ],
    					     cu_cmul_PRECISION(
    					     (gamma_val + ext_dir*4 + spin)[1], buf2[ 3*(gamma_coo + ext_dir*4 + spin)[1] + loc_ind%3 ] )
    					    );
  }
}


__forceinline__ __device__ void
_cuda_block_n_hopping_term_PRECISION_plus_2threads_opt(         cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
                                                                int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
                                                                cu_cmplx_PRECISION *buf, int ext_dir, cu_cmplx_PRECISION *Dgpu,
                                                                int* gamma_coo, cu_cmplx_PRECISION* gamma_val ){

  //if amount==0 then even sites, if amount==1 then odd sites, else no oddeven split
  int a1, n1, k, j, i, **index, *ind, *neighbor, loc_ind, spin, w, v, idx_in_cublock;
  cu_cmplx_PRECISION *buf1, *buf2, *leta, *lphi; //eta and phi are already shifted by 'start'

  _DD_gateway_hopping( s, ext_dir, start, idx, &index, &neighbor, &loc_ind, &idx_in_cublock, &k, &j, &i, &buf, &buf1, &buf2, &Dgpu, &ind, 2 );

  if( amount==_EVEN_SITES ){
    a1=0; n1=s->dir_length_even[ext_dir]; //for the + part
  }
  else if( amount==_ODD_SITES ){
    a1=s->dir_length_even[ext_dir]; n1=a1+s->dir_length_odd[ext_dir];
  }
  else{
    a1 = 0;
    n1 = s->dir_length[ext_dir];
  }
  // less threads in charge of this portion of execution, compute contribution due to even sites
  if( idx<2*(n1-a1) ){
    // lattice indices
    i = idx/2 + a1;
    k = ind[i];
    j = neighbor[4*k+ext_dir];
    // local spinors
    lphi = phi + 12*j;
    leta = eta + 12*k;
    // spin within site
    spin = loc_ind*2;
    // prp_T_PRECISION(...)

    for( v=0; v<3; v++ ){
      buf1[ loc_ind*3 + v ] = cu_csub_PRECISION( lphi[ loc_ind*3 + v ],
                                                 cu_cmul_PRECISION(
                                                 (gamma_val + ext_dir*4 + spin)[0], lphi[ 3*(gamma_coo + ext_dir*4 + spin)[0] + (loc_ind*3+v)%3 ] )
                                                );
    }
  }
  //__syncthreads();
  if( idx<2*(n1-a1) ){
    // nmvm_PRECISION(...), twice

    for( v=0; v<3; v++ ){
      buf2[ loc_ind*3+v ] = make_cu_cmplx_PRECISION(0.0,0.0);
      for( w=0; w<3; w++ ){
        buf2[ loc_ind*3+v ] = cu_csub_PRECISION( buf2[ loc_ind*3+v ], cu_cmul_PRECISION( Dgpu[ ((loc_ind*3+v)*3)%9 + w ], buf1[ ((loc_ind*3+v)/3)*3 + w ] ) );
      }
    }

  }
  //__syncthreads();
  if( idx<6*(n1-a1) ){
    // pbp_su3_T_PRECISION(...)

    for( v=0; v<3; v++ ){
      leta[ loc_ind*3+v ] = cu_csub_PRECISION( leta[ loc_ind*3+v ], buf2[ loc_ind*3+v ] );
      leta[ 6 + loc_ind*3+v ] = cu_cadd_PRECISION( leta[ 6 + loc_ind*3+v ],
                                               cu_cmul_PRECISION(
                                               (gamma_val + ext_dir*4 + spin)[1], buf2[ 3*(gamma_coo + ext_dir*4 + spin)[1] + (loc_ind*3+v)%3 ] )
                                               );
    }
  }
}


__forceinline__ __device__ void
_cuda_block_n_hopping_term_PRECISION_plus_6threads_opt(		cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
								cu_cmplx_PRECISION *buf, int ext_dir, cu_cmplx_PRECISION *Dgpu,
								int* gamma_coo, cu_cmplx_PRECISION* gamma_val ){
  //if amount==0 then even sites, if amount==1 then odd sites, else no oddeven split
  int a1, n1, k, j, i, **index, *ind, *neighbor, loc_ind, spin, w, idx_in_cublock;
  cu_cmplx_PRECISION *buf1, *buf2, *leta, *lphi; //eta and phi are already shifted by 'start'

  _DD_gateway_hopping( s, ext_dir, start, idx, &index, &neighbor, &loc_ind, &idx_in_cublock, &k, &j, &i, &buf, &buf1, &buf2, &Dgpu, &ind, 6 );

  if( amount==_EVEN_SITES ){
    a1=0; n1=s->dir_length_even[ext_dir]; //for the + part
  }
  else if( amount==_ODD_SITES ){
    a1=s->dir_length_even[ext_dir]; n1=a1+s->dir_length_odd[ext_dir];
  }
  else{
    a1 = 0;
    n1 = s->dir_length[ext_dir];
  }
  // less threads in charge of this portion of execution, compute contribution due to even sites
  if( idx<6*(n1-a1) ){
    // lattice indices
    i = idx/6 + a1;
    k = ind[i];
    j = neighbor[4*k+ext_dir];
    // local spinors
    lphi = phi + 12*j;
    leta = eta + 12*k;
    // spin within site
    spin = (loc_ind/3)*2;
    // prp_T_PRECISION(...)
    buf1[ loc_ind ] = cu_csub_PRECISION( lphi[ loc_ind ],
    					 cu_cmul_PRECISION(
    					 (gamma_val + ext_dir*4 + spin)[0], lphi[ 3*(gamma_coo + ext_dir*4 + spin)[0] + loc_ind%3 ] )
    					);
  }
  __syncthreads();
  if( idx<6*(n1-a1) ){
    // nmvm_PRECISION(...), twice
    buf2[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
    for( w=0; w<3; w++ ){
      buf2[ loc_ind ] = cu_csub_PRECISION( buf2[ loc_ind ], cu_cmul_PRECISION( Dgpu[ (loc_ind*3)%9 + w ], buf1[ (loc_ind/3)*3 + w ] ) );
    }
  }
  __syncthreads();
  if( idx<6*(n1-a1) ){
    // pbp_su3_T_PRECISION(...)
    leta[ loc_ind ] = cu_csub_PRECISION( leta[ loc_ind ], buf2[ loc_ind ] );
    leta[ 6 + loc_ind ] = cu_cadd_PRECISION( leta[ 6 + loc_ind ],
    					     cu_cmul_PRECISION(
    					     (gamma_val + ext_dir*4 + spin)[1], buf2[ 3*(gamma_coo + ext_dir*4 + spin)[1] + loc_ind%3 ] )
    					     );
  }
}


__forceinline__ __device__ void
_cuda_block_n_hopping_term_PRECISION_minus_6threads_naive(	cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
								cu_cmplx_PRECISION *buf, int ext_dir ){
  //if amount==0 then even sites, if amount==1 then odd sites, else no oddeven split
  // dir '=' {0,1,2,3} = {T,Z,Y,X}
  int a1, n1, a2, n2, k=0, j=0, i=0, **index = s->oe_index,
      *ind, *neighbor = s->op.neighbor_table, loc_ind=idx%6, spin, w, *gamma_coo,
      idx_in_cublock = idx%blockDim.x;
  cu_cmplx_PRECISION *gamma_val;

  cu_cmplx_PRECISION *buf1, *buf2;
  buf += (idx_in_cublock/6)*12;
  buf1 = buf;
  buf2 = buf + 6;

  cu_config_PRECISION* D = s->op.D + (start/12)*36;
  cu_config_PRECISION* D_pt;

  cu_cmplx_PRECISION *leta, *lphi; //eta and phi are already shifted by 'start'

  if( amount==_EVEN_SITES ){
    a1=0; n1=s->dir_length_even[ext_dir]; //for the + part
    a2=n1; n2=a2+s->dir_length_odd[ext_dir]; //for the - part
  }
  else if( amount==_ODD_SITES ){
    a1=s->dir_length_even[ext_dir]; n1=a1+s->dir_length_odd[ext_dir];
    a2=0; n2=a1;
  }
  else{
    a2 = 0;
    n2 = s->dir_length[ext_dir];
  }

  //a1 = 0; n1 = length_even[mu]+length_odd[mu];
  //a2 = 0; n2 = n1;

  ind = index[ext_dir];

  //and now, compute the contribution due to odd sites
  if( idx<6*(n2-a2) ){
    i = idx/6 + a2;
    k = ind[i];
    j = neighbor[4*k+ext_dir];
    D_pt = D + 36*k + 9*ext_dir;

    lphi = phi + 12*k;
    leta = eta + 12*j;

    spin = (loc_ind/3)*2;
    //with this setup, gamma_val[0] gives spins 0 and 1, and gamma_val[1] spins 2 and 3
    gamma_val = s->gamma_info_vals + ext_dir*4 + spin;
    gamma_coo = s->gamma_info_coo  + ext_dir*4 + spin;

    // prn_T_PRECISION(...)
    buf1[ loc_ind ] = cu_cadd_PRECISION( lphi[ loc_ind ],
    					 cu_cmul_PRECISION( gamma_val[0], lphi[ 3*gamma_coo[0] + loc_ind%3 ] )
    					);

  }

  __syncthreads();

  if( idx<6*(n2-a2) ){
    // nmvmh_PRECISION(...), twice
    buf2[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
    for( w=0; w<3; w++ ){
      buf2[ loc_ind ] = cu_csub_PRECISION( buf2[ loc_ind ],
      					   cu_cmul_PRECISION(
      					   cu_conj_PRECISION(D_pt[ loc_ind%3 + w*3 ]), buf1[ (loc_ind/3)*3 + w ] )
      					  );
    }
  }

  __syncthreads();

  if( idx<6*(n2-a2) ){
    // pbn_su3_T_PRECISION(...)
    leta[ loc_ind ] = cu_csub_PRECISION( leta[ loc_ind ], buf2[ loc_ind ] );
    leta[ 6 + loc_ind ] = cu_csub_PRECISION( leta[ 6 + loc_ind ],
    					     cu_cmul_PRECISION( gamma_val[1], buf2[ 3*gamma_coo[1] + loc_ind%3 ] )
    					    );
  }

}


__forceinline__ __device__ void
_cuda_block_n_hopping_term_PRECISION_minus_2threads_opt(        cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
                                                                int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
                                                                cu_cmplx_PRECISION *buf, int ext_dir, cu_cmplx_PRECISION *Dgpu,
                                                                cu_cmplx_PRECISION *gamma_val_loc, int *gamma_coo_loc ){
  //if amount==0 then even sites, if amount==1 then odd sites, else no oddeven split
  int a1, n1, a2, n2, k, j, i, **index, *ind, *neighbor, loc_ind, spin, w, v, idx_in_cublock;
  cu_cmplx_PRECISION *buf1, *buf2, *leta, *lphi; //eta and phi are already shifted by 'start'

  _DD_gateway_hopping( s, ext_dir, start, idx, &index, &neighbor,
                       &loc_ind, &idx_in_cublock, &k, &j, &i, &buf,
                       &buf1, &buf2, &Dgpu, &ind, 2 );

  if( amount==_EVEN_SITES ){
    a1=0; n1=s->dir_length_even[ext_dir]; //for the + part
    a2=n1; n2=a2+s->dir_length_odd[ext_dir]; //for the - part
  }
  else if( amount==_ODD_SITES ){
    a1=s->dir_length_even[ext_dir]; n1=a1+s->dir_length_odd[ext_dir];
    a2=0; n2=a1;
  }
  else{
    a2 = 0;
    n2 = s->dir_length[ext_dir];
  }
  //and now, compute the contribution due to odd sites
  if( idx<2*(n2-a2) ){
    // lattice site indices
    i = idx/2 + a2;
    k = ind[i];
    j = neighbor[4*k+ext_dir];
    // site spinors
    lphi = phi + 12*k;
    leta = eta + 12*j;
    // sub-site spin
    spin = loc_ind*2;
    // prn_T_PRECISION(...)

    for( v=0; v<3; v++ ){
      buf1[ loc_ind*3+v ] = cu_cadd_PRECISION( lphi[ loc_ind*3+v ],
                                               cu_cmul_PRECISION(
                                               (gamma_val_loc+ext_dir*4+spin)[0], lphi[ 3*(gamma_coo_loc+ext_dir*4+spin)[0] + (loc_ind*3+v)%3 ] )
                                              );
    }
  }
  //__syncthreads();
  if( idx<2*(n2-a2) ){
    // nmvmh_PRECISION(...), twice

    for( v=0; v<3; v++ ){
      buf2[ loc_ind*3+v ] = make_cu_cmplx_PRECISION(0.0,0.0);
      for( w=0; w<3; w++ ){
        buf2[ loc_ind*3+v ] = cu_csub_PRECISION( buf2[ loc_ind*3+v ],
                                                 cu_cmul_PRECISION(
                                                 cu_conj_PRECISION(Dgpu[ (loc_ind*3+v)%3 + w*3 ]), buf1[ ((loc_ind*3+v)/3)*3 + w ] )
                                                );
      }
    }

  }
  //__syncthreads();
  if( idx<2*(n2-a2) ){
    // pbn_su3_T_PRECISION(...)

    for( v=0; v<3; v++ ){
      leta[ loc_ind*3+v ] = cu_csub_PRECISION( leta[ loc_ind*3+v ], buf2[ loc_ind*3+v ] );
      leta[ 6 + loc_ind*3+v ] = cu_csub_PRECISION( leta[ 6 + loc_ind*3+v ],
                                                   cu_cmul_PRECISION(
                                                   (gamma_val_loc+ext_dir*4+spin)[1], buf2[ 3*(gamma_coo_loc+ext_dir*4+spin)[1] + (loc_ind*3+v)%3 ] )
                                                  );
    }

  }
}


__forceinline__ __device__ void
_cuda_block_n_hopping_term_PRECISION_minus_6threads_opt(	cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
								cu_cmplx_PRECISION *buf, int ext_dir, cu_cmplx_PRECISION *Dgpu,
								cu_cmplx_PRECISION *gamma_val_loc, int *gamma_coo_loc ){
  //if amount==0 then even sites, if amount==1 then odd sites, else no oddeven split
  int a1, n1, a2, n2, k, j, i, **index, *ind, *neighbor, loc_ind, spin, w, idx_in_cublock;
  cu_cmplx_PRECISION *buf1, *buf2, *leta, *lphi; //eta and phi are already shifted by 'start'

  _DD_gateway_hopping( s, ext_dir, start, idx, &index, &neighbor,
  		       &loc_ind, &idx_in_cublock, &k, &j, &i, &buf,
  		       &buf1, &buf2, &Dgpu, &ind, 6 );

  if( amount==_EVEN_SITES ){
    a1=0; n1=s->dir_length_even[ext_dir]; //for the + part
    a2=n1; n2=a2+s->dir_length_odd[ext_dir]; //for the - part
  }
  else if( amount==_ODD_SITES ){
    a1=s->dir_length_even[ext_dir]; n1=a1+s->dir_length_odd[ext_dir];
    a2=0; n2=a1;
  }
  else{
    a2 = 0;
    n2 = s->dir_length[ext_dir];
  }
  //and now, compute the contribution due to odd sites
  if( idx<6*(n2-a2) ){
    // lattice site indices
    i = idx/6 + a2;
    k = ind[i];
    j = neighbor[4*k+ext_dir];
    // site spinors
    lphi = phi + 12*k;
    leta = eta + 12*j;
    // sub-site spin
    spin = (loc_ind/3)*2;
    // prn_T_PRECISION(...)
    buf1[ loc_ind ] = cu_cadd_PRECISION( lphi[ loc_ind ],
    					 cu_cmul_PRECISION(
    					 (gamma_val_loc+ext_dir*4+spin)[0], lphi[ 3*(gamma_coo_loc+ext_dir*4+spin)[0] + loc_ind%3 ] )
    					);
  }
  __syncthreads();
  if( idx<6*(n2-a2) ){
    // nmvmh_PRECISION(...), twice
    buf2[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
    for( w=0; w<3; w++ ){
      buf2[ loc_ind ] = cu_csub_PRECISION( buf2[ loc_ind ],
      					   cu_cmul_PRECISION(
      					   cu_conj_PRECISION(Dgpu[ loc_ind%3 + w*3 ]), buf1[ (loc_ind/3)*3 + w ] )
      					  );
    }
  }
  __syncthreads();
  if( idx<6*(n2-a2) ){
    // pbn_su3_T_PRECISION(...)
    leta[ loc_ind ] = cu_csub_PRECISION( leta[ loc_ind ], buf2[ loc_ind ] );
    leta[ 6 + loc_ind ] = cu_csub_PRECISION( leta[ 6 + loc_ind ],
    					     cu_cmul_PRECISION(
    					     (gamma_val_loc+ext_dir*4+spin)[1], buf2[ 3*(gamma_coo_loc+ext_dir*4+spin)[1] + loc_ind%3 ] )
    					    );
  }
}


__forceinline__ __device__ void
_cuda_block_hopping_term_PRECISION_plus_6threads_naive(		cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
								cu_cmplx_PRECISION *buf, int ext_dir, int* gamma_coo,
								cu_cmplx_PRECISION* gamma_val ){
  //if amount==0 then even sites, if amount==1 then odd sites, else no oddeven split
  // dir '=' {0,1,2,3} = {T,Z,Y,X}
  int a1, n1, k=0, j=0, i=0, **index = s->oe_index, *ind,
      *neighbor = s->op.neighbor_table, loc_ind=idx%6, spin, w,
      idx_in_cublock = idx%blockDim.x;

  cu_cmplx_PRECISION *buf1, *buf2;
  buf += (idx_in_cublock/6)*12;
  buf1 = buf;
  buf2 = buf + 6;

  cu_config_PRECISION* D = s->op.D + (start/12)*36;
  cu_config_PRECISION* D_pt;

  cu_cmplx_PRECISION *leta, *lphi; //eta and phi are already shifted by 'start'

  if( amount==_EVEN_SITES ){
    a1=0; n1=s->dir_length_even[ext_dir]; //for the + part
  }
  else if( amount==_ODD_SITES ){
    a1=s->dir_length_even[ext_dir]; n1=a1+s->dir_length_odd[ext_dir];
  }
  else{
    a1 = 0;
    n1 = s->dir_length[ext_dir];
  }

  ind = index[ext_dir];

  // less threads in charge of this portion of execution, compute contribution due to even sites
  if( idx<6*(n1-a1) ){
    i = idx/6 + a1;
    k = ind[i];
    j = neighbor[4*k+ext_dir];
    D_pt = D + 36*k + 9*ext_dir;

    lphi = phi + 12*j;
    leta = eta + 12*k;

    spin = (loc_ind/3)*2;

    // prp_T_PRECISION(...)
    buf1[ loc_ind ] = cu_csub_PRECISION( lphi[ loc_ind ],
    					 cu_cmul_PRECISION(
    					 (gamma_val + ext_dir*4 + spin)[0], lphi[ 3*(gamma_coo + ext_dir*4 + spin)[0] + loc_ind%3 ] )
    					);
  }
  __syncthreads();
  if( idx<6*(n1-a1) ){
    // mvm_PRECISION(...), twice
    buf2[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
    for( w=0; w<3; w++ ){
      buf2[ loc_ind ] = cu_cadd_PRECISION( buf2[ loc_ind ],
      					   cu_cmul_PRECISION(
      					   D_pt[ (loc_ind*3)%9 + w ], buf1[ (loc_ind/3)*3 + w ] )
      					  );
    }
  }
  __syncthreads();
  if( idx<6*(n1-a1) ){
    // pbp_su3_T_PRECISION(...)
    leta[ loc_ind ] = cu_csub_PRECISION( leta[ loc_ind ], buf2[ loc_ind ] );
    leta[ 6 + loc_ind ] = cu_cadd_PRECISION( leta[ 6 + loc_ind ],
    					     cu_cmul_PRECISION(
    					     (gamma_val + ext_dir*4 + spin)[1], buf2[ 3*(gamma_coo + ext_dir*4 + spin)[1] + loc_ind%3 ] )
    					    );
  }
}


__forceinline__ __device__ void
_cuda_block_hopping_term_PRECISION_plus_2threads_opt(           cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
                                                                int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
                                                                cu_cmplx_PRECISION *buf, int ext_dir, cu_cmplx_PRECISION *Dgpu,
                                                                int* gamma_coo, cu_cmplx_PRECISION* gamma_val ){

  //if amount==0 then even sites, if amount==1 then odd sites, else no oddeven split
  int a1, n1, k, j, i, **index, *ind, *neighbor, loc_ind, spin, w, v, idx_in_cublock;
  cu_cmplx_PRECISION *buf1, *buf2, *leta, *lphi; //eta and phi are already shifted by 'start'

  _DD_gateway_hopping( s, ext_dir, start, idx, &index, &neighbor,
                       &loc_ind, &idx_in_cublock, &k, &j, &i, &buf,
                       &buf1, &buf2, &Dgpu, &ind, 2 );

  if( amount==_EVEN_SITES ){
    a1=0; n1=s->dir_length_even[ext_dir]; //for the + part
  }
  else if( amount==_ODD_SITES ){
    a1=s->dir_length_even[ext_dir]; n1=a1+s->dir_length_odd[ext_dir];
  }
  else{
    a1 = 0;
    n1 = s->dir_length[ext_dir];
  }
  // less threads in charge of this portion of execution, compute contribution due to even sites
  if( idx<2*(n1-a1) ){
    // lattice site indices
    i = idx/2 + a1;
    k = ind[i];
    j = neighbor[4*k+ext_dir];
    // site spinors
    lphi = phi + 12*j;
    leta = eta + 12*k;
    // sub-site spin
    spin = loc_ind*2;
    // prp_T_PRECISION(...)

    for( v=0; v<3; v++ ){
      buf1[ loc_ind*3+v ] = cu_csub_PRECISION( lphi[ loc_ind*3+v ],
                                               cu_cmul_PRECISION(
                                               (gamma_val + ext_dir*4 + spin)[0], lphi[ 3*(gamma_coo + ext_dir*4 + spin)[0] + (loc_ind*3+v)%3 ] )
                                              );
    }

  }
  //__syncthreads();
  if( idx<2*(n1-a1) ){
    // mvm_PRECISION(...), twice

    for( v=0; v<3; v++ ){
      buf2[ loc_ind*3+v ] = make_cu_cmplx_PRECISION(0.0,0.0);
      for( w=0; w<3; w++ ){
        buf2[ loc_ind*3+v ] = cu_cadd_PRECISION( buf2[ loc_ind*3+v ],
                                                 cu_cmul_PRECISION(
                                                 Dgpu[ ((loc_ind*3+v)*3)%9 + w ], buf1[ ((loc_ind*3+v)/3)*3 + w ] )
                                                );
      }

    }
  }
  //__syncthreads();
  if( idx<2*(n1-a1) ){
    // pbp_su3_T_PRECISION(...)

    for( v=0; v<3; v++ ){
      leta[ loc_ind*3+v ] = cu_csub_PRECISION( leta[ loc_ind*3+v ], buf2[ loc_ind*3+v ] );
      leta[ 6 + loc_ind*3+v ] = cu_cadd_PRECISION( leta[ 6 + loc_ind*3+v ],
                                                   cu_cmul_PRECISION(
                                                   (gamma_val + ext_dir*4 + spin)[1], buf2[ 3*(gamma_coo + ext_dir*4 + spin)[1] + (loc_ind*3+v)%3 ] )
                                                 );
    }

  }
}


__forceinline__ __device__ void
_cuda_block_hopping_term_PRECISION_plus_6threads_opt(		cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
								cu_cmplx_PRECISION *buf, int ext_dir, cu_cmplx_PRECISION *Dgpu,
								int* gamma_coo, cu_cmplx_PRECISION* gamma_val ){

  //if amount==0 then even sites, if amount==1 then odd sites, else no oddeven split
  int a1, n1, k, j, i, **index, *ind, *neighbor, loc_ind, spin, w, idx_in_cublock;
  cu_cmplx_PRECISION *buf1, *buf2, *leta, *lphi; //eta and phi are already shifted by 'start'

  _DD_gateway_hopping( s, ext_dir, start, idx, &index, &neighbor,
  		       &loc_ind, &idx_in_cublock, &k, &j, &i, &buf,
  		       &buf1, &buf2, &Dgpu, &ind, 6 );

  if( amount==_EVEN_SITES ){
    a1=0; n1=s->dir_length_even[ext_dir]; //for the + part
  }
  else if( amount==_ODD_SITES ){
    a1=s->dir_length_even[ext_dir]; n1=a1+s->dir_length_odd[ext_dir];
  }
  else{
    a1 = 0;
    n1 = s->dir_length[ext_dir];
  }
  // less threads in charge of this portion of execution, compute contribution due to even sites
  if( idx<6*(n1-a1) ){
    // lattice site indices
    i = idx/6 + a1;
    k = ind[i];
    j = neighbor[4*k+ext_dir];
    // site spinors
    lphi = phi + 12*j;
    leta = eta + 12*k;
    // sub-site spin
    spin = (loc_ind/3)*2;
    // prp_T_PRECISION(...)
    buf1[ loc_ind ] = cu_csub_PRECISION( lphi[ loc_ind ],
    					 cu_cmul_PRECISION(
    					 (gamma_val + ext_dir*4 + spin)[0], lphi[ 3*(gamma_coo + ext_dir*4 + spin)[0] + loc_ind%3 ] )
    					);
  }
  __syncthreads();
  if( idx<6*(n1-a1) ){
    // mvm_PRECISION(...), twice
    buf2[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
    for( w=0; w<3; w++ ){
      buf2[ loc_ind ] = cu_cadd_PRECISION( buf2[ loc_ind ],
      					   cu_cmul_PRECISION(
      					   Dgpu[ (loc_ind*3)%9 + w ], buf1[ (loc_ind/3)*3 + w ] )
      					  );
    }
  }
  __syncthreads();
  if( idx<6*(n1-a1) ){
    // pbp_su3_T_PRECISION(...)
    leta[ loc_ind ] = cu_csub_PRECISION( leta[ loc_ind ], buf2[ loc_ind ] );
    leta[ 6 + loc_ind ] = cu_cadd_PRECISION( leta[ 6 + loc_ind ],
    					     cu_cmul_PRECISION(
    					     (gamma_val + ext_dir*4 + spin)[1], buf2[ 3*(gamma_coo + ext_dir*4 + spin)[1] + loc_ind%3 ] )
    					    );
  }
}


__forceinline__ __device__ void
_cuda_block_hopping_term_PRECISION_minus_6threads_naive(	cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
								cu_cmplx_PRECISION *buf, int ext_dir ){

  //if amount==0 then even sites, if amount==1 then odd sites, else no oddeven split

  // dir '=' {0,1,2,3} = {T,Z,Y,X}
  int a1, n1, a2, n2, k=0, j=0, i=0, **index = s->oe_index,
      *ind, *neighbor = s->op.neighbor_table, loc_ind=idx%6, spin, w, *gamma_coo,
      idx_in_cublock = idx%blockDim.x;
  cu_cmplx_PRECISION *gamma_val;
  cu_cmplx_PRECISION *buf1, *buf2;

  buf += (idx_in_cublock/6)*12;
  buf1 = buf;
  buf2 = buf + 6;

  cu_config_PRECISION* D = s->op.D + (start/12)*36;
  cu_config_PRECISION* D_pt;

  cu_cmplx_PRECISION *leta, *lphi; //eta and phi are already shifted by 'start'

  if( amount==_EVEN_SITES ){
    a1=0; n1=s->dir_length_even[ext_dir]; //for the + part
    a2=n1; n2=a2+s->dir_length_odd[ext_dir]; //for the - part
  }
  else if( amount==_ODD_SITES ){
    a1=s->dir_length_even[ext_dir]; n1=a1+s->dir_length_odd[ext_dir];
    a2=0; n2=a1;
  }
  else{
    a2 = 0;
    n2 = s->dir_length[ext_dir];
  }

  ind = index[ext_dir];

  //and now, compute the contribution due to odd sites
  if( idx<6*(n2-a2) ){
    i = idx/6 + a2;
    k = ind[i];
    j = neighbor[4*k+ext_dir];
    D_pt = D + 36*k + 9*ext_dir;

    lphi = phi + 12*k;
    leta = eta + 12*j;

    spin = (loc_ind/3)*2;
    //with this setup, gamma_val[0] gives spins 0 and 1, and gamma_val[1] spins 2 and 3
    gamma_val = s->gamma_info_vals + ext_dir*4 + spin;
    gamma_coo = s->gamma_info_coo  + ext_dir*4 + spin;

    // prn_T_PRECISION(...)
    buf1[ loc_ind ] = cu_cadd_PRECISION( lphi[ loc_ind ],
    					 cu_cmul_PRECISION(
    					 gamma_val[0], lphi[ 3*gamma_coo[0] + loc_ind%3 ] )
    					);
  }
  __syncthreads();
  if( idx<6*(n2-a2) ){
    // mvmh_PRECISION(...), twice
    buf2[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
    for( w=0; w<3; w++ ){
      buf2[ loc_ind ] = cu_cadd_PRECISION( buf2[ loc_ind ],
      					   cu_cmul_PRECISION(
      					   cu_conj_PRECISION(D_pt[ loc_ind%3 + w*3 ]), buf1[ (loc_ind/3)*3 + w ] )
      					  );
    }
  }
  __syncthreads();
  if( idx<6*(n2-a2) ){
    // pbn_su3_T_PRECISION(...)
    leta[ loc_ind ] = cu_csub_PRECISION( leta[ loc_ind ], buf2[ loc_ind ] );
    leta[ 6 + loc_ind ] = cu_csub_PRECISION( leta[ 6 + loc_ind ],
    					     cu_cmul_PRECISION(
    					     gamma_val[1], buf2[ 3*gamma_coo[1] + loc_ind%3 ] )
    					    );
  }
}


__forceinline__ __device__ void
_cuda_block_hopping_term_PRECISION_minus_2threads_opt(          cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
                                                                int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
                                                                cu_cmplx_PRECISION *buf, int ext_dir, cu_cmplx_PRECISION *Dgpu,
                                                                int *gamma_coo, cu_cmplx_PRECISION *gamma_val ){

  //if amount==0 then even sites, if amount==1 then odd sites, else no oddeven split
  int a1, n1, a2, n2, k, j, i, **index, *ind, *neighbor, loc_ind, spin, w, v, idx_in_cublock;
  cu_cmplx_PRECISION *buf1, *buf2, *leta, *lphi; //eta and phi are already shifted by 'start'

  _DD_gateway_hopping( s, ext_dir, start, idx, &index, &neighbor, &loc_ind, &idx_in_cublock, &k, &j, &i, &buf, &buf1, &buf2, &Dgpu, &ind, 2 );

  if( amount==_EVEN_SITES ){
    a1=0; n1=s->dir_length_even[ext_dir]; //for the + part
    a2=n1; n2=a2+s->dir_length_odd[ext_dir]; //for the - part
  }
  else if( amount==_ODD_SITES ){
    a1=s->dir_length_even[ext_dir]; n1=a1+s->dir_length_odd[ext_dir];
    a2=0; n2=a1;
  }
  else{
    a2 = 0;
    n2 = s->dir_length[ext_dir];
  }
  //and now, compute the contribution due to odd sites
  if( idx<2*(n2-a2) ){
    // lattice site indices
    i = idx/2 + a2;
    k = ind[i];
    j = neighbor[4*k+ext_dir];
    // site spinors
    lphi = phi + 12*k;
    leta = eta + 12*j;
    // sub-site spin
    spin = loc_ind*2;
    // prn_T_PRECISION(...)

    for( v=0; v<3; v++ ){
      buf1[ loc_ind*3+v ] = cu_cadd_PRECISION( lphi[ loc_ind*3+v ],
                                               cu_cmul_PRECISION(
                                               (gamma_val+ext_dir*4+spin)[0], lphi[ 3*(gamma_coo+ext_dir*4+spin)[0] + (loc_ind*3+v)%3 ] )
                                              );
    }
  }
  //__syncthreads();
  if( idx<2*(n2-a2) ){
    // mvmh_PRECISION(...), twice

    for( v=0; v<3; v++ ){
      buf2[ loc_ind*3+v ] = make_cu_cmplx_PRECISION(0.0,0.0);
      for( w=0; w<3; w++ ){
        buf2[ loc_ind*3+v ] = cu_cadd_PRECISION( buf2[ loc_ind*3+v ],
                                                 cu_cmul_PRECISION(
                                                 cu_conj_PRECISION(Dgpu[ (loc_ind*3+v)%3 + w*3 ]), buf1[ ((loc_ind*3+v)/3)*3 + w ] )
                                                );
      }

    }
  }
  //__syncthreads();
  if( idx<2*(n2-a2) ){
    // pbn_su3_T_PRECISION(...)

    for( v=0; v<3; v++ ){
      leta[ loc_ind*3+v ] = cu_csub_PRECISION( leta[ loc_ind*3+v ], buf2[ loc_ind*3+v ] );
      leta[ 6 + loc_ind*3+v ] = cu_csub_PRECISION( leta[ 6 + loc_ind*3+v ],
                                                   cu_cmul_PRECISION(
                                                   (gamma_val+ext_dir*4+spin)[1], buf2[ 3*(gamma_coo+ext_dir*4+spin)[1] + (loc_ind*3+v)%3 ] )
                                                  );
    }

  }
}


__forceinline__ __device__ void
_cuda_block_hopping_term_PRECISION_minus_6threads_opt(		cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								int amount, schwarz_PRECISION_struct_on_gpu *s, int idx,
								cu_cmplx_PRECISION *buf, int ext_dir, cu_cmplx_PRECISION *Dgpu,
								int *gamma_coo, cu_cmplx_PRECISION *gamma_val ){

  //if amount==0 then even sites, if amount==1 then odd sites, else no oddeven split
  int a1, n1, a2, n2, k, j, i, **index, *ind, *neighbor, loc_ind, spin, w, idx_in_cublock;
  cu_cmplx_PRECISION *buf1, *buf2, *leta, *lphi; //eta and phi are already shifted by 'start'

  _DD_gateway_hopping( s, ext_dir, start, idx, &index, &neighbor, &loc_ind, &idx_in_cublock, &k, &j, &i, &buf, &buf1, &buf2, &Dgpu, &ind, 6 );

  if( amount==_EVEN_SITES ){
    a1=0; n1=s->dir_length_even[ext_dir]; //for the + part
    a2=n1; n2=a2+s->dir_length_odd[ext_dir]; //for the - part
  }
  else if( amount==_ODD_SITES ){
    a1=s->dir_length_even[ext_dir]; n1=a1+s->dir_length_odd[ext_dir];
    a2=0; n2=a1;
  }
  else{
    a2 = 0;
    n2 = s->dir_length[ext_dir];
  }
  //and now, compute the contribution due to odd sites
  if( idx<6*(n2-a2) ){
    // lattice site indices
    i = idx/6 + a2;
    k = ind[i];
    j = neighbor[4*k+ext_dir];
    // site spinors
    lphi = phi + 12*k;
    leta = eta + 12*j;
    // sub-site spin
    spin = (loc_ind/3)*2;
    // prn_T_PRECISION(...)
    buf1[ loc_ind ] = cu_cadd_PRECISION( lphi[ loc_ind ],
    					 cu_cmul_PRECISION(
    					 (gamma_val+ext_dir*4+spin)[0], lphi[ 3*(gamma_coo+ext_dir*4+spin)[0] + loc_ind%3 ] )
    					);
  }
  __syncthreads();
  if( idx<6*(n2-a2) ){
    // mvmh_PRECISION(...), twice
    buf2[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
    for( w=0; w<3; w++ ){
      buf2[ loc_ind ] = cu_cadd_PRECISION( buf2[ loc_ind ],
      					   cu_cmul_PRECISION(
      					   cu_conj_PRECISION(Dgpu[ loc_ind%3 + w*3 ]), buf1[ (loc_ind/3)*3 + w ] )
      					  );
    }
  }
  __syncthreads();
  if( idx<6*(n2-a2) ){
    // pbn_su3_T_PRECISION(...)
    leta[ loc_ind ] = cu_csub_PRECISION( leta[ loc_ind ], buf2[ loc_ind ] );
    leta[ 6 + loc_ind ] = cu_csub_PRECISION( leta[ 6 + loc_ind ],
    					     cu_cmul_PRECISION(
    					     (gamma_val+ext_dir*4+spin)[1], buf2[ 3*(gamma_coo+ext_dir*4+spin)[1] + loc_ind%3 ] )
    					    );
  }
}


__global__ void
cuda_block_solve_update_12threads_opt(                           cu_cmplx_PRECISION* phi, cu_cmplx_PRECISION* r,
                                                                 cu_cmplx_PRECISION* latest_iter, schwarz_PRECISION_struct_on_gpu *s,
                                                                 int thread_id, double csw, int kernel_id, int nr_threads_per_DD_block,
                                                                 int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block ){

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
  if(idx < 12*nr_block_even_sites){

     ( latest_iter + cu_block_ID*blockDim.x + threadIdx.x )[0] = ( tmp2 + cu_block_ID*blockDim.x + threadIdx.x )[0];

  }
  // odd
  if(idx < 12*nr_block_odd_sites){

    ( latest_iter + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0] = ( tmp2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0];

  }
  // even
  if(idx < 12*nr_block_even_sites){

    ( phi + cu_block_ID*blockDim.x + threadIdx.x )[0] = cu_cadd_PRECISION( ( phi  + cu_block_ID*blockDim.x + threadIdx.x )[0],
                                                                           ( tmp2 + cu_block_ID*blockDim.x + threadIdx.x )[0] );

  }
  // odd
  if(idx < 12*nr_block_odd_sites){

    ( phi + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0] = \
                          cu_cadd_PRECISION( ( phi  + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0],
                                             ( tmp2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0] );

  }
  // even
  if(idx < 12*nr_block_even_sites){

    ( r + cu_block_ID*blockDim.x + threadIdx.x )[0] = ( tmp3 + cu_block_ID*blockDim.x + threadIdx.x )[0];

  }
  // odd
  if(idx < 12*nr_block_odd_sites){

    ( r + 12*nr_block_even_sites + cu_block_ID*blockDim.x + threadIdx.x )[0] = make_cu_cmplx_PRECISION(0.0,0.0);

  }

}


__global__ void
cuda_block_solve_update_6threads_opt(				cu_cmplx_PRECISION* phi, cu_cmplx_PRECISION* r,
								cu_cmplx_PRECISION* latest_iter, schwarz_PRECISION_struct_on_gpu *s,
								int thread_id, double csw, int kernel_id, int nr_threads_per_DD_block,
								int* DD_blocks_to_compute, int num_latt_site_var, block_struct* block ){

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
      ( latest_iter + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] =
			( tmp2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
    }
  }
  // odd
  if(idx < 6*nr_block_odd_sites){
    for(i=0; i<2; i++){
      ( latest_iter + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] =
		( tmp2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
    }
  }
  // even
  if(idx < 6*nr_block_even_sites){
    for(i=0; i<2; i++){
      ( phi + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] =
		make_cu_cmplx_PRECISION( cu_creal_PRECISION(( phi + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) +
                                         cu_creal_PRECISION(( tmp2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]),
                                         cu_cimag_PRECISION(( phi + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) + 
                                         cu_cimag_PRECISION(( tmp2 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]) );
    }
  }
  // odd
  if(idx < 6*nr_block_odd_sites){
    for(i=0; i<2; i++){
      ( phi + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = 
		make_cu_cmplx_PRECISION( cu_creal_PRECISION(
					 ( phi + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]
					 ) + \
                                         cu_creal_PRECISION(
                                         ( tmp2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]
                                         ),
                                         cu_cimag_PRECISION(
                                         ( phi + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0]
                                         ) + \
                                         cu_cimag_PRECISION(
                                         ( tmp2 + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0])
                                         );
    }
  }
  // even
  if(idx < 6*nr_block_even_sites){
    for(i=0; i<2; i++){
      ( r + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] =
		( tmp3 + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
    }
  }
  // odd
  if(idx < 6*nr_block_odd_sites){
    for(i=0; i<2; i++){
      ( r + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] =
		make_cu_cmplx_PRECISION(0.0,0.0);
    }
  }

}


// WARNING: the use of this function may lead to performance reduction
__global__ void
cuda_block_n_hopping_term_PRECISION(				cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
								schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                     		double csw, int nr_DD_blocks_to_compute, int* DD_blocks_to_compute,
                                                     		int num_latt_site_var, block_struct* block, int sites_to_compute ){

  if( nr_DD_blocks_to_compute==0 ){ return; }

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
    tot_shared_mem = 1*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION);
    for( dir=0; dir<4; dir++ ){
      cuda_block_n_hopping_term_PRECISION_plus_6threads_naive<<< nr_threads/threads_per_cublock,
      								 threads_per_cublock, tot_shared_mem >>>
                                                             ( out, in, s, thread_id, csw, nr_threads_per_DD_block,
                                                               DD_blocks_to_compute, num_latt_site_var, block, dir,
                                                               sites_to_compute
                                                              );
      cuda_block_n_hopping_term_PRECISION_minus_6threads_naive<<< nr_threads/threads_per_cublock,
      								  threads_per_cublock, tot_shared_mem >>>
                                                              ( out, in, s, thread_id, csw, nr_threads_per_DD_block,
                                                                DD_blocks_to_compute, num_latt_site_var, block, dir,
                                                                sites_to_compute
                                                               );
    }
  }
}


// Use of dynamic parallelism to make _D2D copies
__global__ void
cuda_blocks_vector_copy_noncontig_PRECISION_dyn_dev(		cuda_vector_PRECISION out, cuda_vector_PRECISION in,
								int* DD_blocks_to_compute, block_struct* block,
								int num_latt_site_var, schwarz_PRECISION_struct_on_gpu *s ){
                                         
  int block_id, start;

  block_id = DD_blocks_to_compute[blockIdx.x];
  start = block[block_id].start * num_latt_site_var;

  if( threadIdx.x==0 ){
    cudaMemcpyAsync(out + start, in + start, (s->block_vector_size)*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToDevice);
  }

}


__forceinline__ __device__ void
_cuda_block_PRECISION_boundary_op_plus_naive(			cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int block_id,
								schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf,
								int ext_dir, block_struct* block ){

  int i, loc_ind=idx%6, spin, w, *gamma_coo,
      idx_in_cublock = idx%blockDim.x, index, neighbor_index,
      *bbl = s->block_boundary_length;
  cu_cmplx_PRECISION *gamma_val;

  cu_cmplx_PRECISION *buf1, *buf2;
  buf += (idx_in_cublock/6)*12;
  buf1 = buf;
  buf2 = buf + 6;

  cu_config_PRECISION* D = s->op.D;
  cu_config_PRECISION* D_pt;

  cu_cmplx_PRECISION *eta_pt, *phi_pt; //eta and phi are already shifted by 'start'

  i = idx/6;
  i *= 2;
  i += bbl[ 2*ext_dir ];

  index = block[block_id].bt_on_gpu[i];
  neighbor_index = block[block_id].bt_on_gpu[i+1];

  D_pt = D + 36*index + 9*ext_dir;

  phi_pt = phi + 12*neighbor_index;
  eta_pt = eta + 12*index;

  spin = (loc_ind/3)*2;
  //with this setup, gamma_val[0] gives spins 0 and 1, and gamma_val[1] spins 2 and 3
  //gamma_val = s->gamma_info_vals + ext_dir*4 + spin;
  //gamma_coo = s->gamma_info_coo  + ext_dir*4 + spin;
  gamma_val = gamma_info_vals_PRECISION + ext_dir*4 + spin;
  gamma_coo = gamma_info_coo_PRECISION  + ext_dir*4 + spin;

  // prp_T_PRECISION(...)
  buf1[ loc_ind ] = cu_csub_PRECISION( phi_pt[ loc_ind ],
  				       cu_cmul_PRECISION(
  				       gamma_val[0], phi_pt[ 3*gamma_coo[0] + loc_ind%3 ] )
  				      );
  __syncthreads();
  // mvm_PRECISION(...), twice
  buf2[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
  for( w=0; w<3; w++ ){
    buf2[ loc_ind ] = cu_cadd_PRECISION( buf2[ loc_ind ],
    					 cu_cmul_PRECISION(
    					 D_pt[ (loc_ind*3)%9 + w ], buf1[ (loc_ind/3)*3 + w ] )
    					);
  }
  __syncthreads();
  //pbp_su3_T_PRECISION( buf2, eta_pt );
  eta_pt[ loc_ind ] = cu_csub_PRECISION( eta_pt[ loc_ind ], buf2[ loc_ind ] );
  eta_pt[ 6 + loc_ind ] = cu_cadd_PRECISION( eta_pt[ 6 + loc_ind ],
  					     cu_cmul_PRECISION(
  					     gamma_val[1], buf2[ 3*gamma_coo[1] + loc_ind%3 ] )
  					    );
}


__global__ void
cuda_block_PRECISION_boundary_op_plus_naive(			cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
	                                                        schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
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


__forceinline__ __device__ void
_cuda_block_PRECISION_boundary_op_minus_naive(			cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int block_id,
								schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf,
								int ext_dir, block_struct* block ){
  int i, loc_ind=idx%6, spin, w, *gamma_coo,
      idx_in_cublock = idx%blockDim.x, index, neighbor_index,
      *bbl = s->block_boundary_length;
  cu_cmplx_PRECISION *gamma_val;

  cu_cmplx_PRECISION *buf1, *buf2;
  buf += (idx_in_cublock/6)*12;
  buf1 = buf;
  buf2 = buf + 6;

  cu_config_PRECISION* D = s->op.D;
  cu_config_PRECISION* D_pt;

  cu_cmplx_PRECISION *eta_pt, *phi_pt; //eta and phi are already shifted by 'start'

  i = idx/6;
  i *= 2;
  i += bbl[ 2*ext_dir + 1 ];

  index = block[block_id].bt_on_gpu[i];
  neighbor_index = block[block_id].bt_on_gpu[i+1];

  D_pt = D + 36*neighbor_index + 9*ext_dir;

  phi_pt = phi + 12*neighbor_index;
  eta_pt = eta + 12*index;

  spin = (loc_ind/3)*2;
  //with this setup, gamma_val[0] gives spins 0 and 1, and gamma_val[1] spins 2 and 3
  //gamma_val = s->gamma_info_vals + ext_dir*4 + spin;
  //gamma_coo = s->gamma_info_coo  + ext_dir*4 + spin;
  gamma_val = gamma_info_vals_PRECISION + ext_dir*4 + spin;
  gamma_coo = gamma_info_coo_PRECISION  + ext_dir*4 + spin;

  // prn_T_PRECISION(...)
  buf1[ loc_ind ] = cu_cadd_PRECISION( phi_pt[ loc_ind ],
  				       cu_cmul_PRECISION(
  				       gamma_val[0], phi_pt[ 3*gamma_coo[0] + loc_ind%3 ] )
  				      );

  __syncthreads();

  // mvmh_PRECISION(...), twice
  buf2[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
  for( w=0; w<3; w++ ){
    buf2[ loc_ind ] = cu_cadd_PRECISION( buf2[ loc_ind ],
    					 cu_cmul_PRECISION(
    					 cu_conj_PRECISION(D_pt[ loc_ind%3 + w*3 ]), buf1[ (loc_ind/3)*3 + w ] )
    					);
  }

  __syncthreads();

  //pbn_su3_T_PRECISION( buf2, eta_pt );
  eta_pt[ loc_ind ] = cu_csub_PRECISION( eta_pt[ loc_ind ], buf2[ loc_ind ] );
  eta_pt[ 6 + loc_ind ] = cu_csub_PRECISION( eta_pt[ 6 + loc_ind ],
  					     cu_cmul_PRECISION(
  					     gamma_val[1], buf2[ 3*gamma_coo[1] + loc_ind%3 ] )
  					    );
}


__global__ void
cuda_block_PRECISION_boundary_op_minus_naive(			cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
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


// TODO: move the following boundary-related (host) functions to cuda_schwarz_generic.c ?

__forceinline__ __device__ void
_cuda_n_block_PRECISION_boundary_op_plus_naive(			cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int block_id,
								schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf,
								int ext_dir, block_struct* block ){
  int i, loc_ind=idx%6, spin, w, *gamma_coo,
      idx_in_cublock = idx%blockDim.x, index, neighbor_index,
      *bbl = s->block_boundary_length;
  cu_cmplx_PRECISION *gamma_val;

  cu_cmplx_PRECISION *buf1, *buf2;
  buf += (idx_in_cublock/6)*12;
  buf1 = buf;
  buf2 = buf + 6;

  cu_config_PRECISION* D = s->op.D;
  cu_config_PRECISION* D_pt;

  cu_cmplx_PRECISION *eta_pt, *phi_pt; //eta and phi are already shifted by 'start'

  i = idx/6;
  i *= 2;
  i += bbl[ 2*ext_dir ];

  index = block[block_id].bt_on_gpu[i];
  neighbor_index = block[block_id].bt_on_gpu[i+1];

  D_pt = D + 36*index + 9*ext_dir;

  phi_pt = phi + 12*neighbor_index;
  eta_pt = eta + 12*index;

  spin = (loc_ind/3)*2;
  //with this setup, gamma_val[0] gives spins 0 and 1, and gamma_val[1] spins 2 and 3
  //gamma_val = s->gamma_info_vals + ext_dir*4 + spin;
  //gamma_coo = s->gamma_info_coo  + ext_dir*4 + spin;
  gamma_val = gamma_info_vals_PRECISION + ext_dir*4 + spin;
  gamma_coo = gamma_info_coo_PRECISION  + ext_dir*4 + spin;

  // prp_T_PRECISION(...)
  buf1[ loc_ind ] = cu_csub_PRECISION( phi_pt[ loc_ind ],
  				       cu_cmul_PRECISION(
  				       gamma_val[0], phi_pt[ 3*gamma_coo[0] + loc_ind%3 ] )
  				      );
  __syncthreads();
  // nmvm_PRECISION(...), twice
  buf2[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
  for( w=0; w<3; w++ ){
    buf2[ loc_ind ] = cu_csub_PRECISION( buf2[ loc_ind ],
    					 cu_cmul_PRECISION(
    					 D_pt[ (loc_ind*3)%9 + w ], buf1[ (loc_ind/3)*3 + w ] )
    					);
  }
  __syncthreads();
  //pbp_su3_T_PRECISION( buf2, eta_pt );
  eta_pt[ loc_ind ] = cu_csub_PRECISION( eta_pt[ loc_ind ], buf2[ loc_ind ] );
  eta_pt[ 6 + loc_ind ] = cu_cadd_PRECISION( eta_pt[ 6 + loc_ind ],
  					     cu_cmul_PRECISION(
  					     gamma_val[1], buf2[ 3*gamma_coo[1] + loc_ind%3 ] )
  					    );
}


__forceinline__ __device__ void
_cuda_n_block_PRECISION_boundary_op_minus_naive(		cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int block_id,
								schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf,
								int ext_dir, block_struct* block ){

  int i, loc_ind=idx%6, spin, w, *gamma_coo,
      idx_in_cublock = idx%blockDim.x, index, neighbor_index,
      *bbl = s->block_boundary_length;
  cu_cmplx_PRECISION *gamma_val;

  cu_cmplx_PRECISION *buf1, *buf2;
  buf += (idx_in_cublock/6)*12;
  buf1 = buf;
  buf2 = buf + 6;

  cu_config_PRECISION* D = s->op.D;
  cu_config_PRECISION* D_pt;

  cu_cmplx_PRECISION *eta_pt, *phi_pt; //eta and phi are already shifted by 'start'

  i = idx/6;
  i *= 2;
  i += bbl[ 2*ext_dir + 1 ];

  index = block[block_id].bt_on_gpu[i];
  neighbor_index = block[block_id].bt_on_gpu[i+1];

  D_pt = D + 36*neighbor_index + 9*ext_dir;

  phi_pt = phi + 12*neighbor_index;
  eta_pt = eta + 12*index;

  spin = (loc_ind/3)*2;
  //with this setup, gamma_val[0] gives spins 0 and 1, and gamma_val[1] spins 2 and 3
  //gamma_val = s->gamma_info_vals + ext_dir*4 + spin;
  //gamma_coo = s->gamma_info_coo  + ext_dir*4 + spin;
  gamma_val = gamma_info_vals_PRECISION + ext_dir*4 + spin;
  gamma_coo = gamma_info_coo_PRECISION  + ext_dir*4 + spin;

  // prn_T_PRECISION(...)
  buf1[ loc_ind ] = cu_cadd_PRECISION( phi_pt[ loc_ind ],
  				       cu_cmul_PRECISION(
  				       gamma_val[0], phi_pt[ 3*gamma_coo[0] + loc_ind%3 ] )
  				      );

  __syncthreads();

  // nmvmh_PRECISION(...), twice
  buf2[ loc_ind ] = make_cu_cmplx_PRECISION(0.0,0.0);
  for( w=0; w<3; w++ ){
    //buf2[ loc_ind ] = cu_cadd_PRECISION( buf2[ loc_ind ], cu_cmul_PRECISION(
    //cu_conj_PRECISION(D_pt[ loc_ind%3 + w*3 ]), buf1[ (loc_ind/3)*3 + w ] ) );
    buf2[ loc_ind ] = cu_csub_PRECISION( buf2[ loc_ind ],
    					 cu_cmul_PRECISION(
    					 cu_conj_PRECISION(D_pt[ loc_ind%3 + w*3 ]), buf1[ (loc_ind/3)*3 + w ] )
    					);
  }

  __syncthreads();

  //pbn_su3_T_PRECISION( buf2, eta_pt );
  eta_pt[ loc_ind ] = cu_csub_PRECISION( eta_pt[ loc_ind ], buf2[ loc_ind ] );
  eta_pt[ 6 + loc_ind ] = cu_csub_PRECISION( eta_pt[ 6 + loc_ind ],
  					     cu_cmul_PRECISION(
  					     gamma_val[1], buf2[ 3*gamma_coo[1] + loc_ind%3 ] )
  					    );
}


/*
__device__ void
_cuda_block_d_plus_clover_PRECISION_6threads_naive(		cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
								schwarz_PRECISION_struct_on_gpu *s, int idx, cu_cmplx_PRECISION *buf,
								int ext_dir ){

  int k=0, j=0, i=0, **index = s->index, *ind,
      *neighbor = s->op.neighbor_table, loc_ind=idx%6, spin, w, *gamma_coo,
      idx_in_cublock = idx%blockDim.x;
  cu_cmplx_PRECISION *gamma_val;

  cu_cmplx_PRECISION *buf1, *buf2, *buf3, *buf4;
  buf += (idx_in_cublock/6)*24;
  buf1 = buf + 0*6;
  buf2 = buf + 1*6;
  buf3 = buf + 2*6;
  buf4 = buf + 3*6;

  cu_config_PRECISION* D = s->op.D + (start/12)*36;
  cu_config_PRECISION* D_pt;

  cu_cmplx_PRECISION *leta, *lphi; //eta and phi are already shifted by 'start'

  ind = index[ext_dir];

  i = idx/6;
  k = ind[i];
  j = neighbor[4*k+ext_dir];
  D_pt = D + 36*k + 9*ext_dir;

  // already added <start> to the original input spinors
  lphi = phi;
  leta = eta;

  spin = (loc_ind/3)*2;
  //with this setup, gamma_val[0] gives spins 0 and 1, and gamma_val[1] spins 2 and 3
  //gamma_val = s->gamma_info_vals + ext_dir*4 + spin;
  //gamma_coo = s->gamma_info_coo  + ext_dir*4 + spin;
  gamma_val = gamma_info_vals_PRECISION + ext_dir*4 + spin;
  gamma_coo = gamma_info_coo_PRECISION  + ext_dir*4 + spin;

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
*/


__forceinline__ __device__ void
_cuda_site_clover_PRECISION(					cu_cmplx_PRECISION *eta, cu_cmplx_PRECISION *phi, int start,
                                                                schwarz_PRECISION_struct_on_gpu *s, int idx,
                                                                cu_config_PRECISION *op_clov, double csw ){
  int local_idx = idx%6;
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
        eta_site[local_idx] = cu_cadd_PRECISION( eta_site[local_idx],
        					 cu_cmul_PRECISION(
        					 op_clov_site[matrx_indx], phi_site[i + k*6] )
        					);
      }
      else if( (i+k*6)<local_idx ){
        matrx_indx = 12 +15*k + (14 - ( (5-i-1)*(5-i-1 + 1)/2 + (5-local_idx%6) ));
        eta_site[local_idx] = cu_cadd_PRECISION( eta_site[local_idx],
        					 cu_cmul_PRECISION(
        					 cu_conj_PRECISION( op_clov_site[matrx_indx] ), phi_site[i + k*6] )
        					);
      }
      else{ // i==local_idx
        eta_site[local_idx] = cu_cadd_PRECISION( eta_site[local_idx],
        					 cu_cmul_PRECISION(
        					 op_clov_site[local_idx], phi_site[local_idx] )
        					);
      }
    }
    local_idx += 6;
  }
}


// FIXME:
//		1. use shared memory to have better global memory accesses
//		2. switch to an stream different from <default> ... ?
__global__ void
_cuda_boundary_comms_copy_PRECISION(				cu_cmplx_PRECISION *out, cu_cmplx_PRECISION *in,
								int *bound_table, int site_size ){

  int j, idx, site_id;
  cu_cmplx_PRECISION *in_pt;

  idx = threadIdx.x + blockDim.x * blockIdx.x;
  j = idx/12;

  site_id = bound_table[j];

  in_pt = in + site_id*site_size;
  in_pt += idx%12;

  out[idx] = in_pt[0];
}


__global__ void
cuda_n_block_PRECISION_boundary_op_plus_naive(			cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
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


__global__ void
cuda_n_block_PRECISION_boundary_op_minus_naive(			cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
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


__global__ void
cuda_site_clover_PRECISION(					cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
			                                        schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                        		                        double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                            			int num_latt_site_var, block_struct* block ){

  // IMPORTANT: each component of each site is independent, and so, 12 threads could be used per
  //            lattice site, but here we use 6 due to loading s->op.clover into shared memory
  //            collaboratively by all the threads

  int i, idx, DD_block_id, block_id, cublocks_per_DD_block, cu_block_ID, size_D_clov, start, nr_D_dumps;

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
  size_D_clov = (csw!=0.0) ? 42 : 12;

  // this is the DD-block start of the spinors (phi, r, latest_iter and temporary ones)
  start = block[block_id].start * num_latt_site_var;

  out += start;
  in += start;

  cu_config_PRECISION *op = s->op.clover_gpustorg;
  //cu_config_PRECISION *op = s->op.oe_clover_gpustorg;
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

  // odd
  //if(idx < 2*nr_block_odd_sites){
    for(i=0; i<2; i++){
      in_o[blockDim.x*i + threadIdx.x] = ( in + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
    }

    for(i=0; i<2; i++){
      //out_o[blockDim.x*i + threadIdx.x] = ( out + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
    }

    // The factor of 12 comes from (the factor of 16 comes from the nr of lattice sites
    // per CUDA block: 96/6 = 16): 72*16=1152 ----> 1152/96=12. This implies 12 dumps of data into shared_memory
    nr_D_dumps = (blockDim.x/6)*size_D_clov/blockDim.x; // = 7 always
    //nr_D_dumps /= nr_D_dumps;
    for(i=0; i<nr_D_dumps; i++){
      clov_o[blockDim.x*i + threadIdx.x] = ( op + cu_block_ID*blockDim.x*nr_D_dumps + blockDim.x*i + threadIdx.x )[0];
    }
  //}

  __syncthreads();

  i = idx/6;

  _cuda_site_clover_PRECISION(out_o, in_o, start, s, idx, clov_o, csw);

  __syncthreads();

  //if(idx < 2*nr_block_odd_sites){
    for(i=0; i<2; i++){
      ( out + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = out_o[blockDim.x*i + threadIdx.x];
    }
  //}

  __syncthreads();

  i = idx/6;
}


__global__ void
cuda_clover_diag_PRECISION(                                     cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in, \
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id, \
                                                                int nr_threads_per_DD_block, int* DD_blocks_to_compute, \
                                                                int num_latt_site_var, block_struct* block ){

  int idx, DD_block_id, block_id, size_D_clov, start;

  idx = threadIdx.x + blockDim.x * blockIdx.x;

  // not really a DD block id, but rather a linear counting of a grouping (per DD block) of CUDA threads
  DD_block_id = idx/nr_threads_per_DD_block;

  // offsetting idx to make it zero at the beginning of the threads living within a DD block
  idx = idx%nr_threads_per_DD_block;

  // this int will be the ACTUAL DD block ID, in the sense of accessing data from e.g. block_struct* block
  block_id = DD_blocks_to_compute[DD_block_id];

  //cublocks_per_DD_block = nr_threads_per_DD_block/blockDim.x;

  // the size of the local matrix to apply
  size_D_clov = 12;

  // this is the DD-block start of the spinors (phi, r, latest_iter and temporary ones)
  start = block[block_id].start * num_latt_site_var;

  out += start;
  in += start;

  cu_config_PRECISION *op = s->op.clover_gpustorg;
  op += (start/num_latt_site_var)*size_D_clov;

  // main operation of this kernel : out[idx] = in[idx]*op[idx]
  out[idx] = cu_cmul_PRECISION( in[idx], op[idx] );

}


//-----------------------------------------------------------------------------------------------------------
// host functions


__global__ void
cuda_block_diag_oo_inv_PRECISION_6threads_opt(			cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block ){
  int i, idx, DD_block_id, block_id, cublocks_per_DD_block, cu_block_ID, size_D_oeclov, start, nr_D_dumps;
  int nr_block_even_sites, nr_block_odd_sites;

  cu_cmplx_PRECISION *shared_data_loc, *in_o, *out_o, *clov_vect_b_o;

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
  //size_D_oeclov = (csw!=0.0) ? 72 : 12;
  size_D_oeclov = (csw!=0.0) ? 42 : 12;

  // this is the DD-block start of the spinors (phi, r, latest_iter and temporary ones)
  start = block[block_id].start * num_latt_site_var;

  out += start;
  in += start;

  if( csw != 0 ){

    // this operator is stored in column form!
    //cu_config_PRECISION *op_oe_vect = s->op.oe_clover_vectorized;
    cu_config_PRECISION *op_oe_vect = s->op.oe_clover_gpustorg;
    // FIXME: instead of 12, use num_latt_site_var
    op_oe_vect += (start/12)*size_D_oeclov;

    extern __shared__ cu_cmplx_PRECISION shared_data[];

    shared_data_loc = shared_data;

    in_o = shared_data_loc + 0*(2*blockDim.x);
    out_o = shared_data_loc + 1*(2*blockDim.x);

    clov_vect_b_o = (cu_cmplx_PRECISION*)((cu_cmplx_PRECISION*)shared_data_loc + 2*(2*blockDim.x));

    // odd
    if(idx < 6*nr_block_odd_sites){
      for(i=0; i<2; i++){
        in_o[blockDim.x*i + threadIdx.x] = ( in + 12*nr_block_even_sites +
                                             cu_block_ID*blockDim.x*2 +
                                             blockDim.x*i + threadIdx.x )[0];
      }
      // The factor of 12 comes from (the factor of 16 comes from the nr of lattice sites per CUDA block:
      // 96/6 = 16): 72*16=1152 ----> 1152/96=12. This implies 12 dumps of data into shared_memory
      nr_D_dumps = (blockDim.x/6)*size_D_oeclov/blockDim.x; // = 7 always
      //nr_D_dumps /= nr_D_dumps;
      for(i=0; i<nr_D_dumps; i++){
        clov_vect_b_o[blockDim.x*i + threadIdx.x] = ( op_oe_vect +
                                                      42*nr_block_even_sites +
                                                      cu_block_ID*blockDim.x*nr_D_dumps +
                                                      blockDim.x*i + threadIdx.x )[0];
      }
    }
    __syncthreads();
    // FUNCTION: chi = D_{oo}^{-1} * eta_{0}
    if(idx < 6*nr_block_odd_sites){
      _cuda_block_diag_oo_inv_PRECISION_6threads_opt(out_o, in_o, start, s, idx, clov_vect_b_o, csw);
    }
    __syncthreads();
    // odd
    if(idx < 6*nr_block_odd_sites){
      for(i=0; i<2; i++){
        ( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] =
          			out_o[blockDim.x*i + threadIdx.x];
      }
    }
  }
  else{
    // this operator is stored in column form!
    //cu_config_PRECISION *op_oe_vect = s->op.oe_clover_vectorized;
    cu_config_PRECISION *op_oe_vect = s->op.oe_clover_gpustorg;
    op_oe_vect += (start/num_latt_site_var)*size_D_oeclov;

    // offsetting to the <odd> portion of the DD block
    in += 12*nr_block_even_sites;
    out += 12*nr_block_even_sites;
    op_oe_vect += 12*nr_block_even_sites;

    // shift idx to live at the beginning of the (1/6-th)-of-site for the threads's corresponding site
    idx *= 2;

    for( i=0; i<2; i++ ){
      out[idx+i] = cu_cdiv_PRECISION( in[idx+i], op_oe_vect[idx+i] );
    }

  }
}


__global__ void
cuda_block_diag_oo_inv_PRECISION_2threads_opt(			cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block ){
  int i, idx, DD_block_id, block_id, cublocks_per_DD_block, cu_block_ID, size_D_oeclov, start, nr_D_dumps;
  int nr_block_even_sites, nr_block_odd_sites;

  cu_cmplx_PRECISION *shared_data_loc, *in_o, *out_o, *clov_vect_b_o;

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
  //size_D_oeclov = (csw!=0.0) ? 72 : 12;
  size_D_oeclov = (csw!=0.0) ? 42 : 12;

  // this is the DD-block start of the spinors (phi, r, latest_iter and temporary ones)
  start = block[block_id].start * num_latt_site_var;

  out += start;
  in += start;

  if( csw != 0.0 ){

    // this operator is stored in column form!
    //cu_config_PRECISION *op_oe_vect = s->op.oe_clover_vectorized;
    cu_config_PRECISION *op_oe_vect = s->op.oe_clover_gpustorg;
    // FIXME: instead of 12, use num_latt_site_var
    op_oe_vect += (start/12)*size_D_oeclov;

    extern __shared__ cu_cmplx_PRECISION shared_data[];

    shared_data_loc = shared_data;

    in_o = shared_data_loc + 0*(6*blockDim.x);
    out_o = shared_data_loc + 1*(6*blockDim.x);

    clov_vect_b_o = (cu_cmplx_PRECISION*)((cu_cmplx_PRECISION*)shared_data_loc + 2*(6*blockDim.x));

    // odd
    if(idx < 2*nr_block_odd_sites){
      for(i=0; i<6; i++){
        in_o[blockDim.x*i + threadIdx.x] = ( in + 12*nr_block_even_sites +
                                             cu_block_ID*blockDim.x*6 +
                                             blockDim.x*i + threadIdx.x )[0];
      }
      // the factor of 12 comes from (the factor of 16 comes from the nr of lattice sites per CUDA
      // block: 96/6 = 16): 72*16=1152 ----> 1152/96=12. This implies 12 dumps of data into shared_memory
      nr_D_dumps = (blockDim.x/2)*size_D_oeclov/blockDim.x; // = 21 always
      //nr_D_dumps /= nr_D_dumps;
      for(i=0; i<nr_D_dumps; i++){
        clov_vect_b_o[blockDim.x*i + threadIdx.x] = ( op_oe_vect + 42*nr_block_even_sites +
                                                      cu_block_ID*blockDim.x*nr_D_dumps +
                                                      blockDim.x*i + threadIdx.x )[0];
      }
    }

    if( blockDim.x>32 ) __syncthreads();

    // FUNCTION: chi = D_{oo}^{-1} * eta_{0}
    if(idx < 2*nr_block_odd_sites){
      _cuda_block_diag_oo_inv_PRECISION_2threads_opt(out_o, in_o, start, s, idx, clov_vect_b_o, csw);
    }

    if( blockDim.x>32 ) __syncthreads();

    // odd
    if(idx < 2*nr_block_odd_sites){
      for(i=0; i<6; i++){
        ( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x*6 + blockDim.x*i + threadIdx.x )[0] =
        					out_o[blockDim.x*i + threadIdx.x];
      }
    }

  }
  else{
    // this operator is stored in column form!
    //cu_config_PRECISION *op_oe_vect = s->op.oe_clover_vectorized;
    cu_config_PRECISION *op_oe_vect = s->op.oe_clover_gpustorg;
    op_oe_vect += (start/num_latt_site_var)*size_D_oeclov;

    // offsetting to the <odd> portion of the DD block
    in += 12*nr_block_even_sites;
    out += 12*nr_block_even_sites;
    op_oe_vect += 12*nr_block_even_sites;

    // shift idx to live at the beginning of the half-site for the threads's corresponding site
    idx *= 6;

    for( i=0; i<6; i++ ){
      out[idx+i] = cu_cdiv_PRECISION( in[idx+i], op_oe_vect[idx+i] );
    }

  }

}


__global__ void
cuda_block_diag_ee_PRECISION_6threads_opt(			cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
       	                                                        double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block ){
  int i, idx, DD_block_id, block_id, cublocks_per_DD_block, cu_block_ID, size_D_oeclov, start, nr_D_dumps;
  int nr_block_odd_sites;

  cu_cmplx_PRECISION *shared_data_loc, *in_o, *out_o, *clov_vect_b_o;

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
  //size_D_oeclov = (csw!=0.0) ? 72 : 12;
  size_D_oeclov = (csw!=0.0) ? 42 : 12;

  // this is the DD-block start of the spinors (phi, r, latest_iter and temporary ones)
  start = block[block_id].start * num_latt_site_var;

  out += start;
  in += start;

  if( csw!=0 ){

    // this operator is stored in column form!
    //cu_config_PRECISION *op_oe_vect = s->op.oe_clover_vectorized;
    cu_config_PRECISION *op_oe_vect = s->op.oe_clover_gpustorg;
    // FIXME: instead of 12, use num_latt_site_var
    op_oe_vect += (start/12)*size_D_oeclov;

    extern __shared__ cu_cmplx_PRECISION shared_data[];

    shared_data_loc = shared_data;

    in_o = shared_data_loc + 0*(2*blockDim.x);
    out_o = shared_data_loc + 1*(2*blockDim.x);

    clov_vect_b_o = (cu_cmplx_PRECISION*)((cu_cmplx_PRECISION*)shared_data_loc + 2*(2*blockDim.x));

    // odd
    if(idx < 6*nr_block_odd_sites){
      for(i=0; i<2; i++){
        //in_o[blockDim.x*i + threadIdx.x] = ( in + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 +
        // blockDim.x*i + threadIdx.x )[0];
        in_o[blockDim.x*i + threadIdx.x] = ( in + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0];
      }
      // the factor of 12 comes from (the factor of 16 comes from the nr of lattice sites per CUDA block: 96/6 = 16):
      // 72*16=1152 ----> 1152/96=12. This implies 12 dumps of data into shared_memory
      nr_D_dumps = (blockDim.x/6)*size_D_oeclov/blockDim.x; // = 7 always
      //nr_D_dumps /= nr_D_dumps;
      for(i=0; i<nr_D_dumps; i++){
        //clov_vect_b_o[blockDim.x*i + threadIdx.x] = ( op_oe_vect + 42*nr_block_even_sites +
        // cu_block_ID*blockDim.x*nr_D_dumps + blockDim.x*i + threadIdx.x )[0];
        clov_vect_b_o[blockDim.x*i + threadIdx.x] = ( op_oe_vect + cu_block_ID*blockDim.x*nr_D_dumps +
                                                      blockDim.x*i + threadIdx.x )[0];
      }
    }
    __syncthreads();
    // FUNCTION: chi = D_{oo}^{-1} * eta_{0}
    if(idx < 6*nr_block_odd_sites){
      _cuda_block_diag_ee_PRECISION_6threads_opt(out_o, in_o, start, s, idx, clov_vect_b_o, csw);
    }
    __syncthreads();
    // odd
    if(idx < 6*nr_block_odd_sites){
      for(i=0; i<2; i++){
        //( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] =
        // out_o[blockDim.x*i + threadIdx.x];
        ( out + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] = out_o[blockDim.x*i + threadIdx.x];
      }
    }
  }
  else{

    // this operator is stored in column form!
    //cu_config_PRECISION *op_oe_vect = s->op.oe_clover_vectorized;
    cu_config_PRECISION *op_oe_vect = s->op.oe_clover_gpustorg;
    op_oe_vect += (start/num_latt_site_var)*size_D_oeclov;

    // shift idx to live at the beginning of the (1/6-th)-of-site for the threads's corresponding site
    idx *= 2;

    for( i=0; i<2; i++ ){
      out[idx+i] = cu_cmul_PRECISION( in[idx+i], op_oe_vect[idx+i] );
    }


  }
}


__global__ void
cuda_block_diag_ee_PRECISION_2threads_opt(			cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block ){
  int i, idx, DD_block_id, block_id, cublocks_per_DD_block, cu_block_ID, size_D_oeclov, start, nr_D_dumps;
  int nr_block_odd_sites;

  cu_cmplx_PRECISION *shared_data_loc, *in_o, *out_o, *clov_vect_b_o;

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
  //size_D_oeclov = (csw!=0.0) ? 72 : 12;
  size_D_oeclov = (csw!=0.0) ? 42 : 12;

  // this is the DD-block start of the spinors (phi, r, latest_iter and temporary ones)
  start = block[block_id].start * num_latt_site_var;

  out += start;
  in += start;

  if( csw != 0.0 ){

    // this operator is stored in column form!
    //cu_config_PRECISION *op_oe_vect = s->op.oe_clover_vectorized;
    cu_config_PRECISION *op_oe_vect = s->op.oe_clover_gpustorg;
    // FIXME: instead of 12, use num_latt_site_var
    op_oe_vect += (start/12)*size_D_oeclov;

    extern __shared__ cu_cmplx_PRECISION shared_data[];

    shared_data_loc = shared_data;

    in_o = shared_data_loc + 0*(6*blockDim.x);
    out_o = shared_data_loc + 1*(6*blockDim.x);

    clov_vect_b_o = (cu_cmplx_PRECISION*)((cu_cmplx_PRECISION*)shared_data_loc + 2*(6*blockDim.x));

    // odd
    if(idx < 2*nr_block_odd_sites){
      for(i=0; i<6; i++){
        //in_o[blockDim.x*i + threadIdx.x] = ( in + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 +
        // blockDim.x*i + threadIdx.x )[0];
        in_o[blockDim.x*i + threadIdx.x] = ( in + cu_block_ID*blockDim.x*6 + blockDim.x*i + threadIdx.x )[0];
      }
      // the factor of 12 comes from (the factor of 16 comes from the nr of lattice sites per CUDA block: 96/6 = 16):
      // 72*16=1152 ----> 1152/96=12. This implies 12 dumps of data into shared_memory
      nr_D_dumps = (blockDim.x/2)*size_D_oeclov/blockDim.x; // = 7 always
      //nr_D_dumps /= nr_D_dumps;
      for(i=0; i<nr_D_dumps; i++){
        //clov_vect_b_o[blockDim.x*i + threadIdx.x] = ( op_oe_vect + 42*nr_block_even_sites +
        // cu_block_ID*blockDim.x*nr_D_dumps + blockDim.x*i + threadIdx.x )[0];
        clov_vect_b_o[blockDim.x*i + threadIdx.x] = ( op_oe_vect + cu_block_ID*blockDim.x*nr_D_dumps +
                                                      blockDim.x*i + threadIdx.x )[0];
      }
    }

    if(blockDim.x>32) __syncthreads();

    // FUNCTION: chi = D_{oo}^{-1} * eta_{0}
    if(idx < 2*nr_block_odd_sites){
      _cuda_block_diag_ee_PRECISION_2threads_opt(out_o, in_o, start, s, idx, clov_vect_b_o, csw);
    }

    if(blockDim.x>32) __syncthreads();

    // update tmp2 and tmp3
    // odd
    if(idx < 2*nr_block_odd_sites){
      for(i=0; i<6; i++){
        //( out + 12*nr_block_even_sites + cu_block_ID*blockDim.x*2 + blockDim.x*i + threadIdx.x )[0] =
        // out_o[blockDim.x*i + threadIdx.x];
        ( out + cu_block_ID*blockDim.x*6 + blockDim.x*i + threadIdx.x )[0] = out_o[blockDim.x*i + threadIdx.x];
      }
    }

  }
  else{

    cu_config_PRECISION *op_oe_vect = s->op.oe_clover_gpustorg;
    op_oe_vect += (start/num_latt_site_var)*size_D_oeclov;

    idx *= 6;

    for( i=0; i<6; i++ ){
      out[idx+i] = cu_cmul_PRECISION( in[idx+i], op_oe_vect[idx+i] );
    }

  }

}


// gamma_val and gamma_coo are both loaded into shared memory for this kernel
__global__ void
cuda_block_n_hopping_term_PRECISION_plus_6threads_naive(	cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                        	schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block, int ext_dir, int amount ){
  int idx, *gamma_coo, DD_block_id, block_id, start;
  cu_cmplx_PRECISION *gamma_val, *shared_data, *tmp_loc;

  idx = threadIdx.x + blockDim.x * blockIdx.x;

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
  shared_data = shared_data_bare;

  gamma_coo = (int*)shared_data;
  shared_data = (cu_cmplx_PRECISION*)((int*)shared_data + 16);

  gamma_val = shared_data;
  shared_data = shared_data + 16;

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
  _cuda_block_n_hopping_term_PRECISION_plus_6threads_naive( out, in, start, amount, s, idx,
                                                            tmp_loc, ext_dir, gamma_coo, gamma_val );
}


// gamma_val and gamma_coo are both loaded into shared memory for this kernel
__global__ void
cuda_block_n_hopping_term_PRECISION_plus_2threads_opt(          cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block, int ext_dir,
                                                                int amount ){
  int idx, i, *gamma_coo, start, int_offset_Dgpu, sites_offset_Dgpu, nr_dumps_Dgpu;
  cu_cmplx_PRECISION *gamma_val, *Dgpu_local, *tmp_loc, *Dgpu;

  idx = threadIdx.x + blockDim.x * blockIdx.x;

  DD_gateway_hopping( s, DD_blocks_to_compute, nr_threads_per_DD_block, num_latt_site_var, &idx, &start, &int_offset_Dgpu,
                      &sites_offset_Dgpu, &out, &in, &nr_dumps_Dgpu, &gamma_coo, &gamma_val, &tmp_loc, &Dgpu_local, amount,
                      block, ext_dir, 2 );

  // due to computing the _plus_ kernel
  int_offset_Dgpu += 0*8;
  if( amount==_EVEN_SITES ){
    int_offset_Dgpu += 0*4;
  }
  else if( amount==_ODD_SITES ){
    int_offset_Dgpu += 1*4;
  }
  else{
    // TODO
  }

  // each site-and-dir of Dgpu has 9 entries
  // created offset of Dgpu
  Dgpu = s->op.Dgpu[int_offset_Dgpu] + sites_offset_Dgpu*9;
  for( i=0; i<nr_dumps_Dgpu; i++ ){
    // not all threads are needed when loading Dgpu
    if( (i*blockDim.x + idx%blockDim.x) < ((blockDim.x/2)*9) ){
      Dgpu_local[ i*blockDim.x + threadIdx.x ] = Dgpu[ i*blockDim.x + threadIdx.x ];
    }
  }
  if(blockDim.x>32) __syncthreads();
  _cuda_block_n_hopping_term_PRECISION_plus_2threads_opt( out, in, start, amount, s, idx,
                                                          tmp_loc, ext_dir, Dgpu_local, gamma_coo, gamma_val );
}


// gamma_val and gamma_coo are both loaded into shared memory for this kernel
__global__ void
cuda_block_n_hopping_term_PRECISION_plus_6threads_opt(		cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block, int ext_dir,
                                                                int amount ){
  int idx, i, *gamma_coo, start, int_offset_Dgpu, sites_offset_Dgpu, nr_dumps_Dgpu;
  cu_cmplx_PRECISION *gamma_val, *Dgpu_local, *tmp_loc, *Dgpu;

  idx = threadIdx.x + blockDim.x * blockIdx.x;

  DD_gateway_hopping( s, DD_blocks_to_compute, nr_threads_per_DD_block, num_latt_site_var, &idx, &start, &int_offset_Dgpu,
                      &sites_offset_Dgpu, &out, &in, &nr_dumps_Dgpu, &gamma_coo, &gamma_val, &tmp_loc, &Dgpu_local, amount,
                      block, ext_dir, 6 );

  // due to computing the _plus_ kernel
  int_offset_Dgpu += 0*8;
  if( amount==_EVEN_SITES ){
    int_offset_Dgpu += 0*4;
  }
  else if( amount==_ODD_SITES ){
    int_offset_Dgpu += 1*4;
  }
  else{
    // TODO
  }

  // each site-and-dir of Dgpu has 9 entries
  // created offset of Dgpu
  Dgpu = s->op.Dgpu[int_offset_Dgpu] + sites_offset_Dgpu*9;
  for( i=0; i<nr_dumps_Dgpu; i++ ){
    // not all threads are needed when loading Dgpu
    if( (i*blockDim.x + idx%blockDim.x) < ((blockDim.x/6)*9) ){
      Dgpu_local[ i*blockDim.x + threadIdx.x ] = Dgpu[ i*blockDim.x + threadIdx.x ];
    }
  }
  __syncthreads();
  _cuda_block_n_hopping_term_PRECISION_plus_6threads_opt( out, in, start, amount, s, idx,
                                                          tmp_loc, ext_dir, Dgpu_local, gamma_coo, gamma_val );
}


// gamma_val and gamma_coo are both loaded into shared memory for this kernel
__global__ void
cuda_block_n_hopping_term_PRECISION_minus_6threads_naive(	cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block, int ext_dir,
                                                                int amount ){
  int idx, DD_block_id, block_id, start, nr_block_even_sites;
  cu_cmplx_PRECISION *tmp_loc, *shared_data_loc;

  idx = threadIdx.x + blockDim.x * blockIdx.x;
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
  shared_data_loc = shared_data;

  //the following are 'bare' values, i.e. with respect to the 0th element within a CUDA block
  // EVEN
  tmp_loc = shared_data_loc;
  if(idx < 6*nr_block_even_sites){
    tmp_loc[threadIdx.x] = make_cu_cmplx_PRECISION(0.0,0.0);
    tmp_loc[threadIdx.x + blockDim.x] = make_cu_cmplx_PRECISION(0.0,0.0);
  } //even
  _cuda_block_n_hopping_term_PRECISION_minus_6threads_naive(out, in, start, amount, s, idx, tmp_loc, ext_dir);
}


// gamma_val and gamma_coo are both loaded into shared memory for this kernel
__global__ void
cuda_block_n_hopping_term_PRECISION_minus_2threads_opt(         cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block, int ext_dir,
                                                                int amount ){
  int idx, i, *gamma_coo, start, int_offset_Dgpu, sites_offset_Dgpu, nr_dumps_Dgpu;
  cu_cmplx_PRECISION *gamma_val, *Dgpu_local, *tmp_loc, *Dgpu;

  idx = threadIdx.x + blockDim.x * blockIdx.x;

  DD_gateway_hopping( s, DD_blocks_to_compute, nr_threads_per_DD_block, num_latt_site_var, &idx, &start, &int_offset_Dgpu,
                      &sites_offset_Dgpu, &out, &in, &nr_dumps_Dgpu, &gamma_coo, &gamma_val, &tmp_loc, &Dgpu_local, amount,
                      block, ext_dir, 2 );

  int_offset_Dgpu = ext_dir;
  // due to computing the _minus_ kernel
  int_offset_Dgpu += 1*8;
  if( amount==_EVEN_SITES ){
    int_offset_Dgpu += 0*4;
  }
  else if( amount==_ODD_SITES ){
    int_offset_Dgpu += 1*4;
  }
  else{
    // TODO
  }

  // created offset of Dgpu
  Dgpu = s->op.Dgpu[int_offset_Dgpu] + sites_offset_Dgpu*9;
  for( i=0; i<nr_dumps_Dgpu; i++ ){
    // not all threads are needed when loading Dgpu
    if( (i*blockDim.x + idx%blockDim.x) < ((blockDim.x/2)*9) ){
      Dgpu_local[ i*blockDim.x + threadIdx.x ] = Dgpu[ i*blockDim.x + threadIdx.x ];
    }
  }
  if(blockDim.x>32) __syncthreads();
  _cuda_block_n_hopping_term_PRECISION_minus_2threads_opt( out, in, start, amount, s, idx,
                                                           tmp_loc, ext_dir, Dgpu_local, gamma_val, gamma_coo );
}


// gamma_val and gamma_coo are both loaded into shared memory for this kernel
__global__ void
cuda_block_n_hopping_term_PRECISION_minus_6threads_opt(		cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block, int ext_dir,
                                                                int amount ){
  int idx, i, *gamma_coo, start, int_offset_Dgpu, sites_offset_Dgpu, nr_dumps_Dgpu;
  cu_cmplx_PRECISION *gamma_val, *Dgpu_local, *tmp_loc, *Dgpu;

  idx = threadIdx.x + blockDim.x * blockIdx.x;

  DD_gateway_hopping( s, DD_blocks_to_compute, nr_threads_per_DD_block, num_latt_site_var, &idx, &start, &int_offset_Dgpu,
                      &sites_offset_Dgpu, &out, &in, &nr_dumps_Dgpu, &gamma_coo, &gamma_val, &tmp_loc, &Dgpu_local, amount,
                      block, ext_dir, 6 );

  int_offset_Dgpu = ext_dir;
  // due to computing the _minus_ kernel
  int_offset_Dgpu += 1*8;
  if( amount==_EVEN_SITES ){
    int_offset_Dgpu += 0*4;
  }
  else if( amount==_ODD_SITES ){
    int_offset_Dgpu += 1*4;
  }
  else{
    // TODO
  }

  // created offset of Dgpu
  Dgpu = s->op.Dgpu[int_offset_Dgpu] + sites_offset_Dgpu*9;
  for( i=0; i<nr_dumps_Dgpu; i++ ){
    // not all threads are needed when loading Dgpu
    if( (i*blockDim.x + idx%blockDim.x) < ((blockDim.x/6)*9) ){
      Dgpu_local[ i*blockDim.x + threadIdx.x ] = Dgpu[ i*blockDim.x + threadIdx.x ];
    }
  }
  __syncthreads();
  _cuda_block_n_hopping_term_PRECISION_minus_6threads_opt( out, in, start, amount, s, idx,
                                                           tmp_loc, ext_dir, Dgpu_local, gamma_val, gamma_coo );
}


__global__ void
cuda_block_hopping_term_PRECISION_plus_6threads_naive(		cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block, int ext_dir,
                                                                int amount ){
  int idx, *gamma_coo, DD_block_id, block_id, start;
  cu_cmplx_PRECISION* gamma_val;

  idx = threadIdx.x + blockDim.x * blockIdx.x;

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
  _cuda_block_hopping_term_PRECISION_plus_6threads_naive( out, in, start, amount, s, idx,
                                                          tmp_loc, ext_dir, gamma_coo, gamma_val);
}


__global__ void
cuda_block_hopping_term_PRECISION_plus_2threads_opt(            cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block, int ext_dir,
                                                                int amount ){
  int idx, i, *gamma_coo, start, int_offset_Dgpu, sites_offset_Dgpu, nr_dumps_Dgpu;
  cu_cmplx_PRECISION *gamma_val, *Dgpu_local, *tmp_loc, *Dgpu;

  idx = threadIdx.x + blockDim.x * blockIdx.x;

  DD_gateway_hopping( s, DD_blocks_to_compute, nr_threads_per_DD_block, num_latt_site_var, &idx, &start, &int_offset_Dgpu,
                      &sites_offset_Dgpu, &out, &in, &nr_dumps_Dgpu, &gamma_coo, &gamma_val, &tmp_loc, &Dgpu_local, amount,
                      block, ext_dir, 2 );

  int_offset_Dgpu = ext_dir;
  // due to computing the _minus_ kernel
  int_offset_Dgpu += 0*8;
  if( amount==_EVEN_SITES ){
    int_offset_Dgpu += 0*4;
  }
  else if( amount==_ODD_SITES ){
    int_offset_Dgpu += 1*4;
  }
  else{
    // TODO
  }

  // created offset of Dgpu
  Dgpu = s->op.Dgpu[int_offset_Dgpu] + sites_offset_Dgpu*9;

  for( i=0; i<nr_dumps_Dgpu; i++ ){
    // not all threads are needed when loading Dgpu
    if( (i*blockDim.x + idx%blockDim.x) < ((blockDim.x/2)*9) ){
      Dgpu_local[ i*blockDim.x + threadIdx.x ] = Dgpu[ i*blockDim.x + threadIdx.x ];
    }
  }
  if(blockDim.x>32) __syncthreads();
  _cuda_block_hopping_term_PRECISION_plus_2threads_opt( out, in, start, amount, s, idx,
                                                        tmp_loc, ext_dir, Dgpu_local, gamma_coo, gamma_val );
}


__global__ void
cuda_block_hopping_term_PRECISION_plus_6threads_opt(		cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block, int ext_dir,
                                                                int amount ){
  int idx, i, *gamma_coo, start, int_offset_Dgpu, sites_offset_Dgpu, nr_dumps_Dgpu;
  cu_cmplx_PRECISION *gamma_val, *Dgpu_local, *tmp_loc, *Dgpu;

  idx = threadIdx.x + blockDim.x * blockIdx.x;

  DD_gateway_hopping( s, DD_blocks_to_compute, nr_threads_per_DD_block, num_latt_site_var, &idx, &start, &int_offset_Dgpu,
                      &sites_offset_Dgpu, &out, &in, &nr_dumps_Dgpu, &gamma_coo, &gamma_val, &tmp_loc, &Dgpu_local, amount,
                      block, ext_dir, 6 );

  int_offset_Dgpu = ext_dir;
  // due to computing the _minus_ kernel
  int_offset_Dgpu += 0*8;
  if( amount==_EVEN_SITES ){
    int_offset_Dgpu += 0*4;
  }
  else if( amount==_ODD_SITES ){
    int_offset_Dgpu += 1*4;
  }
  else{
    // TODO
  }

  // created offset of Dgpu
  Dgpu = s->op.Dgpu[int_offset_Dgpu] + sites_offset_Dgpu*9;

  for( i=0; i<nr_dumps_Dgpu; i++ ){
    // not all threads are needed when loading Dgpu
    if( (i*blockDim.x + idx%blockDim.x) < ((blockDim.x/6)*9) ){
      Dgpu_local[ i*blockDim.x + threadIdx.x ] = Dgpu[ i*blockDim.x + threadIdx.x ];
    }
  }
  __syncthreads();
  _cuda_block_hopping_term_PRECISION_plus_6threads_opt( out, in, start, amount, s, idx,
                                                        tmp_loc, ext_dir, Dgpu_local, gamma_coo, gamma_val );
}


// gamma_val and gamma_coo are both loaded into shared memory for this kernel
__global__ void
cuda_block_hopping_term_PRECISION_minus_6threads_naive(		cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block, int ext_dir,
                                                                int amount ){
  int idx, DD_block_id, block_id, start, nr_block_even_sites;
  cu_cmplx_PRECISION *shared_data_loc, *tmp_loc;

  idx = threadIdx.x + blockDim.x * blockIdx.x;
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
  shared_data_loc = shared_data;

  //the following are 'bare' values, i.e. with respect to the 0th element within a CUDA block
  // EVEN
  tmp_loc = shared_data_loc;

  if(idx < 6*nr_block_even_sites){
    tmp_loc[threadIdx.x] = make_cu_cmplx_PRECISION(0.0,0.0);
    tmp_loc[threadIdx.x + blockDim.x] = make_cu_cmplx_PRECISION(0.0,0.0);
  } //even
  _cuda_block_hopping_term_PRECISION_minus_6threads_naive( out, in, start, amount, s, idx, tmp_loc, ext_dir );
}


__global__ void
cuda_block_hopping_term_PRECISION_minus_2threads_opt(           cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block, int ext_dir,
                                                                int amount ){
  int idx, i, *gamma_coo, start, int_offset_Dgpu, sites_offset_Dgpu, nr_dumps_Dgpu;
  cu_cmplx_PRECISION *gamma_val, *Dgpu_local, *tmp_loc, *Dgpu;

  idx = threadIdx.x + blockDim.x * blockIdx.x;

  DD_gateway_hopping( s, DD_blocks_to_compute, nr_threads_per_DD_block, num_latt_site_var, &idx, &start, &int_offset_Dgpu,
                      &sites_offset_Dgpu, &out, &in, &nr_dumps_Dgpu, &gamma_coo, &gamma_val, &tmp_loc, &Dgpu_local, amount,
                      block, ext_dir, 2 );

  int_offset_Dgpu = ext_dir;
  // due to computing the _minus_ kernel
  int_offset_Dgpu += 1*8;
  if( amount==_EVEN_SITES ){
    int_offset_Dgpu += 0*4;
  }
  else if( amount==_ODD_SITES ){
    int_offset_Dgpu += 1*4;
  }
  else{
    // TODO
  }

  // created offset of Dgpu
  Dgpu = s->op.Dgpu[int_offset_Dgpu] + sites_offset_Dgpu*9;
  for( i=0; i<nr_dumps_Dgpu; i++ ){
    // not all threads are needed when loading Dgpu
    if( (i*blockDim.x + idx%blockDim.x) < ((blockDim.x/2)*9) ){
      Dgpu_local[ i*blockDim.x + threadIdx.x ] = Dgpu[ i*blockDim.x + threadIdx.x ];
    }
  }
  if(blockDim.x>32) __syncthreads();
  _cuda_block_hopping_term_PRECISION_minus_2threads_opt( out, in, start, amount, s, idx,
                                                         tmp_loc, ext_dir, Dgpu_local, gamma_coo, gamma_val );
}


__global__ void
cuda_block_hopping_term_PRECISION_minus_6threads_opt(		cu_cmplx_PRECISION* out, cu_cmplx_PRECISION* in,
                                                                schwarz_PRECISION_struct_on_gpu *s, int thread_id,
                                                                double csw, int nr_threads_per_DD_block, int* DD_blocks_to_compute,
                                                                int num_latt_site_var, block_struct* block, int ext_dir,
                                                                int amount ){
  int idx, i, *gamma_coo, start, int_offset_Dgpu, sites_offset_Dgpu, nr_dumps_Dgpu;
  cu_cmplx_PRECISION *gamma_val, *Dgpu_local, *tmp_loc, *Dgpu;

  idx = threadIdx.x + blockDim.x * blockIdx.x;

  DD_gateway_hopping( s, DD_blocks_to_compute, nr_threads_per_DD_block, num_latt_site_var, &idx, &start, &int_offset_Dgpu,
                      &sites_offset_Dgpu, &out, &in, &nr_dumps_Dgpu, &gamma_coo, &gamma_val, &tmp_loc, &Dgpu_local, amount,
                      block, ext_dir, 6 );

  int_offset_Dgpu = ext_dir;
  // due to computing the _minus_ kernel
  int_offset_Dgpu += 1*8;
  if( amount==_EVEN_SITES ){
    int_offset_Dgpu += 0*4;
  }
  else if( amount==_ODD_SITES ){
    int_offset_Dgpu += 1*4;
  }
  else{
    // TODO
  }

  // created offset of Dgpu
  Dgpu = s->op.Dgpu[int_offset_Dgpu] + sites_offset_Dgpu*9;
  for( i=0; i<nr_dumps_Dgpu; i++ ){
    // not all threads are needed when loading Dgpu
    if( (i*blockDim.x + idx%blockDim.x) < ((blockDim.x/6)*9) ){
      Dgpu_local[ i*blockDim.x + threadIdx.x ] = Dgpu[ i*blockDim.x + threadIdx.x ];
    }
  }
  __syncthreads();
  _cuda_block_hopping_term_PRECISION_minus_6threads_opt( out, in, start, amount, s, idx,
                                                         tmp_loc, ext_dir, Dgpu_local, gamma_coo, gamma_val );
}


extern "C" void
cuda_apply_block_schur_complement_PRECISION(			cuda_vector_PRECISION out, cuda_vector_PRECISION in,
                                                                schwarz_PRECISION_struct *s, level_struct *l,
                                                                int nr_DD_blocks_to_compute,
                                                                int* DD_blocks_to_compute, cudaStream_t *streams,
                                                                int stream_id, int sites_to_solve ){
  if( nr_DD_blocks_to_compute==0 ){ return; }

  int dir, size_D_oeclov, threads_per_cublock_diagops, threads_per_cublock, nr_threads, nr_threads_per_DD_block;
  cu_cmplx_PRECISION **tmp, *tmp0, *tmp1;
  size_t tot_shared_mem;

  tmp = (s->s_on_gpu_cpubuff).oe_buf;
  tmp0 = tmp[0];
  tmp1 = tmp[1];

  // this is the size of the local matrix, i.e. per lattice site. 12^2=144, but ... (??)
  //size_D_oeclov = (g.csw!=0.0) ? 72 : 12;
  size_D_oeclov = (g.csw!=0.0) ? 42 : 12;

  // -*-*-*-*-* DIAGONAL EVEN OPERATION : block_diag_ee_PRECISION( out, in, start, s, l, threading ); (tunable! -- type1)

  threads_per_cublock_diagops = g.CUDA_threads_per_CUDA_block_type1[0];

  // nr sites per DD block
  nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites;
  nr_threads = nr_threads * g.CUDA_threads_per_lattice_site_type1[0]; // threads per site
  nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute

  nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

  // no shared memory needed in the simple operation of multiplying against a diagonal (when g.csw=0)
  if( g.csw != 0.0 ){
    tot_shared_mem = 2*((12/g.CUDA_threads_per_lattice_site_type1[0])*threads_per_cublock_diagops)*sizeof(cu_cmplx_PRECISION) +
                     1*size_D_oeclov*(threads_per_cublock_diagops/g.CUDA_threads_per_lattice_site_type1[0])*sizeof(cu_config_PRECISION);
  }
  else{
    tot_shared_mem = 0;
  }

  if( g.CUDA_threads_per_lattice_site_type1[0]==6 ){
    cuda_block_diag_ee_PRECISION_6threads_opt<<< nr_threads/threads_per_cublock_diagops,
                                                 threads_per_cublock_diagops, tot_shared_mem,
                                                 streams[stream_id]
                                             >>>
                                             ( out, in, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                               DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block );
  }
  else if( g.CUDA_threads_per_lattice_site_type1[0]==2 ){
    cuda_block_diag_ee_PRECISION_2threads_opt<<< nr_threads/threads_per_cublock_diagops,
                                                 threads_per_cublock_diagops, tot_shared_mem,
                                                 streams[stream_id]
                                             >>>
                                             ( out, in, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                               DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block );
  }
  else {
    // change the following exit and error output
    if( g.my_rank==0 ) printf("ERROR : CUDA threads per lattice site can be only 2 or 6 \n");
    exit(1);
  }

  // -*-*-*-*-* DEFINE : vector_PRECISION_define( tmp[0], 0, start + 12*s->num_block_even_sites, start + s->block_vector_size, l ); ( use type2? ) (tunable! -- type2)

  nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites;
  nr_threads = nr_threads * 12;
  nr_threads = nr_threads*nr_DD_blocks_to_compute;
  nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

  threads_per_cublock = 3 * g.CUDA_threads_per_CUDA_block_type2[0];

  cuda_block_oe_vector_PRECISION_define_12threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                        0, streams[stream_id]
                                                    >>>
                                                    ( tmp0, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                      DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block,
                                                      _ODD_SITES, make_cu_cmplx_PRECISION(0.0,0.0) );

  // -*-*-*-*-* HOPPING (tunable! -- type1)

  //block_hopping_term_PRECISION( tmp[0], in, start, _ODD_SITES, s, l, threading );

  threads_per_cublock = g.CUDA_threads_per_CUDA_block_type1[0];

  tot_shared_mem = 1*((12/g.CUDA_threads_per_lattice_site_type1[0])*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) +
                   1*9*(threads_per_cublock/g.CUDA_threads_per_lattice_site_type1[0])*sizeof(cu_cmplx_PRECISION);
                   //16*sizeof(cu_cmplx_PRECISION) + 16*sizeof(int);

  for( dir=0; dir<4; dir++ ){

    nr_threads = (sites_to_solve==_EVEN_SITES)?s->dir_length_even[dir]:s->dir_length_odd[dir];
    nr_threads = nr_threads * g.CUDA_threads_per_lattice_site_type1[0];
    nr_threads = nr_threads*nr_DD_blocks_to_compute;
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

    if( g.CUDA_threads_per_lattice_site_type1[0]==6 ){
      cuda_block_hopping_term_PRECISION_plus_6threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                             tot_shared_mem, streams[stream_id]
                                                         >>>
                                                         ( tmp0, in, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                           DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block,
                                                           dir, _ODD_SITES );
    }
    else if( g.CUDA_threads_per_lattice_site_type1[0]==2 ){
      cuda_block_hopping_term_PRECISION_plus_2threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                             tot_shared_mem, streams[stream_id]
                                                         >>>
                                                         ( tmp0, in, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                           DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block,
                                                           dir, _ODD_SITES );
    }
    else {
      // change the following exit and error output
      if( g.my_rank==0 ) printf("ERROR : CUDA threads per lattice site can be only 2 or 6 \n");
      exit(1);
    }

    if( g.CUDA_threads_per_lattice_site_type1[0]==6 ){
      cuda_block_hopping_term_PRECISION_minus_6threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                              tot_shared_mem, streams[stream_id]
                                                          >>>
                                                          ( tmp0, in, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                            DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block,
                                                            dir, _ODD_SITES );
    }
    else if( g.CUDA_threads_per_lattice_site_type1[0]==2 ){
      cuda_block_hopping_term_PRECISION_minus_2threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                              tot_shared_mem, streams[stream_id]
                                                          >>>
                                                          ( tmp0, in, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                            DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block,
                                                            dir, _ODD_SITES );
    }
    else {
      // change the following exit and error output
      if( g.my_rank==0 ) printf("ERROR : CUDA threads per lattice site can be only 2 or 6 \n");
      exit(1);
    }

  }

  // -*-*-*-*-* INVERSION OF ODD BLOCK DIAGONAL (tunable! -- type1)

  //block_diag_oo_inv_PRECISION( tmp[1], tmp[0], start, s, l, threading );
  // diag_oo inv

  threads_per_cublock_diagops = g.CUDA_threads_per_CUDA_block_type1[0];

  // nr sites per DD block
  nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites;
  nr_threads = nr_threads * g.CUDA_threads_per_lattice_site_type1[0]; // threads per site
  nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
  nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

  // no shared memory needed when applying with diag (when g.csw=0)
  if( g.csw != 0.0 ){
    tot_shared_mem = 2*((12/g.CUDA_threads_per_lattice_site_type1[0])*threads_per_cublock_diagops)*sizeof(cu_cmplx_PRECISION) +
                     1*size_D_oeclov*(threads_per_cublock_diagops/g.CUDA_threads_per_lattice_site_type1[0])*sizeof(cu_config_PRECISION);
  }
  else{
    tot_shared_mem = 0;
  }

  if( g.CUDA_threads_per_lattice_site_type1[0]==6 ){
    cuda_block_diag_oo_inv_PRECISION_6threads_opt<<< nr_threads/threads_per_cublock_diagops, threads_per_cublock_diagops,
                                                     tot_shared_mem, streams[stream_id]
                                                 >>>
                                                 ( tmp1, tmp0, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                   DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block );
  }
  else if( g.CUDA_threads_per_lattice_site_type1[0]==2 ){
    cuda_block_diag_oo_inv_PRECISION_2threads_opt<<< nr_threads/threads_per_cublock_diagops, threads_per_cublock_diagops,
                                                     tot_shared_mem, streams[stream_id]
                                                 >>>
                                                 ( tmp1, tmp0, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                   DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block );
  }
  else {
    // change the following exit and error output
    if( g.my_rank==0 ) printf("ERROR : CUDA threads per lattice site can be only 2 or 6 \n");
    exit(1);
  }

  // -*-*-*-*-* HOPPING NEGATIVE (tunable! -- type1)

  //block_n_hopping_term_PRECISION( out, tmp[1], start, _EVEN_SITES, s, l, threading );
  // hopping term, even sites

  threads_per_cublock = g.CUDA_threads_per_CUDA_block_type1[0];
  tot_shared_mem = 1*((12/g.CUDA_threads_per_lattice_site_type1[0])*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) +
                   1*9*(threads_per_cublock/g.CUDA_threads_per_lattice_site_type1[0])*sizeof(cu_cmplx_PRECISION);
                   //16*sizeof(cu_cmplx_PRECISION) + 16*sizeof(int);

  for( dir=0; dir<4; dir++ ){

    nr_threads = (sites_to_solve==_EVEN_SITES)?s->dir_length_even[dir]:s->dir_length_odd[dir];
    nr_threads = nr_threads * g.CUDA_threads_per_lattice_site_type1[0];
    nr_threads = nr_threads * nr_DD_blocks_to_compute;
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

    if( g.CUDA_threads_per_lattice_site_type1[0]==6 ){
      cuda_block_n_hopping_term_PRECISION_plus_6threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                               tot_shared_mem, streams[stream_id]
                                                           >>>
                                                           ( out, tmp1, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                             DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block,
                                                             dir, _EVEN_SITES );
    }
    else if( g.CUDA_threads_per_lattice_site_type1[0]==2 ){
      cuda_block_n_hopping_term_PRECISION_plus_2threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                               tot_shared_mem, streams[stream_id]
                                                           >>>
                                                           ( out, tmp1, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                             DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block,
                                                             dir, _EVEN_SITES );
    }
    else {
      // change the following exit and error output
      if( g.my_rank==0 ) printf("ERROR : CUDA threads per lattice site can be only 2 or 6 \n");
      exit(1);
    }

    if( g.CUDA_threads_per_lattice_site_type1[0]==6 ){
      cuda_block_n_hopping_term_PRECISION_minus_6threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                                tot_shared_mem, streams[stream_id]
                                                            >>>
                                                            ( out, tmp1, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                              DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block,
                                                              dir, _EVEN_SITES );
    }
    else if( g.CUDA_threads_per_lattice_site_type1[0]==2 ){
      cuda_block_n_hopping_term_PRECISION_minus_2threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                                tot_shared_mem, streams[stream_id]
                                                            >>>
                                                            ( out, tmp1, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                              DD_blocks_to_compute, l->num_lattice_site_var, (s->cu_s).block,
                                                              dir, _EVEN_SITES );
    }
    else {
      // change the following exit and error output
      if( g.my_rank==0 ) printf("ERROR : CUDA threads per lattice site can be only 2 or 6 \n");
      exit(1);
    }

  }
}


extern "C" void
cuda_blocks_vector_copy_noncontig_PRECISION_naive(		cuda_vector_PRECISION out, cuda_vector_PRECISION in,
                                                                int nr_DD_blocks_to_compute, schwarz_PRECISION_struct* s,
                                                                level_struct *l, int* DD_blocks_to_compute,
                                                                cudaStream_t *streams ){
  if( nr_DD_blocks_to_compute==0 ){ return; }

  int i, b_start;
  for( i=0; i<nr_DD_blocks_to_compute; i++ ){
    b_start = s->block[DD_blocks_to_compute[i]].start * l->num_lattice_site_var;
    cuda_vector_PRECISION_copy( (void*)out, (void*)in, b_start, s->block_vector_size,
                                l, _D2D, _CUDA_ASYNC, 0, streams );
  }
}


// Use of dynamic parallelism --> good idea, but not as good as odd-even launch
// (see cuda_block_oe_vector_PRECISION_copy_6threads_opt(...))
extern "C" void
cuda_blocks_vector_copy_noncontig_PRECISION_dyn(		cuda_vector_PRECISION out, cuda_vector_PRECISION in,
                                                                int nr_DD_blocks_to_compute, schwarz_PRECISION_struct* s,
                                                                level_struct *l, int* DD_blocks_to_compute,
                                                                cudaStream_t *streams, int stream_id ){
  if( nr_DD_blocks_to_compute==0 ){ return; }

  cuda_blocks_vector_copy_noncontig_PRECISION_dyn_dev<<< nr_DD_blocks_to_compute, 32, 0, streams[stream_id] >>>
                                                     ( out, in, DD_blocks_to_compute, (s->cu_s).block,
                                                       l->num_lattice_site_var, s->s_on_gpu );
}


extern "C" void
cuda_block_solve_oddeven_PRECISION(				cuda_vector_PRECISION phi, cuda_vector_PRECISION r,
                                                                cuda_vector_PRECISION latest_iter, int start,
                                                                int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s,
                                                                level_struct *l, struct Thread *threading, int stream_id,
                                                                cudaStream_t *streams, int solve_at_cpu, int color, 
                                                                int* DD_blocks_to_compute_gpu, int* DD_blocks_to_compute_cpu ){
  if( nr_DD_blocks_to_compute==0 ){ return; }

  if(solve_at_cpu){
    block_solve_oddeven_PRECISION( (vector_PRECISION)phi, (vector_PRECISION)r, (vector_PRECISION)latest_iter,
                                   start, s, l, threading );
  } else {

    // -*-*-*-*-* SOME INITIAL COMMENTS AND INITS

    int threads_per_cublock, nr_threads, size_D_oeclov, nr_threads_per_DD_block, dir, threads_per_cublock_diagops;
    size_t tot_shared_mem;

    // The 'nr_threads' var needed is computed like this: max between num_block_even_sites and num_block_odd_sites,
    // and then for each lattice site (of those even-odd), we need 12/2 really independent components, due to gamma5
    // symmetry. I.e. each thread is in charge of one site component !

    // this is the size of the local matrix, i.e. per lattice site
    size_D_oeclov = (g.csw!=0.0) ? 42 : 12;

    // ingredients composing shared memory:
    //                                     1. for memory associated to spinors, we first multiply threads_per_cublock by 2,
    // 					      this is to account for gamma5 symmetry (because we're thinking this way: 6 CUDA
    //					      threads correspond to a single lattice site), then, the factor of 4 comes from
    //					      the sub-spinors we need to use within the kernel: phi_?, r_?, tmp_2_?, tmp_3_?,
    //					      and finally the factor of 2 comes from the odd-even preconditioning taken here
    //                                     2. size_D_oeclov gives us the size of the local matrix per site, hence we need to
    //					      multiply by threads_per_cublock/6 (which gives us the nr of sites per CUDA block),
    //					      and then the factor of 2 comes from odd-even
    //
    // it's fundamental to think about the implementation here in the following way:
    //
    //                                     each CUDA block computes a certain nr of lattice sites, say X, but we're using
    //					   odd-even preconditioning, therefore that same CUDA block is in charge not only
    //					   of computing those X (say, even) sites, but also of computing the associated X
    //					   (then, odd) sites through odd-even preconditioning

    cu_cmplx_PRECISION **tmp, *tmp2, *tmp3;
    tmp = (s->s_on_gpu_cpubuff).oe_buf;
    tmp2 = tmp[2];
    tmp3 = tmp[3];

    // -*-*-*-*-* COPYING tmp3 <- r (tunable! -- type2)

    // nr sites per DD block
    nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites;
    nr_threads = nr_threads*12; // threads per site
    nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

    // it's important to accomodate for the factor of 3 in 4*3=12
    threads_per_cublock = 3 * g.CUDA_threads_per_CUDA_block_type2[0];

    // IMPORTANT: out of the three following options for copying odd-even, we choose the
    //            most efficient one, which in turn is not necessary obvious
    //cuda_blocks_vector_copy_noncontig_PRECISION_naive(tmp3, r, nr_DD_blocks_to_compute, s, l,
    // DD_blocks_to_compute_cpu, streams);
    //cuda_blocks_vector_copy_noncontig_PRECISION_dyn(tmp3, r, nr_DD_blocks_to_compute, s, l,
    // DD_blocks_to_compute_gpu, streams, 0);

    cuda_block_oe_vector_PRECISION_copy_12threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                        0, streams[stream_id]
                                                    >>>
                                                    ( tmp3, r, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                      DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block,
                                                      _FULL_SYSTEM );

    // -*-*-*-*-* INVERSION OF ODD BLOCK DIAGONAL (tunable! -- type1)

    // diag_oo inv
    // nr sites per DD block
    nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites;
    nr_threads = nr_threads * ( g.CUDA_threads_per_lattice_site_type1[0] ); // threads per site
    nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
    threads_per_cublock_diagops = g.CUDA_threads_per_CUDA_block_type1[0];
    if( g.csw != 0 ){
      tot_shared_mem = 2*(( 12/g.CUDA_threads_per_lattice_site_type1[0] )*threads_per_cublock_diagops)*sizeof(cu_cmplx_PRECISION) +
                       1*size_D_oeclov*(threads_per_cublock_diagops/g.CUDA_threads_per_lattice_site_type1[0])*sizeof(cu_config_PRECISION);
    }
    else{
      tot_shared_mem = 0;
    }
    if( g.CUDA_threads_per_lattice_site_type1[0]==2 ){
      cuda_block_diag_oo_inv_PRECISION_2threads_opt<<< nr_threads/threads_per_cublock_diagops, threads_per_cublock_diagops,
                                                       tot_shared_mem, streams[stream_id]
                                                   >>>
                                                   ( tmp2, tmp3, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                     DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block );
    } else if( g.CUDA_threads_per_lattice_site_type1[0]==6 ){
      cuda_block_diag_oo_inv_PRECISION_6threads_opt<<< nr_threads/threads_per_cublock_diagops, threads_per_cublock_diagops,
                                                       tot_shared_mem, streams[stream_id]
                                                   >>>
                                                   ( tmp2, tmp3, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                     DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block );
    } else {
      // change the following exit and error output
      if( g.my_rank==0 ) printf("ERROR : CUDA threads per lattice site can be only 2 or 6 \n");
      exit(1);
    }

    // -*-*-*-*-* HOPPING TERM NEGATIVE (tunable! -- type1)

    // hopping term, even sites

    threads_per_cublock = g.CUDA_threads_per_CUDA_block_type1[0];

    tot_shared_mem = 1*((12/g.CUDA_threads_per_lattice_site_type1[0])*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) +
                     1*9*(threads_per_cublock/g.CUDA_threads_per_lattice_site_type1[0])*sizeof(cu_cmplx_PRECISION);
                     //16*sizeof(cu_cmplx_PRECISION) + 16*sizeof(int);

    for( dir=0; dir<4; dir++ ){

      nr_threads = s->dir_length_even[dir];
      nr_threads = nr_threads * g.CUDA_threads_per_lattice_site_type1[0];
      nr_threads = nr_threads*nr_DD_blocks_to_compute;
      nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

      if( g.CUDA_threads_per_lattice_site_type1[0]==6 ){
        cuda_block_n_hopping_term_PRECISION_plus_6threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                                 tot_shared_mem, streams[stream_id]
                                                             >>>
                                                             ( tmp3, tmp2, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                               DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block,
                                                               dir, _EVEN_SITES );
      }
      else if( g.CUDA_threads_per_lattice_site_type1[0]==2 ){
        cuda_block_n_hopping_term_PRECISION_plus_2threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                                 tot_shared_mem, streams[stream_id]
                                                             >>>
                                                             ( tmp3, tmp2, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                               DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block,
                                                               dir, _EVEN_SITES );
      }
      else {
        // change the following exit and error output
        if( g.my_rank==0 ) printf("ERROR : CUDA threads per lattice site can be only 2 or 6 \n");
        exit(1);
      }

      if( g.CUDA_threads_per_lattice_site_type1[0]==6 ){
        cuda_block_n_hopping_term_PRECISION_minus_6threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                                  tot_shared_mem, streams[stream_id]
                                                              >>>
                                                              ( tmp3, tmp2, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                                DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block,
                                                                dir, _EVEN_SITES );
      }
      else if( g.CUDA_threads_per_lattice_site_type1[0]==2 ){
        cuda_block_n_hopping_term_PRECISION_minus_2threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                                  tot_shared_mem, streams[stream_id]
                                                              >>>
                                                              ( tmp3, tmp2, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                                DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block,
                                                                dir, _EVEN_SITES );
      }
      else {
        // change the following exit and error output
        if( g.my_rank==0 ) printf("ERROR : CUDA threads per lattice site can be only 2 or 6 \n");
        exit(1);
      }

    }

    // -*-*-*-*-* MINIMAL RESIDUAL ON EVEN SITES

    local_minres_PRECISION_CUDA( NULL, tmp3, tmp2, s, l, nr_DD_blocks_to_compute,
                                 DD_blocks_to_compute_gpu, streams, stream_id, _EVEN_SITES );

    // -*-*-*-*-* HOPPING TERM NEGATIVE (tunable! -- type1)

    // hopping term, odd sites

    threads_per_cublock = g.CUDA_threads_per_CUDA_block_type1[0];

    tot_shared_mem = 1*( (12/g.CUDA_threads_per_lattice_site_type1[0]) * threads_per_cublock )*sizeof(cu_cmplx_PRECISION) +
                     1*9*( threads_per_cublock/g.CUDA_threads_per_lattice_site_type1[0] )*sizeof(cu_cmplx_PRECISION);
                     //16*sizeof(cu_cmplx_PRECISION) + 16*sizeof(int);

    for( dir=0; dir<4; dir++ ){

      nr_threads = s->dir_length_odd[dir];
      nr_threads = nr_threads * g.CUDA_threads_per_lattice_site_type1[0];
      nr_threads = nr_threads*nr_DD_blocks_to_compute;
      nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

      if( g.CUDA_threads_per_lattice_site_type1[0]==6 ){
        cuda_block_n_hopping_term_PRECISION_plus_6threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                                 tot_shared_mem, streams[stream_id]
                                                             >>>
                                                             ( tmp3, tmp2, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                               DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block,
                                                               dir, _ODD_SITES );
      }
      else if( g.CUDA_threads_per_lattice_site_type1[0]==2 ) {
        cuda_block_n_hopping_term_PRECISION_plus_2threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                                 tot_shared_mem, streams[stream_id]
                                                             >>>
                                                             ( tmp3, tmp2, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                               DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block,
                                                               dir, _ODD_SITES );
      }
      else {
        // change the following exit and error output
        if( g.my_rank==0 ) printf("ERROR : CUDA threads per lattice site can be only 2 or 6 \n");
        exit(1);
      }

      if( g.CUDA_threads_per_lattice_site_type1[0]==6 ){
        cuda_block_n_hopping_term_PRECISION_minus_6threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                                  tot_shared_mem, streams[stream_id]
                                                              >>>
                                                              ( tmp3, tmp2, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                                DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block,
                                                                dir, _ODD_SITES );
      }
      else if( g.CUDA_threads_per_lattice_site_type1[0]==2 ) {
        cuda_block_n_hopping_term_PRECISION_minus_2threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                                  tot_shared_mem, streams[stream_id]
                                                              >>>
                                                              ( tmp3, tmp2, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                                DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block,
                                                                dir, _ODD_SITES );
      }
      else {
        // change the following exit and error output
        if( g.my_rank==0 ) printf("ERROR : CUDA threads per lattice site can be only 2 or 6 \n");
        exit(1);
      }

    }

    // -*-*-*-*-* INVERSION OF ODD BLOCK DIAGONAL (tunable! -- type1)

    // diag_oo inv
    // nr sites per DD block
    nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites;
    nr_threads = nr_threads * ( g.CUDA_threads_per_lattice_site_type1[0] ); // threads per site
    nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
    threads_per_cublock_diagops = g.CUDA_threads_per_CUDA_block_type1[0];
    if( g.csw != 0.0 ){
      tot_shared_mem = 2*(( 12/g.CUDA_threads_per_lattice_site_type1[0] )*threads_per_cublock_diagops)*sizeof(cu_cmplx_PRECISION) +
                       1*size_D_oeclov*(threads_per_cublock_diagops/g.CUDA_threads_per_lattice_site_type1[0])*sizeof(cu_config_PRECISION);
    }
    else{
      tot_shared_mem = 0;
    }

    if( g.CUDA_threads_per_lattice_site_type1[0]==2 ){
      cuda_block_diag_oo_inv_PRECISION_2threads_opt<<< nr_threads/threads_per_cublock_diagops, threads_per_cublock_diagops,
                                                       tot_shared_mem, streams[stream_id]
                                                   >>>
                                                   ( tmp2, tmp3, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                     DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block );
    } else if( g.CUDA_threads_per_lattice_site_type1[0]==6 ){
      cuda_block_diag_oo_inv_PRECISION_6threads_opt<<< nr_threads/threads_per_cublock_diagops, threads_per_cublock_diagops,
                                                       tot_shared_mem, streams[stream_id]
                                                   >>>
                                                   ( tmp2, tmp3, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                     DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block );
    } else {
      // change the following exit and error output
      if( g.my_rank==0 ) printf("ERROR : CUDA threads per lattice site can be only 2 or 6 \n");
      exit(1);
    }

    // -*-*-*-*-* FINAL UPDATE WITHIN BLOCK_SOLVE (tunable! -- type2)

    // update phi and latest_iter, and r
    // nr sites per DD block

    nr_threads = (s->num_block_odd_sites > s->num_block_even_sites) ? s->num_block_odd_sites : s->num_block_even_sites;
    nr_threads = nr_threads * 12; // threads per site
    nr_threads = nr_threads*nr_DD_blocks_to_compute; // nr of DD blocks to compute
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

    threads_per_cublock = 3 * g.CUDA_threads_per_CUDA_block_type2[0];

    cuda_block_solve_update_12threads_opt<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                             0, streams[stream_id]
                                         >>>
                                         ( phi, r, latest_iter, s->s_on_gpu, g.my_rank, g.csw, 0, nr_threads_per_DD_block,
                                           DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block );
  }
}

// TODO: move the following boundary-related (host) functions to cuda_schwarz_generic.c ?

extern "C" void
cuda_block_PRECISION_boundary_op( 				cuda_vector_PRECISION eta, cuda_vector_PRECISION phi,
	                                                        int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s,
	                                                        level_struct *l, struct Thread *threading, int stream_id,
	                                                        cudaStream_t *streams, int color, int* DD_blocks_to_compute_gpu,
	                                                        int* DD_blocks_to_compute_cpu ){
  if( nr_DD_blocks_to_compute==0 ){ return; }

  int dir, nr_threads, nr_threads_per_DD_block, threads_per_cublock, tot_shared_mem;

  threads_per_cublock = 96;
  tot_shared_mem = 1*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) +
                   16*sizeof(cu_cmplx_PRECISION) + 16*sizeof(int);

  for( dir=0; dir<4; dir++ ){
    // both directions (+ and -) are independent of each other... <<NOT AS IN THE HOPPING TERM>>
    //nr_threads = s->num_boundary_sites[dir*2] + s->num_boundary_sites[dir*2+1];
    nr_threads = s->num_boundary_sites[dir*2];
    nr_threads = nr_threads*(12/2);
    nr_threads = nr_threads*nr_DD_blocks_to_compute;
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;
    cuda_block_PRECISION_boundary_op_plus_naive<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                   tot_shared_mem, streams[stream_id]
                                               >>>
                                               ( eta, phi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                 DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block,
                                                 dir );

    cuda_block_PRECISION_boundary_op_minus_naive<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                    tot_shared_mem, streams[stream_id]
                                                >>>
                                                ( eta, phi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                  DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block,
                                                  dir );
  }
}

extern "C" void
cuda_n_block_PRECISION_boundary_op(				cuda_vector_PRECISION eta, cuda_vector_PRECISION phi,
	                                                        int nr_DD_blocks_to_compute, schwarz_PRECISION_struct *s,
	                                                        level_struct *l, struct Thread *threading, int stream_id,
	                                                        cudaStream_t *streams, int color, int* DD_blocks_to_compute_gpu,
	                                                        int* DD_blocks_to_compute_cpu ){
  if( nr_DD_blocks_to_compute==0 ){ return; }

  int dir, nr_threads, nr_threads_per_DD_block, threads_per_cublock, tot_shared_mem;

  threads_per_cublock = 96;
  tot_shared_mem = 1*(2*threads_per_cublock)*sizeof(cu_cmplx_PRECISION) +
                   16*sizeof(cu_cmplx_PRECISION) + 16*sizeof(int);

  for( dir=0; dir<4; dir++ ){

    // both directions (+ and -) are independent of each other... <<NOT AS IN THE HOPPING TERM>>
    //nr_threads = s->num_boundary_sites[dir*2] + s->num_boundary_sites[dir*2+1];
    nr_threads = s->num_boundary_sites[dir*2];
    nr_threads = nr_threads*(12/2);
    nr_threads = nr_threads*nr_DD_blocks_to_compute;
    nr_threads_per_DD_block = nr_threads/nr_DD_blocks_to_compute;

    cuda_n_block_PRECISION_boundary_op_plus_naive<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                     tot_shared_mem, streams[stream_id]
                                                 >>>
                                                 ( eta, phi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                   DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block,
                                                   dir );
    cuda_n_block_PRECISION_boundary_op_minus_naive<<< nr_threads/threads_per_cublock, threads_per_cublock,
                                                      tot_shared_mem, streams[stream_id]
                                                  >>>
                                                  ( eta, phi, s->s_on_gpu, g.my_rank, g.csw, nr_threads_per_DD_block,
                                                    DD_blocks_to_compute_gpu, l->num_lattice_site_var, (s->cu_s).block,
                                                    dir );
  }
}


extern "C" void
cuda_boundary_comms_copy_PRECISION(				cuda_vector_PRECISION out, cuda_vector_PRECISION in,
                                                                int *bound_table, int num_sites, level_struct *l ){
  int nr_threads = num_sites*l->num_lattice_site_var, threads_per_cublock=32;

  _cuda_boundary_comms_copy_PRECISION<<< nr_threads/threads_per_cublock, threads_per_cublock >>>
                                     ( out, in, bound_table, l->num_lattice_site_var );
}

#endif
