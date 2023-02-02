#ifndef CUDA_DIRAC_PRECISION_HEADER_CUDA
#define CUDA_DIRAC_PRECISION_HEADER_CUDA

#include "cuda_schwarz_PRECISION.h"
#include "cuda_vectors_PRECISION.h"

// device functions

__global__ void cuda_block_d_plus_clover_PRECISION_6threads_naive(
    cu_cmplx_PRECISION *out, cu_cmplx_PRECISION *in, schwarz_PRECISION_struct_on_gpu *s,
    int thread_id, double csw, int nr_threads_per_DD_block, int *DD_blocks_to_compute,
    int num_latt_site_var, block_struct *block, int ext_dir);

// host functions

extern void cuda_block_d_plus_clover_PRECISION(
    cuda_vector_PRECISION eta, cuda_vector_PRECISION phi, int nr_DD_blocks_to_compute,
    schwarz_PRECISION_struct *s, level_struct *l, struct Thread *threading, int stream_id,
    cudaStream_t *streams, int color, int *DD_blocks_to_compute_gpu, int *DD_blocks_to_compute_cpu);

extern void cuda_d_plus_clover_PRECISION(cuda_vector_PRECISION eta, cuda_vector_PRECISION phi,
                                         operator_PRECISION_struct *op, level_struct *l,
                                         struct Thread *threading);

extern void cuda_clover_PRECISION(cuda_vector_PRECISION eta, cuda_vector_PRECISION phi,
                                  cuda_config_PRECISION clover, int length, cudaStream_t* stream);

extern void cuda_d_plus_clover_PRECISION_vectorwrapper(vector_PRECISION eta, vector_PRECISION phi,
                                                       operator_PRECISION_struct *op,
                                                       level_struct *l, struct Thread *threading);

extern __constant__ cu_cmplx_PRECISION gamma_info_vals_PRECISION[16];
extern __constant__ int gamma_info_coo_PRECISION[16];

#endif
