#ifndef CUDA_ALGORITHM_STRUCTS_PRECISION_H
#define CUDA_ALGORITHM_STRUCTS_PRECISION_H
#include "block_struct.h"
#include "cuda_vectors_PRECISION.h"

typedef struct
{
    cu_config_PRECISION *oe_clover_vectorized;
    int *neighbor_table;
    cu_config_PRECISION *D;
    cu_cmplx_PRECISION *Dgpu[16];
    int nr_elems_Dgpu[16];
    cu_cmplx_PRECISION *clover_gpustorg;
    cu_cmplx_PRECISION *oe_clover_gpustorg;
} operator_PRECISION_struct_on_gpu;

// CUDA structs:
//	cuda_schwarz_PRECISION_struct:
//		the elements of this struct will be accessed from the CPU, but their content
//		are pointers pointing to GPU-data
//	schwarz_PRECISION_struct_on_gpu:
//		the elements of this struct will be accessed from within the GPU !
typedef struct
{
    cuda_vector_PRECISION buf1, buf2, buf3, buf4, buf5, buf6;
    int **DD_blocks_in_comms, **DD_blocks_notin_comms, **DD_blocks;
    block_struct *block;
    cuda_vector_PRECISION local_minres_buffer[3];
} cuda_schwarz_PRECISION_struct;
typedef struct
{
    cu_cmplx_PRECISION *oe_buf[4];
    cu_config_PRECISION *oe_clover_vectorized;
    operator_PRECISION_struct_on_gpu op;
    int num_block_even_sites, num_block_odd_sites;
    int block_vector_size;
    int *oe_index[4];
    int *index[4];
    int dir_length_even[4], dir_length_odd[4];
    int dir_length[4];
    int block_boundary_length[9];
    cu_cmplx_PRECISION gamma_info_vals[16];
    int gamma_info_coo[16];
    cu_cmplx_PRECISION *alphas;
} schwarz_PRECISION_struct_on_gpu;

#endif // CUDA_ALGORITHM_STRUCTS_PRECISION_H
