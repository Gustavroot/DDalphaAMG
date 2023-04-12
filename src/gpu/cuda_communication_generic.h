#ifndef CUDA_COMMUNICATION_PRECISION_H
#define CUDA_COMMUNICATION_PRECISION_H

#include <mpi.h>
#include "cuda_vectors_PRECISION.h"

typedef struct
{
    int max_length[4],
        comm_start[8], in_use[8], offset, comm,
        num_even_boundary_sites[8], num_odd_boundary_sites[8],
        num_boundary_sites[8];
    int *boundary_table_gpu[8];
    cuda_vector_PRECISION buffer_gpu[8];
    MPI_Request sreqs[8], rreqs[8];
} cuda_comm_PRECISION_struct;

#endif // CUDA_COMMUNICATION_PRECISION_H