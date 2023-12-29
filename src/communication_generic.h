/** \file communication_generic.h
 *  \brief Defines structs for MPI communication.
 */

#ifndef COMMUNICATION_PRECISION_H
#define COMMUNICATION_PRECISION_H
#ifndef IMPORT_FROM_EXTERN_C
#include <mpi.h>
#endif
#include "complex_types_PRECISION.h"
#ifdef CUDA_OPT
#include "gpu/cuda_vectors_PRECISION.h"
#endif

typedef struct
{
    int *boundary_table[8], max_length[4],
        comm_start[8], in_use[8], offset, comm,
        num_even_boundary_sites[8], num_odd_boundary_sites[8],
        num_boundary_sites[8];
#ifdef CUDA_OPT
    int *boundary_table_gpu[8];
    cuda_vector_PRECISION buffer_gpu[8];
#endif
    vector_PRECISION buffer[8];
#ifdef GPU2GPU_COMMS_VIA_CPUS
    vector_PRECISION buffer2[8];
#endif
    MPI_Request sreqs[8], rreqs[8];
} comm_PRECISION_struct;

typedef struct
{
    int ilde, dist_local_lattice[4], dist_inner_lattice_sites,
        *permutation, *gather_list, gather_list_length;
    vector_PRECISION buffer, transfer_buffer;
    MPI_Request *reqs;
    MPI_Group level_comm_group;
    MPI_Comm level_comm;
} gathering_PRECISION_struct;
#endif // COMMUNICATION_PRECISION_H
