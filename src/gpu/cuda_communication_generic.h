/** \file cuda_communication_generic.h
 *  \brief Defines the cuda_comm_PRECISION_struct.
 */

#ifndef CUDA_COMMUNICATION_PRECISION_H
#define CUDA_COMMUNICATION_PRECISION_H

#include <mpi.h>

#include "cuda_vectors_PRECISION.h"
#ifdef GPU2GPU_COMMS_VIA_CPUS
typedef PRECISION _Complex *vector_PRECISION;
#endif

/**
 * \brief CUDA version of comm_PRECISION_struct.
 * 
 * As the buffers need to be allocated on the GPU, it makes sense that the management variables
 * are duplicated as well.
 */
typedef struct {
  int
      /** \brief The length of allocated communication buffers. */
      max_length[4],
      comm_start[8],
      /** \brief Indicates that there is a pending communication in that direction. */
      in_use[8],
      /** \brief Number of elements per communicated boundary site. */
      offset,
      /** \brief Indicates that communication neccessary i.e. there is more than 1 MPI process.*/
      comm,
      /** \brief Number of even boundary sites per direction. */
      num_even_boundary_sites[8],
      /** \brief Number of odd boundary sites per direction. */
      num_odd_boundary_sites[8],
      /** \brief Number of boundary sites per direction. */
      num_boundary_sites[8];
  /** \brief An array of indices of boundary sites per direction. */
  int *boundary_table_gpu[8];
  /** \brief The buffer to send or receive data in either direction.
   *
   *  It is noteworthy that only the sender OR receivere needs a dedicated buffer per direction.
   *  The other communication party can send/receive directly to the projection vector.
   */
  cuda_vector_PRECISION buffer_gpu[8];
#ifdef GPU2GPU_COMMS_VIA_CPUS
  vector_PRECISION buffer[8],buffer2[8];
#endif
  MPI_Request
      /** \brief Send request handles of MPI. */
      sreqs[8],
      /** \brief Receive request handles of MPI. */
      rreqs[8];
} cuda_comm_PRECISION_struct;

#endif  // CUDA_COMMUNICATION_PRECISION_H
