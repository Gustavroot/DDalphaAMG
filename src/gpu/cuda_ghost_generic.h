#ifndef CUDA_GHOST_PRECISION_H
#define CUDA_GHOST_PRECISION_H

#include "cuda_communication_PRECISION.h"
#include "level_struct.h"

void cuda_ghost_alloc_PRECISION(int buffer_size, cuda_comm_PRECISION_struct *c, level_struct *l);
void cuda_ghost_free_PRECISION(cuda_comm_PRECISION_struct *c, level_struct *l);
void cuda_ghost_sendrecv_init_PRECISION(int type, cuda_comm_PRECISION_struct *c, level_struct *l);
void cuda_ghost_sendrecv_PRECISION(cuda_vector_PRECISION phi, int mu, int dir,
                                   cuda_comm_PRECISION_struct *c, int amount, level_struct *l);
void cuda_ghost_wait_PRECISION(cuda_vector_PRECISION phi, int mu, int dir,
                               cuda_comm_PRECISION_struct *c, int amount, level_struct *l);

#endif  // CUDA_GHOST_PRECISION_H