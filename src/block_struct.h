/**
 *  \brief Somehow deals with blocks.
 *
 *  This was moved here from main during refactoring.
 */

#ifndef BLOCK_STRUCT_H
#define BLOCK_STRUCT_H

typedef struct block_struct
{
    int start, color, no_comm, *bt;
#ifdef CUDA_OPT
    int *bt_on_gpu;
#endif
} block_struct;

#endif // BLOCK_STRUCT_H
