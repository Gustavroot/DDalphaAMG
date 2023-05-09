#ifdef CUDA_OPT
#ifndef SCHWARZ_PRECISION_CUDA
#define SCHWARZ_PRECISION_CUDA
#include "level_struct.h"
#include "cuda_algorithm_structs_PRECISION.h"


void smoother_PRECISION_def_CUDA(level_struct *l);
void smoother_PRECISION_free_CUDA(level_struct *l);

void schwarz_PRECISION_init_CUDA(schwarz_PRECISION_struct *s, level_struct *l);
void schwarz_PRECISION_alloc_CUDA(schwarz_PRECISION_struct *s, level_struct *l);
void schwarz_PRECISION_free_CUDA(schwarz_PRECISION_struct *s, level_struct *l);

void schwarz_PRECISION_setup_CUDA(schwarz_PRECISION_struct *s, operator_double_struct *op_in, level_struct *l);

void schwarz_PRECISION_def_CUDA(schwarz_PRECISION_struct *s, operator_double_struct *op, level_struct *l);

void schwarz_PRECISION_CUDA(vector_PRECISION phi, vector_PRECISION D_phi, vector_PRECISION eta, const int cycles, int res,
                            schwarz_PRECISION_struct *s, level_struct *l, struct Thread *threading);

#endif
#endif
