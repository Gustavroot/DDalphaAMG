#ifndef ODDEVEN_PRECISION_CUDA
#define ODDEVEN_PRECISION_CUDA

extern void cuda_block_solve_oddeven_PRECISION( cuda_vector_PRECISION phi, cuda_vector_PRECISION r, cuda_vector_PRECISION latest_iter,
                                                schwarz_PRECISION_struct *s, int stream_id, cudaStream_t *stream );

#endif
