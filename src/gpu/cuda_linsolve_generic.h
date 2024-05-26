#ifndef LINSOLVE_PRECISION_HEADER_CUDA
#define LINSOLVE_PRECISION_HEADER_CUDA

#ifdef __cplusplus
extern "C" {
#endif

void cuda_fgmres_PRECISION_struct_init(gmres_PRECISION_struct *p);

void cuda_fgmres_PRECISION_struct_alloc(int m, int n, int vl, PRECISION tol, const int type,
                                   const int prec_kind, void (*precond)(), void (*eval_op)(),
                                   gmres_PRECISION_struct *p, level_struct *l);

void cuda_fgmres_PRECISION_struct_free(gmres_PRECISION_struct *p, level_struct *l);

void local_minres_PRECISION_CUDA(cuda_vector_PRECISION phi, cuda_vector_PRECISION eta,
                                 cuda_vector_PRECISION latest_iter, schwarz_PRECISION_struct *s,
                                 level_struct *l, int nr_DD_blocks_to_compute,
                                 int *DD_blocks_to_compute, cudaStream_t *streams, int stream_id,
                                 int sites_to_solve);

#ifdef __cplusplus
}
#endif

extern int cuda_richardson_PRECISION_vectorwrapper( gmres_PRECISION_struct *p, level_struct *l, 
                                                    struct Thread *threading );

#ifdef RICHARDSON_SMOOTHER
int cuda_richardson_PRECISION( gmres_PRECISION_struct *p, level_struct *l,
                               struct Thread *threading );
#endif

#endif
