#ifndef CUDA_OUTER_FGMRES_HEADER
#define CUDA_OUTER_FGMRES_HEADER

#ifdef CUDA_OPT
#ifdef __cplusplus
extern "C" {
#endif
  // 1 if the outer FGMRES of the solve can run on the GPU for this configuration
  int cuda_outer_fgmres_applicable( gmres_double_struct *p, level_struct *l );
  // outer FGMRES on the GPU, same interface and output as fgmres_double (called by all threads)
  int cuda_outer_fgmres_double( gmres_double_struct *p, level_struct *l, struct Thread *threading );
#ifdef __cplusplus
}
#endif
#endif

#endif
