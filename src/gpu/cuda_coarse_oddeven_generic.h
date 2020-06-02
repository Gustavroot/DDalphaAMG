#ifdef CUDA_OPT
#ifndef COARSE_ODDEVEN_PRECISION_HEADER_CUDA
  #define COARSE_ODDEVEN_PRECISION_HEADER_CUDA

  void coarse_apply_schur_complement_PRECISION_CUDA( cuda_vector_PRECISION out, cuda_vector_PRECISION in, operator_PRECISION_struct *op, level_struct *l, struct Thread *threading );

  void coarse_diag_ee_PRECISION_CUDA( cuda_vector_PRECISION y, cuda_vector_PRECISION x, operator_PRECISION_struct *op, level_struct *l, struct Thread *threading );

#endif
#endif
