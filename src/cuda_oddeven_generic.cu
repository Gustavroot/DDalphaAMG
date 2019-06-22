#ifdef CUDA_OPT
//#include "main.h"

#include <mpi.h>

extern "C"{

  #define IMPORT_FROM_EXTERN_C
  #include "main.h"
  #undef IMPORT_FROM_EXTERN_C

}

extern "C" void cuda_block_solve_oddeven_PRECISION( cuda_vector_PRECISION phi, cuda_vector_PRECISION r, cuda_vector_PRECISION latest_iter,
                                                    schwarz_PRECISION_struct *s, int stream_id, cudaStream_t *stream ){

  // TODO

}
#endif
