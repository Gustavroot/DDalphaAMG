#ifdef CUDA_OPT


#include <mpi.h>

extern "C"{

  #define IMPORT_FROM_EXTERN_C
  #include "main.h"
  #undef IMPORT_FROM_EXTERN_C

}

extern "C" void cuda_vector_PRECISION_copy( void* out, void* in, int start, int size_of_copy, level_struct *l, int memcpy_kind, int cuda_async_type, int stream_id, cudaStream_t *streams ){

  switch(memcpy_kind){

    case _H2D:

      if( cuda_async_type==_CUDA_ASYNC ){
        cuda_safe_call( cudaMemcpyAsync((cuda_vector_PRECISION)(out) + start, (vector_PRECISION)(in) + start, size_of_copy*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice, streams[stream_id]) );
      }
      else if( cuda_async_type==_CUDA_SYNC ){
        cuda_safe_call( cudaMemcpy((cuda_vector_PRECISION)(out) + start, (vector_PRECISION)(in) + start, size_of_copy*sizeof(cu_cmplx_PRECISION), cudaMemcpyHostToDevice) );
      }
      else{
        if( g.my_rank==0 ){ printf("Wrong option for cuda_async_type in call to cuda_vector_PRECISION_copy(...).\n"); }
        MPI_Abort(MPI_COMM_WORLD, 51);
      }
      break;

    case _D2H:

      if( cuda_async_type==_CUDA_ASYNC ){
        cuda_safe_call( cudaMemcpyAsync((vector_PRECISION)(out) + start, (cuda_vector_PRECISION)(in) + start, size_of_copy*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost, streams[stream_id]) );
      }
      else if( cuda_async_type==_CUDA_SYNC ){
        cuda_safe_call( cudaMemcpy((vector_PRECISION)(out) + start, (cuda_vector_PRECISION)(in) + start, size_of_copy*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost) );
      }
      else{
        if( g.my_rank==0 ){ printf("Wrong option for cuda_async_type in call to cuda_vector_PRECISION_copy(...).\n"); }
        MPI_Abort(MPI_COMM_WORLD, 51);
      }
      break;

    case _D2D:
      // TODO : add call to CUDA kernel here
      printf("D2D cuda_vector_PRECISION_copy(...) not availble yet.\n");
      MPI_Abort(MPI_COMM_WORLD, 51);
      break;

    // In case the direction of copy is not one of {H2D, D2H, D2D}
    default:
      if(g.my_rank==0) { printf("Incorrect copy direction of CUDA vector.\n"); }
      MPI_Abort(MPI_COMM_WORLD, 51);
  }
}

#endif
