#include <mpi.h>

extern "C"{

  #define IMPORT_FROM_EXTERN_C
  #include "main.h"
  #undef IMPORT_FROM_EXTERN_C

}

#ifdef CUDA_OPT

extern "C" void
cuda_vector_PRECISION_copy(void *out, void const * in, int start, int size_of_copy, level_struct *l,
                           int memcpy_kind, int cuda_async_type, int stream_id,
                           cudaStream_t *streams){

  switch(memcpy_kind){

    case _H2D:

      if( cuda_async_type==_CUDA_ASYNC ){
        cuda_safe_call( cudaMemcpyAsync( (cuda_vector_PRECISION)(out) + start,
                                         (vector_PRECISION)(in) + start, size_of_copy*sizeof(cu_cmplx_PRECISION),
                                         cudaMemcpyHostToDevice, streams[stream_id] ) );
      }
      else if( cuda_async_type==_CUDA_SYNC ){
        cuda_safe_call( cudaMemcpy( (cuda_vector_PRECISION)(out) + start,
                                    (vector_PRECISION)(in) + start, size_of_copy*sizeof(cu_cmplx_PRECISION),
                                    cudaMemcpyHostToDevice ) );
      }
      else{
        if( g.my_rank==0 ){ printf("Wrong option for cuda_async_type in call to cuda_vector_PRECISION_copy(...).\n"); }
        MPI_Abort(MPI_COMM_WORLD, 51);
      }
      break;

    case _D2H:

      if( cuda_async_type==_CUDA_ASYNC ){
        cuda_safe_call( cudaMemcpyAsync( (vector_PRECISION)(out) + start, (cuda_vector_PRECISION)(in) +
                                         start, size_of_copy*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost,
                                         streams[stream_id] ) );
      }
      else if( cuda_async_type==_CUDA_SYNC ){
        cuda_safe_call( cudaMemcpy( (vector_PRECISION)(out) + start, (cuda_vector_PRECISION)(in) + start,
                                    size_of_copy*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost ) );
      }
      else{
        if( g.my_rank==0 ){ printf("Wrong option for cuda_async_type in call to cuda_vector_PRECISION_copy(...).\n"); }
        MPI_Abort(MPI_COMM_WORLD, 51);
      }
      break;

    case _D2D:

      if( cuda_async_type==_CUDA_ASYNC ){
        //cuda_safe_call( cudaMemcpyAsync((vector_PRECISION)(out) + start, (cuda_vector_PRECISION)(in) + start,
        // size_of_copy*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost, streams[stream_id]) );
        cuda_safe_call( cudaMemcpyAsync( (cuda_vector_PRECISION)(out) + start, (cuda_vector_PRECISION)(in) + start,
                                         size_of_copy*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToDevice,
                                         streams[stream_id] ) );
      }
      else if( cuda_async_type==_CUDA_SYNC ){
        //cuda_safe_call( cudaMemcpy((vector_PRECISION)(out) + start, (cuda_vector_PRECISION)(in) + start,
        // size_of_copy*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToHost) );
        cuda_safe_call( cudaMemcpy( (cuda_vector_PRECISION)(out) + start, (cuda_vector_PRECISION)(in) + start,
                                    size_of_copy*sizeof(cu_cmplx_PRECISION), cudaMemcpyDeviceToDevice ) );
      }
      else{
        if( g.my_rank==0 ){ printf("Wrong option for cuda_async_type in call to cuda_vector_PRECISION_copy(...).\n"); }
        MPI_Abort(MPI_COMM_WORLD, 51);
      }
      break;

    // In case the direction of copy is not one of {H2D, D2H, D2D}
    default:
      if(g.my_rank==0) { printf("Incorrect copy direction of CUDA vector.\n"); }
      MPI_Abort(MPI_COMM_WORLD, 51);
  }
}

__global__ void _cuda_vector_PRECISION_minus( cuda_vector_PRECISION z, cuda_vector_PRECISION x, cuda_vector_PRECISION y ){

  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  z[idx] = cu_csub_PRECISION( x[idx],y[idx] );
}

extern "C" void cuda_vector_PRECISION_minus( cuda_vector_PRECISION z, cuda_vector_PRECISION x, cuda_vector_PRECISION y, int start,
                                             int length, level_struct *l, int sync_type, int stream_id, cudaStream_t *streams ){

  int nr_threads = length;
  int threads_per_cublock = 32;

  _cuda_vector_PRECISION_minus<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>>
                              ( z+start, x+start, y+start );

  if( sync_type == _CUDA_SYNC ){
    cuda_safe_call( cudaDeviceSynchronize() );
  }

}

__global__ void _cuda_vector_PRECISION_saxpy( cuda_vector_PRECISION z, cuda_vector_PRECISION x, cuda_vector_PRECISION y, cu_cmplx_PRECISION alpha ){

  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  z[idx] = cu_cadd_PRECISION( x[idx] , cu_cmul_PRECISION( alpha,y[idx] ) );
}

extern "C" void cuda_vector_PRECISION_saxpy( cuda_vector_PRECISION z, cuda_vector_PRECISION x, cuda_vector_PRECISION y, cu_cmplx_PRECISION alpha, int start,
                                             int length, level_struct *l, int sync_type, int stream_id, cudaStream_t *streams ){

  int nr_threads = length;
  int threads_per_cublock = 32;

  PROF_PRECISION_START( _LA8 );

  _cuda_vector_PRECISION_saxpy<<< nr_threads/threads_per_cublock, threads_per_cublock, 0, streams[stream_id] >>>
                              ( z+start, x+start, y+start, alpha );

  if( sync_type == _CUDA_SYNC ){
    cuda_safe_call( cudaDeviceSynchronize() );
  }

  PROF_PRECISION_STOP( _LA8, (double)(length)/(double)l->inner_vector_size );
}

#endif
