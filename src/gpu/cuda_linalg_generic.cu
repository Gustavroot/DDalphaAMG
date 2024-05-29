#include <mpi.h>

extern "C"{

  #define IMPORT_FROM_EXTERN_C
  #include "main.h"
  #undef IMPORT_FROM_EXTERN_C
}
//#include "global_defs.h"

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

void cuda_global_inner_product_PRECISION( cuda_vector_PRECISION* V, cuda_vector_PRECISION psi,
     complex_PRECISION *result, int n, int start, int end, level_struct *l, struct Thread *threading ) {

  // TODO

  //error0( "under construction \n" );

  //return make_cu_cmplx_PRECISION(1.0,0.0);
}

extern "C" void cuda_global_inner_product_PRECISION_vectorwrapper( vector_PRECISION* V, vector_PRECISION psi,
                complex_PRECISION *result, int n, int start, int end, level_struct *l, struct Thread *threading ) {

  //complex_PRECISION *dotprod_result=NULL;
  //PUBLIC_MALLOC( dotprod_result, complex_PRECISION, 1 );

  START_MASTER(threading)

  // CUDA stream, only one as only the master thread is in charge of this
  cudaStream_t stream = CU_STREAM_PER_THREAD;
  cudaStream_t* const streams = &stream;

  cuda_vector_PRECISION *V_gpu  = NULL;
  cuda_vector_PRECISION psi_gpu = NULL;

  // allocate input GPU data
  MALLOC( V_gpu, cuda_vector_PRECISION, n );
  CUDA_MALLOC( V_gpu[0], cu_cmplx_PRECISION, n*end );
  for ( int i=1;i<n;i++ ) { V_gpu[i] = V_gpu[0]+i*end; }
  CUDA_MALLOC( psi_gpu, cu_cmplx_PRECISION, end );

  // copy input data to GPUs
  for ( int i=0;i<n;i++ ) {
    cuda_vector_PRECISION_copy( V_gpu[i], V[i], start, end-start, l, _H2D, _CUDA_SYNC, 0, streams );
  }
  cuda_vector_PRECISION_copy( psi_gpu, psi, start, end-start, l, _H2D, _CUDA_SYNC, 0, streams );

  // offload the dot product to the GPUs
  //cu_cmplx_PRECISION cu_dotprod_result = cuda_global_inner_product_PRECISION( V_gpu, psi_gpu, result, n, start, end, l, threading );
  cuda_global_inner_product_PRECISION( V_gpu, psi_gpu, result, n, start, end, l, threading );

  //((PRECISION*)dotprod_result)[0] = cu_creal_PRECISION(cu_dotprod_result);
  //((PRECISION*)dotprod_result)[1] = cu_cimag_PRECISION(cu_dotprod_result);

  END_MASTER(threading)
  SYNC_CORES(threading)

  //complex_PRECISION result = dotprod_result[0];
  //SYNC_CORES(threading)
  //PUBLIC_FREE( dotprod_result, complex_PRECISION, 1 );

  //return result;
}

#endif
