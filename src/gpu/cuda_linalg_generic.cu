#include <mpi.h>

extern "C"{

  #define IMPORT_FROM_EXTERN_C
  #include "main.h"
  #undef IMPORT_FROM_EXTERN_C
}

#ifdef CUDA_OPT

//#include "global_defs.h"

// this macro is used to determine the number of threads per
// CUDA block for dot products offloaded to GPUs
#define imin(a,b) (a<b?a:b)
// IMPORTANT : if changed, change also in src/linsolve_generic.c
static const int threadsPerBlockDP = 256;


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

__global__ void _cuda_process_partial_inner_product_PRECISION( cuda_vector_PRECISION a,
                cuda_vector_PRECISION b, cu_cmplx_PRECISION* c, int N ) {

  // partially taken from https://github.com/jiekebo/CUDA-By-Example/blob/master/5-dotproduct.cu

  __shared__ cu_cmplx_PRECISION cache[threadsPerBlockDP];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;

  cu_cmplx_PRECISION temp = make_cu_cmplx_PRECISION( 0.0,0.0 );
  while (tid < N){
    //temp += a[tid] * b[tid];
    temp = cu_cadd_PRECISION( temp, cu_cmul_PRECISION( cu_conj_PRECISION(a[tid]), b[tid] ) );
    tid += blockDim.x * gridDim.x;
  }

  // set the cache values
  cache[cacheIndex] = temp;

  // synchronize threads in this block
  __syncthreads();

  // for reductions, threadsPerBlock must be a power of 2
  // because of the following code
  int i = blockDim.x/2;
  while (i != 0){
    if (cacheIndex < i) {
      //cache[cacheIndex] += cache[cacheIndex + i];
      cache[cacheIndex] = cu_cadd_PRECISION( cache[cacheIndex],cache[cacheIndex+i] );
    }
    __syncthreads();
    i /= 2;
  }

  if (cacheIndex == 0) {
    c[blockIdx.x] = cache[0];
  }
}

void cuda_global_inner_product_PRECISION( cuda_vector_PRECISION* V, cuda_vector_PRECISION psi,
     complex_PRECISION *result, int n, int start, int end, gmres_PRECISION_struct *p,
     level_struct *l, struct Thread *threading ) {

  const int N = end-start;
  const int blocksPerGrid = imin(32, (N+threadsPerBlockDP-1) / threadsPerBlockDP);

  cudaStream_t stream = CU_STREAM_PER_THREAD;
  cudaStream_t* const streams = &stream;

  vector_PRECISION *partial_sums = p->gpu_dotprods_partial_sums;
  cuda_vector_PRECISION *dev_partial_sums = p->gpu_dotprods_dev_partial_sums;
  vector_PRECISION global_sums = p->gpu_dotprods_global_sums;

  //vector_PRECISION *partial_sums = NULL;
  //cuda_vector_PRECISION *dev_partial_sums = NULL;
  //vector_PRECISION global_sums = NULL;

  //// TODO : move these allocs to some 'setup'/'init' function
  //MALLOC( partial_sums, complex_PRECISION*, n );
  //partial_sums[0] = NULL;
  //MALLOC( partial_sums[0], complex_PRECISION, n*blocksPerGrid );
  //for ( int i=1;i<n;i++ ) { partial_sums[i] = partial_sums[0] + i*blocksPerGrid; }
  //MALLOC( dev_partial_sums, cu_cmplx_PRECISION*, n );
  //dev_partial_sums[0] = NULL;
  //CUDA_MALLOC( dev_partial_sums[0], cu_cmplx_PRECISION, n*blocksPerGrid );
  //for ( int i=1;i<n;i++ ) { dev_partial_sums[i] = dev_partial_sums[0] + i*blocksPerGrid; }
  //MALLOC( global_sums, complex_PRECISION, n );

  for ( int i=0;i<n;i++ ) {
    _cuda_process_partial_inner_product_PRECISION<<<blocksPerGrid, threadsPerBlockDP>>>
                                                 ( V[i]+start, psi+start, dev_partial_sums[i], N );
  }
  cuda_safe_call( cudaDeviceSynchronize() );

  for ( int i=0;i<n;i++ ) {
    cuda_vector_PRECISION_copy( partial_sums[i], dev_partial_sums[i], 0, blocksPerGrid,
                                l, _D2H, _CUDA_SYNC, 0, streams );

    result[i] = 0.0;
    for ( int j=0;j<blocksPerGrid;j++ ) {
      result[i] += partial_sums[i][j];
    }
  }

  // FIXME ? ( is g.num_processes the best way to go here? )
  if ( g.num_processes > 1 ) {
    MPI_Allreduce( result, global_sums, n, MPI_COMPLEX_PRECISION, MPI_SUM, (l->depth==0)?g.comm_cart:l->gs_PRECISION.level_comm );
    for ( int i=0;i<n;i++ ) { result[i] = global_sums[i]; }
  }

  //// TODO : move these allocs to some 'setup'/'init' function
  //FREE( partial_sums[0], complex_PRECISION, n*blocksPerGrid );
  //FREE( partial_sums, complex_PRECISION*, n );
  //CUDA_FREE( dev_partial_sums[0], cu_cmplx_PRECISION, blocksPerGrid );
  //FREE( dev_partial_sums, cu_cmplx_PRECISION*, n*blocksPerGrid );
  //FREE( global_sums, complex_PRECISION, n );

  /*

  // this 'legacy' code does one dot product at a time

  complex_PRECISION *partial_sum = NULL;
  cu_cmplx_PRECISION *dev_partial_sum = NULL;
  complex_PRECISION global_sum;

  // TODO : move these allocs to some 'setup'/'init' function
  MALLOC( partial_sum, complex_PRECISION, blocksPerGrid );
  CUDA_MALLOC( dev_partial_sum, cu_cmplx_PRECISION, blocksPerGrid );

  for ( int i=0;i<n;i++ ) {

    _cuda_process_partial_inner_product_PRECISION<<<blocksPerGrid, threadsPerBlock>>>
                                                 ( V[i]+start, psi+start, dev_partial_sum, N );
    cuda_safe_call( cudaDeviceSynchronize() );

    cuda_vector_PRECISION_copy( partial_sum, dev_partial_sum, 0, blocksPerGrid,
                                l, _D2H, _CUDA_SYNC, 0, streams );

    result[i] = 0.0;
    for ( int j=0;j<blocksPerGrid;j++ ) {
      result[i] += partial_sum[j];
    }

    // FIXME ? ( is g.num_processes the best way to go here? )
    if ( g.num_processes > 1 ) {
      MPI_Allreduce( &(result[i]), &global_sum, 1, MPI_COMPLEX_PRECISION, MPI_SUM, (l->depth==0)?g.comm_cart:l->gs_PRECISION.level_comm );
      result[i] = global_sum;
    }
  }

  // TODO : move these allocs to some 'setup'/'init' function
  FREE( partial_sum, complex_PRECISION, blocksPerGrid );
  CUDA_FREE( dev_partial_sum, cu_cmplx_PRECISION, blocksPerGrid );

  */
}

extern "C" void cuda_global_inner_product_PRECISION_vectorwrapper( vector_PRECISION* V, vector_PRECISION psi,
                complex_PRECISION *result, int n, int start, int end, gmres_PRECISION_struct *p,
                level_struct *l, struct Thread *threading ) {

  START_MASTER(threading)

  // CUDA stream, only one as only the master thread is in charge of this
  cudaStream_t stream = CU_STREAM_PER_THREAD;
  cudaStream_t* const streams = &stream;

  cuda_vector_PRECISION *V_gpu  = NULL;
  cuda_vector_PRECISION psi_gpu = NULL;

  // allocate input GPU vectors
  MALLOC( V_gpu, cuda_vector_PRECISION, n );
  V_gpu[0] = NULL;
  CUDA_MALLOC( V_gpu[0], cu_cmplx_PRECISION, n*end );
  for ( int i=1;i<n;i++ ) { V_gpu[i] = V_gpu[0]+i*end; }
  CUDA_MALLOC( psi_gpu, cu_cmplx_PRECISION, end );

  // copy input data to GPUs
  for ( int i=0;i<n;i++ ) {
    cuda_vector_PRECISION_copy( V_gpu[i], V[i], start, end-start, l, _H2D, _CUDA_SYNC, 0, streams );
  }
  cuda_vector_PRECISION_copy( psi_gpu, psi, start, end-start, l, _H2D, _CUDA_SYNC, 0, streams );

  // offload the dot product to the GPUs
  cuda_global_inner_product_PRECISION( V_gpu, psi_gpu, result, n, start, end, p, l, threading );

  // release the allocated buffer data
  CUDA_FREE( psi_gpu, cu_cmplx_PRECISION, end );
  CUDA_FREE( V_gpu[0], cu_cmplx_PRECISION, n*end );
  FREE( V_gpu, cuda_vector_PRECISION, n );

  END_MASTER(threading)
  SYNC_CORES(threading)
}

#endif
