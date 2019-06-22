#ifndef MISCELLANEOUS_HEADER
  #define MISCELLANEOUS_HEADER

  // Specification of file and line nr to throw on error check
  #define cuda_safe_call( err ) __cuda_safe_call( err, __FILE__, __LINE__ )
  #define cuda_check_error()    __cuda_check_error( __FILE__, __LINE__ )

  void field_saver( void* phi, int length, char* datatype, char* filename );

  void set_cuda_device( int device_id );

  static inline void __cuda_safe_call( cudaError_t err, const char *file, const int line ){
#ifdef CUDA_ERROR_CHECK
    if(cudaSuccess != err){
      fprintf( stderr, "cuda_safe_call() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
      //MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD, 51);
    }
#endif
    //return;
}

  static inline void __cuda_check_error( const char *file, const int line ){
#ifdef CUDA_ERROR_CHECK
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err ){
      fprintf( stderr, "cuda_check_error() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
      MPI_Abort(MPI_COMM_WORLD, 51);
    }

    // More careful checking. However, this will affect performance. Comment away if needed.
    err = cudaDeviceSynchronize();
    if(cudaSuccess != err){
      fprintf( stderr, "cuda_check_error() with sync failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
      MPI_Abort(MPI_COMM_WORLD, 51);
    }
#endif
    //return;
  }

#endif
