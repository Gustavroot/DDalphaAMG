#ifndef MISCELLANEOUS_HEADER
#define MISCELLANEOUS_HEADER

#include <stdio.h>
#include "global_struct.h"

#ifdef CUDA_OPT

  // Specification of file and line nr to throw on error check
  #define cuda_safe_call( err ) __cuda_safe_call( err, __FILE__, __LINE__ )
  #define cuda_check_error( check_type ) __cuda_check_error( check_type, __FILE__, __LINE__ )

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

  static inline void __cuda_check_error( const int check_type, const char *file, const int line ){
#ifdef CUDA_ERROR_CHECK
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err ){
      fprintf( stderr, "cuda_check_error() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
      MPI_Abort(MPI_COMM_WORLD, 51);
    }

    if( check_type==_SOFT_CHECK ){}
    else if( check_type==_HARD_CHECK ){
      err = cudaDeviceSynchronize();
      if(cudaSuccess != err){
        fprintf( stderr, "cuda_check_error() with sync failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
        MPI_Abort(MPI_COMM_WORLD, 51);
      }
    }
    else{
      if( g.my_rank==0 ){ printf("Unknown type for check_type for cucda_check_error\n"); }
    }

#endif
    //return;
  }

#endif

  void coarsest_level_resets( level_struct* l, struct Thread* threading );
  void set_some_coarsest_level_improvs_params_for_setup( level_struct* l, struct Thread* threading );
  void set_some_coarsest_level_improvs_params_for_solve( level_struct* l, struct Thread* threading );

#endif
