#ifndef ALLOC_CONTROL_H
#define ALLOC_CONTROL_H

#include "console_out.h"
#ifdef CUDA_OPT
#include "miscellaneous.h"
#endif
#include <malloc.h>


#ifdef SSE
  #define MALLOC( variable, kind, length ) do{ if ( variable != NULL ) { \
  printf0("malloc of \"%s\" failed: pointer is not NULL (%s:%d).\n", #variable, __FILE__, __LINE__ ); } \
  if ( (length) > 0 ) { variable = (kind*) memalign( 64, sizeof(kind) * (length) ); } \
  if ( variable == NULL && (length) > 0 ) { \
  error0("malloc of \"%s\" failed: no memory allocated (%s:%d), current memory used: %lf GB.\n", \
  #variable, __FILE__, __LINE__, g.cur_storage/1024.0 ); } \
  g.cur_storage += (sizeof(kind) * (length))/(1024.0*1024.0); \
  if ( g.cur_storage > g.max_storage ) g.max_storage = g.cur_storage; }while(0)
#else
  #define MALLOC( variable, kind, length ) do{ if ( variable != NULL ) { \
  printf0("malloc of \"%s\" failed: pointer is not NULL (%s:%d).\n", #variable, __FILE__, __LINE__ ); } \
  if ( (length) > 0 ) { variable = (kind*) malloc( sizeof(kind) * (length) ); } \
  if ( variable == NULL && (length) > 0 ) { \
  error0("malloc of \"%s\" failed: no memory allocated (%s:%d), current memory used: %lf GB.\n", \
  #variable, __FILE__, __LINE__, g.cur_storage/1024.0 ); } \
  g.cur_storage += (sizeof(kind) * (length))/(1024.0*1024.0); \
  if ( g.cur_storage > g.max_storage ) g.max_storage = g.cur_storage; }while(0)
#endif

#ifdef CUDA_OPT
  #define CUDA_MALLOC( variable, kind, length ) do{ if ( variable != NULL ) { \
  printf0("cudaMalloc of \"%s\" failed: pointer is not NULL (%s:%d).\n", #variable, __FILE__, __LINE__ ); } \
  if ( (length) > 0 ) { cuda_safe_call( cudaMalloc( (void**) (&( variable )), length*sizeof(kind) ) ); } \
  if ( variable == NULL && (length) > 0 ) { \
  error0("cudaMalloc of \"%s\" failed: no memory allocated (%s:%d), current memory used: %lf GB.\n", \
  #variable, __FILE__, __LINE__, g.cur_gpu_storage/1024.0 ); } \
  g.cur_gpu_storage += (sizeof(kind) * (length))/(1024.0*1024.0); \
  if ( g.cur_gpu_storage > g.max_gpu_storage ) g.max_gpu_storage = g.cur_gpu_storage; }while(0)
#endif

  // allocate and deallocate macros (hugepages, aligned)
  #include <fcntl.h>
  #include <sys/mman.h>
  #define HUGE_PAGE_SIZE (2 * 1024 * 1024)
  #define ROUND_UP_TO_FULL_PAGE(x) \
    (((x) + HUGE_PAGE_SIZE - 1) / HUGE_PAGE_SIZE * HUGE_PAGE_SIZE)
  
  
//   void *tmppointer = (void*)(variable);
//   posix_memalign( &tmppointer, alignment, sizeof(kind)*(length));
//   variable = (kind*)tmppointer; }
  
  #define MALLOC_HUGEPAGES( variable, kind, length, alignment ) do { if ( variable != NULL ) { \
  printf0("malloc of \"%s\" failed: pointer is not NULL (%s:%d).\n", #variable, __FILE__, __LINE__ ); } \
  if ( (length) > 0 ) { \
  variable = (kind*)memalign( alignment, sizeof(kind)*((size_t)length)); } \
  if ( variable == NULL && (length) > 0 ) { \
  error0("malloc of \"%s\" failed: no memory allocated (%s:%d), current memory used: %lf GB.\n", \
  #variable, __FILE__, __LINE__, g.cur_storage/1024.0 ); } \
  g.cur_storage += (sizeof(kind) * (length))/(1024.0*1024.0); \
  if ( g.cur_storage > g.max_storage ) g.max_storage = g.cur_storage; }while(0)
  
  #define FREE_HUGEPAGES( addr, kind, length ) do { free( addr ); addr = NULL; \
  g.cur_storage -= (sizeof(kind) * (length))/(1024.0*1024.0); }while(0)

  #define FREE( variable, kind, length ) do{ if ( variable != NULL ) { \
  free( variable ); variable = NULL; g.cur_storage -= (sizeof(kind) * (length))/(1024.0*1024.0); } else { \
  printf0("multiple free of \"%s\"? pointer is already NULL (%s:%d).\n", #variable, __FILE__, __LINE__ ); } }while(0)

#ifdef CUDA_OPT
  #define CUDA_FREE( variable, kind, length ) do{ if ( variable != NULL ) { \
  cuda_safe_call( cudaFree( variable ) ); variable = NULL; g.cur_gpu_storage -= (sizeof(kind) * (length))/(1024.0*1024.0); } else { \
  printf0("multiple cudaFree of \"%s\"? pointer is already NULL (%s:%d).\n", #variable, __FILE__, __LINE__ ); } }while(0)
#endif
  
  #define PUBLIC_MALLOC( variable, kind, size ) do{ START_MASTER(threading) MALLOC( variable, kind, size ); \
  ((kind**)threading->workspace)[0] = variable; END_MASTER(threading) SYNC_MASTER_TO_ALL(threading) \
  variable = ((kind**)threading->workspace)[0]; SYNC_MASTER_TO_ALL(threading) }while(0)
  
  #define PUBLIC_FREE( variable, kind, size ) do{ SYNC_MASTER_TO_ALL(threading) \
  START_MASTER(threading) FREE( variable, kind, size ); END_MASTER(threading) SYNC_MASTER_TO_ALL(threading) variable = NULL; }while(0)
  
  #define PUBLIC_MALLOC2( variable, kind, size, thrdng ) do{ START_MASTER(thrdng) MALLOC( variable, kind, size ); \
  ((kind**)thrdng->workspace)[0] = variable; END_MASTER(thrdng) SYNC_MASTER_TO_ALL(thrdng) \
  variable = ((kind**)thrdng->workspace)[0]; SYNC_MASTER_TO_ALL(thrdng) }while(0)
  
  #define PUBLIC_FREE2( variable, kind, size, thrdng ) do{ SYNC_MASTER_TO_ALL(thrdng) \
  START_MASTER(thrdng) FREE( variable, kind, size ); END_MASTER(thrdng) SYNC_MASTER_TO_ALL(thrdng) variable = NULL; }while(0)

  #endif // ALLOC_CONTROL_H
