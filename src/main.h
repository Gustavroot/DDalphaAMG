/*
 * Copyright (C) 2016, Matthias Rottmann, Artur Strebel, Gustavo Ramirez, Simon Heybrock, Simone Bacchio, Bjoern Leder, Issaku Kanamori, Tilmann Matthaei, Ke-Long Zhang.
 * 
 * This file is part of the DDalphaAMG solver library.
 * 
 * The DDalphaAMG solver library is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * The DDalphaAMG solver library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * 
 * You should have received a copy of the GNU General Public License
 * along with the DDalphaAMG solver library. If not, see http://www.gnu.org/licenses/.
 * 
 */

#ifdef CUDA_OPT
  #include <cuda.h>
  #include <cuda_runtime.h>
#endif

#ifdef HALF_PREC_STORAGE
  #include <stdint.h>
#endif

#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include "stdbool.h"
#ifndef IMPORT_FROM_EXTERN_C
  #include <mpi.h>
#endif
#include <complex.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>
#include <sys/time.h>

#ifdef JUROPA
#include <mkl.h>
#endif

#include "dd_alpha_amg_parameters.h"
#include "dd_alpha_amg_setup_status.h"

#ifndef MAIN_HEADER
#define MAIN_HEADER

#include "global_defs.h"
#include "global_enums.h"  
#include "console_out.h"
#include "alloc_control.h"


  
  #ifdef DEBUG
    #define DPRINTF0 printf0
  #else
    #define DPRINTF0( ARGS, ... )
  #endif

  #include "util_macros.h"
  
  #ifdef DEBUG
  #define DEBUGOUTPUT_ARRAY( A, FORMAT, INDEX ) do{ \
  char TMPSTRING[100]; sprintf( TMPSTRING, "%s[%d] = %s\n", #A, INDEX, FORMAT ); \
  printf0( TMPSTRING, A[INDEX] ); }while(0)
  #else
  #define DEBUGOUTPUT_ARRAY( A, FORMAT, INDEX )
  #endif
  
  #ifdef DEBUG
  #define DEBUGOUTPUT( A, FORMAT ) do{ \
  char TMPSTRING[100]; sprintf( TMPSTRING, "%s = %s\n", #A, FORMAT ); \
  printf0( TMPSTRING, A ); }while(0)
  #else
  #define DEBUGOUTPUT( A, FORMAT )
  #endif

  #include "vectorization_control.h"
  #include "threading.h"

#include "block_struct.h"

  #include "main_pre_def_float.h"
  #include "main_pre_def_double.h"
  
  typedef struct confbuffer_struct {
    
    double *data;
    struct confbuffer_struct *next;
    
  } confbuffer_struct;

  #include "global_struct.h"
  #include "algorithm_structs.h"
  
#endif

#ifdef SSE
#include "blas_vectorized.h"
#include "sse_blas_vectorized.h"
#include "sse_complex_float_intrinsic.h"
#include "sse_complex_double_intrinsic.h"
#include "sse_coarse_operator_float.h"
#include "sse_coarse_operator_double.h"
#include "sse_linalg_float.h"
#include "sse_linalg_double.h"
#include "sse_interpolation_float.h"
#include "sse_interpolation_double.h"
#include "sse_schwarz_float.h"
#include "sse_schwarz_double.h"
#else
//no intrinsics
#include "interpolation_float.h"
#include "interpolation_double.h"
#endif

#include "data_float.h"
#include "data_double.h"
#include "data_layout.h"
#include "io.h"
#include "init.h"
#include "operator_float.h"
#include "operator_double.h"
#include "dirac.h"
#ifndef IMPORT_FROM_EXTERN_C
  #include "dirac_float.h"
  #include "dirac_double.h"
#endif
#include "oddeven_float.h"
#include "oddeven_double.h"
#include "linalg.h"
#include "linalg_float.h"
#include "linalg_double.h"
#include "ghost_float.h"
#include "ghost_double.h"
#include "linsolve_float.h"
#include "linsolve_double.h"
#include "linsolve.h"
#ifndef IMPORT_FROM_EXTERN_C
  #include "preconditioner.h"
  #include "vcycle_float.h"
  #include "vcycle_double.h"
#endif
#include "solver_analysis.h"
#include "top_level.h"
#include "ghost.h"
#include "init_float.h"
#include "init_double.h"
#include "schwarz_double.h"
#include "schwarz_float.h"
#include "setup_float.h"
#include "setup_double.h"
#include "coarsening_float.h"
#include "coarsening_double.h"
#include "gathering_float.h"
#include "gathering_double.h"
#ifndef IMPORT_FROM_EXTERN_C
  #include "coarse_operator_float.h"
  #include "coarse_operator_double.h"
#endif
#include "coarse_oddeven_float.h"
#include "coarse_oddeven_double.h"
#include "var_table.h"
#ifndef IMPORT_FROM_EXTERN_C
  #include "main_post_def_float.h"
  #include "main_post_def_double.h"
#endif
#ifdef HAVE_LIME
#include <lime.h>
#include <lime_config.h>
#include <dcap-overload.h>
#include <lime_defs.h>
#include <lime_header.h>
#include <lime_writer.h>
#include <lime_reader.h>
#endif
#include "lime_io.h"
#ifdef CUDA_OPT
  #include "gpu/cuda_dirac_float.h"
  #include "gpu/cuda_dirac_double.h"
  #include "gpu/cuda_oddeven_float.h"
  #include "gpu/cuda_oddeven_double.h"
  #include "gpu/cuda_linalg_float.h"
  #include "gpu/cuda_linalg_double.h"
  #include "gpu/cuda_oddeven_linalg_float.h"
  #include "gpu/cuda_oddeven_linalg_double.h"
  #include "gpu/cuda_linsolve_float.h"
  #include "gpu/cuda_linsolve_double.h"
  #include "gpu/cuda_schwarz_double.h"
  #include "gpu/cuda_schwarz_float.h"
  #include "gpu/cuda_coarse_oddeven_float.h"
  #include "gpu/cuda_coarse_oddeven_double.h"
  #include "gpu/cuda_coarse_operator_float.h"
  #include "gpu/cuda_coarse_operator_double.h"
  #include "gpu/cuda_miscellaneous.h"
#endif

#ifdef HALF_PREC_STORAGE
  #include "utils_half_precision.h"
#endif

#include "miscellaneous.h"
#if defined(GCRODR) || defined(POLYPREC)
#include "miscellaneous_double.h"
#include "miscellaneous_float.h"
#endif

#ifdef GCRODR
  #include "gcrodr_double.h"
  #include "gcrodr_float.h"
#endif

//#ifdef BLOCK_JACOBI
#if 0
  #include "block_jacobi_double.h"
  #include "block_jacobi_float.h"
  #include "local_polyprec_double.h"
  #include "local_polyprec_float.h"
#endif

#if defined(GCRODR) || defined(POLYPREC)
  #include <lapacke.h>
#ifdef GCRODR
  ////#include <mkl_scalapack.h>
  ////#include <mkl_blacs.h>
  ////#include <mkl_pblas.h>
#endif
  #include "lapackwrap_double.h"
  #include "lapackwrap_float.h"
#endif

#ifdef POLYPREC
  #include "polyprec_double.h"
  #include "polyprec_float.h"
#endif
