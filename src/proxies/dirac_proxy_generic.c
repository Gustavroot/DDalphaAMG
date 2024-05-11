#include "dirac_proxy_PRECISION.h"

#ifdef CUDA_OPT
#include "gpu/cuda_dirac_PRECISION.h"
#include "gpu/cuda_oddeven_PRECISION.h"
#endif

// this is here for testing purposes, for now,
// of CPU->GPU odd-even smoothers
#include "oddeven_PRECISION.h"

#include <complex.h>
#include "dirac_PRECISION.h"
#include "console_out.h"
#include "linalg_PRECISION.h"
#include "operator.h"

#include <string.h>
#include "data_PRECISION.h"

void d_plus_clover_PRECISION(vector_PRECISION eta, vector_PRECISION phi,
                             operator_PRECISION_struct *op, level_struct *l,
                             struct Thread *threading)
{
#ifdef CUDA_OPT
  cuda_d_plus_clover_PRECISION_vectorwrapper(eta, phi, op, l, threading);
#else
  d_plus_clover_PRECISION_cpu(eta, phi, op, l, threading);
#endif
}

void apply_schur_complement_PRECISION(vector_PRECISION out, vector_PRECISION in,
                                      operator_PRECISION_struct *op, level_struct *l,
                                      struct Thread *threading)
{

//#ifdef CUDA_OPT
//  cuda_apply_schur_complement_PRECISION_vectorwrapper(out, in, op, l, threading);
//#else
//  apply_schur_complement_PRECISION_cpu(out, in, op, l, threading);
//#endif

  int i;
  vector_PRECISION v1, v2, V1[8],V2[8];
  for ( i=0;i<8;i++ ) {
    //MALLOC( v1, complex_PRECISION, l->inner_vector_size );
    V1[i] = (vector_PRECISION) malloc( l->inner_vector_size * sizeof(complex_PRECISION) );
    //MALLOC( v2, complex_PRECISION, l->inner_vector_size );
    V2[i] = (vector_PRECISION) malloc( l->inner_vector_size * sizeof(complex_PRECISION) );
  }

  //memset( v1, 0, l->inner_vector_size * sizeof(complex_PRECISION) );
  //memset( v2, 0, l->inner_vector_size * sizeof(complex_PRECISION) );

  //for ( i=0;i<8;i++ ) {
  //  vector_PRECISION_define( V1[i], 0, 0, l->inner_vector_size, l );
  //  vector_PRECISION_define( V2[i], 0, 0, l->inner_vector_size, l );
  //}

  //vector_PRECISION_define_random( in, 0, l->inner_vector_size, l );

  //cuda_apply_schur_complement_PRECISION_vectorwrapper(out, in, op, l, threading);
  //apply_schur_complement_PRECISION_cpu_dummy(out, in, op, l, threading);

  //int start,end;
  //compute_core_start_end_custom(0, op->num_even_sites*l->num_lattice_site_var, &start, &end, l, threading, 12 );

  //size_t pbs = projection_buffer_size( l->num_lattice_site_var, l->num_lattice_sites );

  //cuda_apply_schur_complement_PRECISION_vectorwrapper(V1[0], in, op, l, threading);
  //vector_PRECISION_copy( V1[0], op->prnT, 0, pbs, l );
  //vector_PRECISION_copy( V1[1], op->prnX, 0, pbs, l );
  //vector_PRECISION_copy( V1[2], op->prnY, 0, pbs, l );
  //vector_PRECISION_copy( V1[3], op->prnZ, 0, pbs, l );
  //vector_PRECISION_copy( V1[4], op->prpT, 0, pbs, l );
  //vector_PRECISION_copy( V1[5], op->prpX, 0, pbs, l );
  //vector_PRECISION_copy( V1[6], op->prpY, 0, pbs, l );
  //vector_PRECISION_copy( V1[7], op->prpZ, 0, pbs, l );

  //apply_schur_complement_PRECISION_cpu_dummy(V2[0], in, op, l, threading);
  //vector_PRECISION_copy( V2[0], op->prnT, 0, pbs, l );
  //vector_PRECISION_copy( V2[1], op->prnX, 0, pbs, l );
  //vector_PRECISION_copy( V2[2], op->prnY, 0, pbs, l );
  //vector_PRECISION_copy( V2[3], op->prnZ, 0, pbs, l );
  //vector_PRECISION_copy( V2[4], op->prpT, 0, pbs, l );
  //vector_PRECISION_copy( V2[5], op->prpX, 0, pbs, l );
  //vector_PRECISION_copy( V2[6], op->prpY, 0, pbs, l );
  //vector_PRECISION_copy( V2[7], op->prpZ, 0, pbs, l );

  //printf0("l->inner_vector_size = %d\n",l->inner_vector_size);

  //printf0("pbs = %lu\n",pbs);

  //printf0( "--- checking first for buffers:\n" );

  //for ( i=0;i<8;i++ ) {

  //  v1 = V1[i];
  //  v2 = V2[i];

  //  vector_PRECISION_minus( v1, v1, v2, 0, l->inner_vector_size, l );
  //  double norm1 = global_norm_PRECISION( v1, 0, l->inner_vector_size, l, threading );
  //  double norm2 = global_norm_PRECISION( v2, 0, l->inner_vector_size, l, threading );

  //  if ( norm2!=0.0 ) {
  //    printf0("relative difference = %.12f\n",norm1/norm2);
  //  } else {
  //    printf0("numerator = %.12f\n",norm1);
  //    printf0("denominator = %.12f\n",norm2);
  //  }

    //for ( int i=0;i<l->inner_vector_size;i++ ) {
    //  if ( creal_PRECISION(v1[i])!=creal_PRECISION(v2[i]) && cimag_PRECISION(v1[i])!=cimag_PRECISION(v2[i]) ) {
    //    printf0("NEQ v1[%d,%d,%d] = %f+%f\n",i,i/12,op->neighbor_table[i/12],creal_PRECISION(v1[i]),cimag_PRECISION(v1[i]));
    //    printf0("NEQ v2[%d,%d,%d] = %f+%f\n",i,i/12,op->neighbor_table[i/12],creal_PRECISION(v2[i]),cimag_PRECISION(v2[i]));
    //  }
    //  // else {
    //  //  printf0("YEQ v1[%d,%d,%d] = %f+%f\n",i,i/12,op->neighbor_table[i/12],creal_PRECISION(v1[i]),cimag_PRECISION(v1[i]));
    //  //  printf0("YEQ v2[%d,%d,%d] = %f+%f\n",i,i/12,op->neighbor_table[i/12],creal_PRECISION(v2[i]),cimag_PRECISION(v2[i]));
    //  //}
    //}
  //}

  printf0( "--- checking now for output:\n" );

  for ( i=0;i<8;i++ ) {
    vector_PRECISION_define( V1[i], 0, 0, l->inner_vector_size, l );
    vector_PRECISION_define( V2[i], 0, 0, l->inner_vector_size, l );
  }

  vector_PRECISION_define_random( in, 0, l->inner_vector_size, l );

  cuda_apply_schur_complement_PRECISION_vectorwrapper(V1[0], in, op, l, threading);
  //apply_schur_complement_PRECISION_cpu_dummy(V2[0], in, op, l, threading);
  apply_schur_complement_PRECISION_cpu(V2[0], in, op, l, threading);

  {

    v1 = V1[0];
    v2 = V2[0];

    //for ( int i=0;i<l->inner_vector_size;i++ ) {
    //  if ( creal_PRECISION(v1[i])!=creal_PRECISION(v2[i]) && cimag_PRECISION(v1[i])!=cimag_PRECISION(v2[i]) ) {
    //    printf0("NEQ v1[%d,%d,%d] = %f+%f\n",i,i/12,op->neighbor_table[i/12],creal_PRECISION(v1[i]),cimag_PRECISION(v1[i]));
    //    printf0("NEQ v2[%d,%d,%d] = %f+%f\n",i,i/12,op->neighbor_table[i/12],creal_PRECISION(v2[i]),cimag_PRECISION(v2[i]));
    //  } else {
    //    printf0("YEQ v1[%d,%d,%d] = %f+%f\n",i,i/12,op->neighbor_table[i/12],creal_PRECISION(v1[i]),cimag_PRECISION(v1[i]));
    //    printf0("YEQ v2[%d,%d,%d] = %f+%f\n",i,i/12,op->neighbor_table[i/12],creal_PRECISION(v2[i]),cimag_PRECISION(v2[i]));
    //  }
    //}

    vector_PRECISION_minus( v1, v1, v2, 0, l->inner_vector_size/2, l );
    double norm1 = global_norm_PRECISION( v1, 0, l->inner_vector_size/2, l, threading );
    double norm2 = global_norm_PRECISION( v2, 0, l->inner_vector_size/2, l, threading );

    if ( norm2!=0.0 ) {
      printf0("relative difference = %.12f\n",norm1/norm2);
    } else {
      printf0("numerator = %.12f\n",norm1);
      printf0("denominator = %.12f\n",norm2);
    }
  }

  //vector_PRECISION_minus( v1, v1, v2, start, end, l );
  //double norm1 = global_norm_PRECISION( v1, 0, op->num_even_sites*l->num_lattice_site_var, l, threading );
  //double norm2 = global_norm_PRECISION( v2, 0, op->num_even_sites*l->num_lattice_site_var, l, threading );

  //vector_PRECISION_minus( v1, v1, v2, 0, l->inner_vector_size, l );
  //double norm1 = global_norm_PRECISION( v1, 0, l->inner_vector_size, l, threading );
  //double norm2 = global_norm_PRECISION( v2, 0, l->inner_vector_size, l, threading );

  //if ( norm2!=0.0 ) {
  //  printf0("relative difference = %.12f\n",norm1/norm2);
  //} else {
  //  printf0("numerator = %.12f\n",norm1);
  //  printf0("denominator = %.12f\n",norm2);
  //}

  //FREE( v1, complex_PRECISION, l->inner_vector_size );
  //free(v1);
  //FREE( v2, complex_PRECISION, l->inner_vector_size );
  //free(v2);

  MPI_Finalize();
  exit(0);
}
