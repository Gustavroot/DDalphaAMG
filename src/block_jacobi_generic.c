/*
 * Copyright (C) 2016, Matthias Rottmann, Artur Strebel, Simon Heybrock, Simone Bacchio, Bjoern Leder.
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

#include "main.h"

//#ifdef BLOCK_JACOBI
#if 0

  // main functions here -- IMPORTANT : all Block Jacobi functions are, for now,
  //			                exclusively per-process operations

#ifdef BJ_DIR_SOLVS
  void bj_direct_op_apply_PRECISION( vector_PRECISION out, vector_PRECISION in, level_struct *l, struct Thread *threading ){

    int start, end;
    compute_core_start_end_custom( 0, l->p_PRECISION.op->num_even_sites, &start, &end, l, threading, 1 );

    int site_size = l->num_lattice_site_var;
    int lda = SIMD_LENGTH_PRECISION*((site_size+SIMD_LENGTH_PRECISION-1)/SIMD_LENGTH_PRECISION);
#ifdef HAVE_TM1p1
    OPERATOR_TYPE_PRECISION *clover = 
                            (g.n_flavours == 2) ? l->p_PRECISION.block_jacobi_PRECISION.bj_doublet_op_inv_vectorized:l->p_PRECISION.block_jacobi_PRECISION.bj_op_inv_vectorized;
#else
    OPERATOR_TYPE_PRECISION *clover = l->p_PRECISION.block_jacobi_PRECISION.bj_op_inv_vectorized;
#endif
    for(int i=start; i<end; i++) {
      for(int j=0; j<site_size; j++)
        out[i*site_size+j] = 0.0;
      cgemv(site_size, clover+i*2*site_size*lda, lda, (float *)(in+i*site_size), (float *)(out+i*site_size));
    }
  }
#endif

  void block_jacobi_apply_PRECISION( vector_PRECISION out, vector_PRECISION in, gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading ){
    if ( out==in ) { return; }

#ifndef OPTIMIZED_COARSE_SELF_COUPLING_PRECISION

    if ( p->block_jacobi_PRECISION.BJ_usable==1 ) {
      local_apply_polyprec_PRECISION( out, NULL, in, 0, l, threading );
    }

#else

#ifdef BJ_DIR_SOLVS
    bj_direct_op_apply_PRECISION( out, in, l, threading );
#else
    if ( p->block_jacobi_PRECISION.BJ_usable==1 ) {
      local_apply_polyprec_PRECISION( out, NULL, in, 0, l, threading );
    }
#endif

#endif
  }

#endif
