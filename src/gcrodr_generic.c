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
#include "proxies/linsolve_proxy_PRECISION.h"

#ifdef GCRODR


// declarations of aux functions
int  fgmresx_PRECISION( gmres_PRECISION_struct*, level_struct*, struct Thread* );
void gev_buildAB_PRECISION(complex_PRECISION**, complex_PRECISION**, complex_PRECISION**, vector_PRECISION*, vector_PRECISION*,
                           int, gmres_PRECISION_struct*, level_struct*, struct Thread*);
void order_pairs_PRECISION( vector_PRECISION, complex_PRECISION*, int*, int );
void build_CU_PRECISION( complex_PRECISION**, vector_PRECISION*, vector_PRECISION*,
                         gmres_PRECISION_struct*, level_struct*, struct Thread*, int );
int  arnoldix_step_PRECISION( vector_PRECISION *V, vector_PRECISION *Z, vector_PRECISION w,
                              complex_PRECISION **H, complex_PRECISION* buffer, int j, void (*prec)(),
                              gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading );
void re_scale_Uk_PRECISION( gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading );


// -------------------------------------------------------------



void flgcrodr_PRECISION_struct_init( gmres_PRECISION_struct *p ) {

  //printf0( "Entered GCRO-DR\n" );
  //exit(0);

  fgmres_PRECISION_struct_init( p );

  p->gcrodr_PRECISION.eigslvr.ordr_idxs = NULL;
  p->gcrodr_PRECISION.eigslvr.ordr_keyscpy = NULL;

  p->gcrodr_PRECISION.C = NULL;
  p->gcrodr_PRECISION.Cc = NULL;
  p->gcrodr_PRECISION.U = NULL;
  p->gcrodr_PRECISION.gev_A = NULL;
  p->gcrodr_PRECISION.gev_B = NULL;
  p->gcrodr_PRECISION.Bbuff = NULL;
  p->gcrodr_PRECISION.eigslvr.w = NULL;
  p->gcrodr_PRECISION.eigslvr.beta = NULL;
  p->gcrodr_PRECISION.eigslvr.vl = NULL;
  p->gcrodr_PRECISION.eigslvr.vr = NULL;
  p->gcrodr_PRECISION.eigslvr.qr_tau = NULL;
  p->gcrodr_PRECISION.lsp_x = NULL;
  p->gcrodr_PRECISION.lsp_diag_G = NULL;
  p->gcrodr_PRECISION.lsp_H = NULL;

  p->gcrodr_PRECISION.Yk = NULL;
  p->gcrodr_PRECISION.Pk = NULL;

  p->gcrodr_PRECISION.QR = NULL;
  p->gcrodr_PRECISION.Q = NULL;
  p->gcrodr_PRECISION.R = NULL;
  p->gcrodr_PRECISION.Rinv = NULL;

  p->gcrodr_PRECISION.ort_B = NULL;
  p->gcrodr_PRECISION.G = NULL;
  p->gcrodr_PRECISION.Gc = NULL;
  p->gcrodr_PRECISION.hatZ = NULL;
  p->gcrodr_PRECISION.hatW = NULL;
//#ifdef BLOCK_JACOBI
#if 0
  p->gcrodr_PRECISION.r_aux = NULL;
#endif

  p->rhs_bk = NULL;

#if defined(SINGLE_ALLREDUCE_ARNOLDI) && defined(PIPELINED_ARNOLDI)
  p->gcrodr_PRECISION.PC = NULL;
  p->gcrodr_PRECISION.DPC = NULL;
#endif
}


void flgcrodr_PRECISION_struct_alloc( int m, int n, long int vl, PRECISION tol, const int type, const int prec_kind,
                                      void (*precond)(), void (*eval_op)(), gmres_PRECISION_struct *p, level_struct *l ) {

  fgmres_PRECISION_struct_alloc( m, n, vl, tol, type, prec_kind, precond, eval_op, p, l );

  if ( l->level==0 ) {

    if ( g.gcrodr_k >= p->restart_length ) {
      error0("The value of k in GCRO-DR needs to be smaller than the restart length m\n");
    }
    p->gcrodr_PRECISION.k = g.gcrodr_k;

    // g_ln is the length m+k of subspaces used in FL-GCRO-DR
    int g_ln = p->restart_length + p->gcrodr_PRECISION.k;
    int i;

    MALLOC( p->gcrodr_PRECISION.Bbuff, complex_PRECISION*, g_ln );

    // for large LSP
    MALLOC( p->gcrodr_PRECISION.lsp_x, complex_PRECISION, g_ln+1 );
    MALLOC( p->gcrodr_PRECISION.lsp_diag_G, complex_PRECISION, p->gcrodr_PRECISION.k );
    MALLOC( p->gcrodr_PRECISION.lsp_H, complex_PRECISION*, p->restart_length );
    MALLOC( p->gcrodr_PRECISION.lsp_H[0], complex_PRECISION, (p->restart_length+1)*p->restart_length );
    for ( i=1;i<m;i++ ) {
      p->gcrodr_PRECISION.lsp_H[i] = p->gcrodr_PRECISION.lsp_H[0] + i*(p->restart_length+1);
    }

    p->gcrodr_PRECISION.Bbuff[0] = NULL;
    MALLOC( p->gcrodr_PRECISION.Bbuff[0], complex_PRECISION, g_ln*(g_ln+1)*2 );
    for ( i=1; i<g_ln; i++ ) {
      p->gcrodr_PRECISION.Bbuff[i] = p->gcrodr_PRECISION.Bbuff[0] + i*(g_ln+1);
    }

    MALLOC( p->gcrodr_PRECISION.eigslvr.ordr_idxs, int, g_ln );
    MALLOC( p->gcrodr_PRECISION.eigslvr.ordr_keyscpy, complex_PRECISION, g_ln );

    p->gcrodr_PRECISION.CU_usable = 0;

    // allocating C and U, which will contain the info associated to the recycling subspace
    MALLOC( p->gcrodr_PRECISION.C, vector_PRECISION, p->gcrodr_PRECISION.k );
    p->gcrodr_PRECISION.C[0] = NULL;
    MALLOC( p->gcrodr_PRECISION.C[0], complex_PRECISION, vl * p->gcrodr_PRECISION.k );
    for ( i=1; i<p->gcrodr_PRECISION.k; i++ ) {
      p->gcrodr_PRECISION.C[i] = p->gcrodr_PRECISION.C[0] + i*vl;
    }
    MALLOC( p->gcrodr_PRECISION.Cc, vector_PRECISION, p->gcrodr_PRECISION.k );
    p->gcrodr_PRECISION.Cc[0] = NULL;
    MALLOC( p->gcrodr_PRECISION.Cc[0], complex_PRECISION, vl * p->gcrodr_PRECISION.k );
    for ( i=1; i<p->gcrodr_PRECISION.k; i++ ) {
      p->gcrodr_PRECISION.Cc[i] = p->gcrodr_PRECISION.Cc[0] + i*vl;
    }
    MALLOC( p->gcrodr_PRECISION.U, vector_PRECISION, p->gcrodr_PRECISION.k );
    p->gcrodr_PRECISION.U[0] = NULL;
    MALLOC( p->gcrodr_PRECISION.U[0], complex_PRECISION, vl * p->gcrodr_PRECISION.k );
    for ( i=1; i<p->gcrodr_PRECISION.k; i++ ) {
      p->gcrodr_PRECISION.U[i] = p->gcrodr_PRECISION.U[0] + i*vl;
    }

//#ifdef BLOCK_JACOBI
#if 0
    MALLOC( p->gcrodr_PRECISION.r_aux, complex_PRECISION, vl );
#endif

    MALLOC( p->gcrodr_PRECISION.gev_A, complex_PRECISION*, g_ln );
    MALLOC( p->gcrodr_PRECISION.gev_B, complex_PRECISION*, g_ln );
    p->gcrodr_PRECISION.gev_A[0] = NULL;
    p->gcrodr_PRECISION.gev_B[0] = NULL;
    MALLOC( p->gcrodr_PRECISION.gev_A[0], complex_PRECISION, g_ln*g_ln );
    MALLOC( p->gcrodr_PRECISION.gev_B[0], complex_PRECISION, g_ln*g_ln );
    for ( i=1; i<g_ln; i++ ) {
      p->gcrodr_PRECISION.gev_A[i] = p->gcrodr_PRECISION.gev_A[0] + i*g_ln;
      p->gcrodr_PRECISION.gev_B[i] = p->gcrodr_PRECISION.gev_B[0] + i*g_ln;
    }

    MALLOC( p->gcrodr_PRECISION.eigslvr.w, complex_PRECISION, g_ln );
    MALLOC( p->gcrodr_PRECISION.eigslvr.beta, complex_PRECISION, g_ln );

    MALLOC( p->gcrodr_PRECISION.eigslvr.vr, complex_PRECISION, g_ln*g_ln );

    // setting values for using the Generalized Eigenvalue Solver from LAPACK
    p->gcrodr_PRECISION.eigslvr.jobvl = 'N';
    p->gcrodr_PRECISION.eigslvr.jobvr = 'V';
    p->gcrodr_PRECISION.eigslvr.A = p->gcrodr_PRECISION.gev_A[0];
    p->gcrodr_PRECISION.eigslvr.lda = p->restart_length + p->gcrodr_PRECISION.k;
    p->gcrodr_PRECISION.eigslvr.B = p->gcrodr_PRECISION.gev_B[0];
    p->gcrodr_PRECISION.eigslvr.ldb = p->restart_length + p->gcrodr_PRECISION.k;
    p->gcrodr_PRECISION.eigslvr.ldvl = p->restart_length + p->gcrodr_PRECISION.k;
    p->gcrodr_PRECISION.eigslvr.ldvr = p->restart_length + p->gcrodr_PRECISION.k;
  
    // matrix Y containing Yk = Zm * Pk
    MALLOC( p->gcrodr_PRECISION.Yk, complex_PRECISION*, p->gcrodr_PRECISION.k );
    p->gcrodr_PRECISION.Yk[0] = NULL;
    MALLOC( p->gcrodr_PRECISION.Yk[0], complex_PRECISION, vl * p->gcrodr_PRECISION.k );
    for ( i=1; i<p->gcrodr_PRECISION.k; i++ ) {
      p->gcrodr_PRECISION.Yk[i] = p->gcrodr_PRECISION.Yk[0] + i*vl;
    }

    MALLOC( p->gcrodr_PRECISION.Pk, complex_PRECISION*, p->gcrodr_PRECISION.k );

    p->gcrodr_PRECISION.syst_size = vl;

    MALLOC( p->gcrodr_PRECISION.QR, complex_PRECISION*, p->gcrodr_PRECISION.k );
    p->gcrodr_PRECISION.QR[0] = NULL;
    MALLOC( p->gcrodr_PRECISION.QR[0], complex_PRECISION, (g_ln+1) * p->gcrodr_PRECISION.k );
    for ( i=1; i<p->gcrodr_PRECISION.k; i++ ) {
      p->gcrodr_PRECISION.QR[i] = p->gcrodr_PRECISION.QR[0] + i*(g_ln+1);
    }

    p->gcrodr_PRECISION.eigslvr.qr_QR = p->gcrodr_PRECISION.QR;
    p->gcrodr_PRECISION.eigslvr.qr_lda = g_ln+1;

    p->gcrodr_PRECISION.Q = p->gcrodr_PRECISION.QR;
    p->gcrodr_PRECISION.eigslvr.qr_Q = p->gcrodr_PRECISION.Q;

    MALLOC( p->gcrodr_PRECISION.R, complex_PRECISION*, p->gcrodr_PRECISION.k );
    p->gcrodr_PRECISION.R[0] = NULL;
    MALLOC( p->gcrodr_PRECISION.R[0], complex_PRECISION, p->gcrodr_PRECISION.k * p->gcrodr_PRECISION.k );
    for ( i=1; i<p->gcrodr_PRECISION.k; i++ ) {
      p->gcrodr_PRECISION.R[i] = p->gcrodr_PRECISION.R[0] + i * p->gcrodr_PRECISION.k;
    }

    p->gcrodr_PRECISION.Rinv = p->gcrodr_PRECISION.R;
    p->gcrodr_PRECISION.eigslvr.qr_R = p->gcrodr_PRECISION.R;
    p->gcrodr_PRECISION.eigslvr.qr_Rinv = p->gcrodr_PRECISION.R;

    MALLOC( p->gcrodr_PRECISION.eigslvr.qr_tau, complex_PRECISION, p->gcrodr_PRECISION.k );
  
    MALLOC( p->gcrodr_PRECISION.ort_B, complex_PRECISION*, g_ln );
    p->gcrodr_PRECISION.ort_B[0] = NULL;
    MALLOC( p->gcrodr_PRECISION.ort_B[0], complex_PRECISION, p->gcrodr_PRECISION.k * g_ln );
    for ( i=1; i<g_ln; i++ ) {
      p->gcrodr_PRECISION.ort_B[i] = p->gcrodr_PRECISION.ort_B[0] + i * p->gcrodr_PRECISION.k;
    }

    MALLOC( p->gcrodr_PRECISION.G, complex_PRECISION*, g_ln );
    p->gcrodr_PRECISION.G[0] = NULL;
    MALLOC( p->gcrodr_PRECISION.G[0], complex_PRECISION, (g_ln+1) * g_ln );
    for ( i=1; i<g_ln; i++ ) {
      p->gcrodr_PRECISION.G[i] = p->gcrodr_PRECISION.G[0] + i * (g_ln+1);
    }
    memset( p->gcrodr_PRECISION.G[0], 0.0, sizeof(complex_PRECISION)*g_ln*(g_ln+1) );
    MALLOC( p->gcrodr_PRECISION.Gc, complex_PRECISION*, g_ln );
    p->gcrodr_PRECISION.Gc[0] = NULL;
    MALLOC( p->gcrodr_PRECISION.Gc[0], complex_PRECISION, (g_ln+1) * g_ln );
    for ( i=1; i<g_ln; i++ ) {
      p->gcrodr_PRECISION.Gc[i] = p->gcrodr_PRECISION.Gc[0] + i * (g_ln+1);
    }
    memset( p->gcrodr_PRECISION.Gc[0], 0.0, sizeof(complex_PRECISION)*g_ln*(g_ln+1) );

    MALLOC( p->gcrodr_PRECISION.hatZ, complex_PRECISION*, g_ln );
    MALLOC( p->gcrodr_PRECISION.hatW, complex_PRECISION*, g_ln+1 );

    for ( i=0; i<p->gcrodr_PRECISION.k; i++ ) {
      p->gcrodr_PRECISION.hatZ[i] = p->gcrodr_PRECISION.U[i];
      p->gcrodr_PRECISION.hatW[i] = p->gcrodr_PRECISION.C[i];
    }
    for ( i=p->gcrodr_PRECISION.k; i<g_ln; i++ ) {
      if ( precond==NULL ) {
        p->gcrodr_PRECISION.hatZ[i] = p->V[i - p->gcrodr_PRECISION.k];
      } else {
        p->gcrodr_PRECISION.hatZ[i] = p->Z[i - p->gcrodr_PRECISION.k];
      }
      p->gcrodr_PRECISION.hatW[i] = p->V[i - p->gcrodr_PRECISION.k];
    }
    p->gcrodr_PRECISION.hatW[g_ln] = p->V[p->restart_length];
  
    p->gcrodr_PRECISION.orth_against_Ck = 0;
    p->gcrodr_PRECISION.finish = 0;

    p->gcrodr_PRECISION.update_CU = 1;

    p->gcrodr_PRECISION.upd_ctr = 0;

    p->was_there_stagnation = 0;

    MALLOC( p->rhs_bk, complex_PRECISION, vl );

#if defined(SINGLE_ALLREDUCE_ARNOLDI) && defined(PIPELINED_ARNOLDI)
    MALLOC( p->gcrodr_PRECISION.PC, vector_PRECISION, p->gcrodr_PRECISION.k );
    p->gcrodr_PRECISION.PC[0] = NULL;
    MALLOC( p->gcrodr_PRECISION.PC[0], complex_PRECISION, vl * p->gcrodr_PRECISION.k );

    MALLOC( p->gcrodr_PRECISION.DPC, vector_PRECISION, p->gcrodr_PRECISION.k );
    p->gcrodr_PRECISION.DPC[0] = NULL;
    MALLOC( p->gcrodr_PRECISION.DPC[0], complex_PRECISION, vl * p->gcrodr_PRECISION.k );

    for ( i=1; i<p->gcrodr_PRECISION.k; i++ )
    {
      p->gcrodr_PRECISION.PC[i] = p->gcrodr_PRECISION.PC[0] + i*vl;
      p->gcrodr_PRECISION.DPC[i] = p->gcrodr_PRECISION.DPC[0] + i*vl;
    }    

    p->gcrodr_PRECISION.recompute_DPCk_poly = 0;
    p->gcrodr_PRECISION.recompute_DPCk_plain = 0;
#endif
  }

}


void flgcrodr_PRECISION_struct_free( gmres_PRECISION_struct *p, level_struct *l ) {

  fgmres_PRECISION_struct_free( p, l );

  if ( l->level==0 ) {
    // g_ln is the length m+k of subspaces used in FL-GCRO-DR
    int g_ln = p->restart_length + p->gcrodr_PRECISION.k;

    // GCRO-DR specific
    FREE( p->gcrodr_PRECISION.Yk[0], complex_PRECISION, p->gcrodr_PRECISION.syst_size * p->gcrodr_PRECISION.k );
    FREE( p->gcrodr_PRECISION.Yk, complex_PRECISION*, p->gcrodr_PRECISION.k );
    FREE( p->gcrodr_PRECISION.C[0], complex_PRECISION, p->gcrodr_PRECISION.syst_size * p->gcrodr_PRECISION.k );
    FREE( p->gcrodr_PRECISION.C, vector_PRECISION, p->gcrodr_PRECISION.k );
    FREE( p->gcrodr_PRECISION.Cc[0], complex_PRECISION, p->gcrodr_PRECISION.syst_size * p->gcrodr_PRECISION.k );
    FREE( p->gcrodr_PRECISION.Cc, vector_PRECISION, p->gcrodr_PRECISION.k );
    FREE( p->gcrodr_PRECISION.U[0], complex_PRECISION, p->gcrodr_PRECISION.syst_size * p->gcrodr_PRECISION.k );
    FREE( p->gcrodr_PRECISION.U, vector_PRECISION, p->gcrodr_PRECISION.k );
    FREE( p->gcrodr_PRECISION.Pk, complex_PRECISION*, p->gcrodr_PRECISION.k );
    FREE( p->gcrodr_PRECISION.ort_B[0], complex_PRECISION, p->gcrodr_PRECISION.k * g_ln );
    FREE( p->gcrodr_PRECISION.ort_B, complex_PRECISION*, g_ln );
    FREE( p->gcrodr_PRECISION.G[0], complex_PRECISION, (g_ln+1) * g_ln );
    FREE( p->gcrodr_PRECISION.G, complex_PRECISION*, g_ln );
    FREE( p->gcrodr_PRECISION.Gc[0], complex_PRECISION, (g_ln+1) * g_ln );
    FREE( p->gcrodr_PRECISION.Gc, complex_PRECISION*, g_ln );
    FREE( p->gcrodr_PRECISION.hatZ, complex_PRECISION*, g_ln );
    FREE( p->gcrodr_PRECISION.hatW, complex_PRECISION*, g_ln+1 );
//#ifdef BLOCK_JACOBI
#if 0
    FREE( p->gcrodr_PRECISION.r_aux, complex_PRECISION, p->gcrodr_PRECISION.syst_size );
#endif

    // for eigensolving
    FREE( p->gcrodr_PRECISION.Bbuff[0], complex_PRECISION, g_ln*(g_ln+1)*2 );
    FREE( p->gcrodr_PRECISION.Bbuff, complex_PRECISION*, g_ln );
    FREE( p->gcrodr_PRECISION.gev_A[0], complex_PRECISION, g_ln*g_ln );
    FREE( p->gcrodr_PRECISION.gev_B[0], complex_PRECISION, g_ln*g_ln );
    FREE( p->gcrodr_PRECISION.gev_A, complex_PRECISION*, g_ln );
    FREE( p->gcrodr_PRECISION.gev_B, complex_PRECISION*, g_ln );

    // for QR
    FREE( p->gcrodr_PRECISION.QR[0], complex_PRECISION, (g_ln+1) * p->gcrodr_PRECISION.k );
    FREE( p->gcrodr_PRECISION.QR, complex_PRECISION*, p->gcrodr_PRECISION.k );
    FREE( p->gcrodr_PRECISION.R[0], complex_PRECISION, p->gcrodr_PRECISION.k * p->gcrodr_PRECISION.k );
    FREE( p->gcrodr_PRECISION.R, complex_PRECISION*, p->gcrodr_PRECISION.k );
    FREE( p->gcrodr_PRECISION.eigslvr.qr_tau, complex_PRECISION, p->gcrodr_PRECISION.k );

    // vectors - QR
    FREE( p->gcrodr_PRECISION.eigslvr.vr, complex_PRECISION, g_ln*g_ln );
    FREE( p->gcrodr_PRECISION.eigslvr.w, complex_PRECISION, g_ln );
    FREE( p->gcrodr_PRECISION.eigslvr.beta, complex_PRECISION, g_ln );

    // for large LSP
    FREE( p->gcrodr_PRECISION.lsp_x, complex_PRECISION, g_ln+1 );
    FREE( p->gcrodr_PRECISION.lsp_diag_G, complex_PRECISION, p->gcrodr_PRECISION.k );
    FREE( p->gcrodr_PRECISION.lsp_H[0], complex_PRECISION, (p->restart_length+1)*p->restart_length );
    FREE( p->gcrodr_PRECISION.lsp_H, complex_PRECISION*, p->restart_length );

    // ints - ordering
    FREE( p->gcrodr_PRECISION.eigslvr.ordr_idxs, int, g_ln );
    FREE( p->gcrodr_PRECISION.eigslvr.ordr_keyscpy, complex_PRECISION, g_ln );

    FREE( p->rhs_bk, complex_PRECISION, p->gcrodr_PRECISION.syst_size );

#if defined(SINGLE_ALLREDUCE_ARNOLDI) && defined(PIPELINED_ARNOLDI)
    FREE( p->gcrodr_PRECISION.PC[0], complex_PRECISION, p->gcrodr_PRECISION.syst_size * p->gcrodr_PRECISION.k );
    FREE( p->gcrodr_PRECISION.PC, vector_PRECISION, p->gcrodr_PRECISION.k );

    FREE( p->gcrodr_PRECISION.DPC[0], complex_PRECISION, p->gcrodr_PRECISION.syst_size * p->gcrodr_PRECISION.k );
    FREE( p->gcrodr_PRECISION.DPC, vector_PRECISION, p->gcrodr_PRECISION.k );
#endif    
  }
}


// ASSUMING : right preconditioner (or no preconditioner)
int flgcrodr_PRECISION( gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading ){

  // in this context, <m> changes! It is not (necessarily) p->restart_length
  int fgmresx_iter=0, m=0, j, ol, k=p->gcrodr_PRECISION.k, i, g_ln, start, end;
  PRECISION beta=0;

  int buff_init_guess = p->initial_guess_zero;

  g_ln = p->restart_length + p->gcrodr_PRECISION.k;

  START_MASTER(threading)
  for ( i=0; i<p->gcrodr_PRECISION.k; i++ ) {
    p->gcrodr_PRECISION.hatZ[i] = p->gcrodr_PRECISION.U[i];
    p->gcrodr_PRECISION.hatW[i] = p->gcrodr_PRECISION.C[i];
  }
  for ( i=p->gcrodr_PRECISION.k; i<g_ln; i++ ) {
    if ( p->preconditioner==NULL ) {
      p->gcrodr_PRECISION.hatZ[i] = p->V[i - p->gcrodr_PRECISION.k];
    } else {
      p->gcrodr_PRECISION.hatZ[i] = p->Z[i - p->gcrodr_PRECISION.k];
    }
    p->gcrodr_PRECISION.hatW[i] = p->V[i - p->gcrodr_PRECISION.k];
  }
  p->gcrodr_PRECISION.hatW[g_ln] = p->V[p->restart_length];
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)

  // compute start and end indices for core
  // this puts zero for all other hyperthreads, so we can call functions below with all hyperthreads
  compute_core_start_end( p->v_start, p->v_end, &start, &end, l, threading );

  // compute initial residual
  if ( p->initial_guess_zero == 0 ) {
    apply_operator_PRECISION( p->w, p->x, p, l, threading ); // compute w = D*x
    vector_PRECISION_minus( p->r, p->b, p->w, start, end, l ); // compute r = b - w
  } else {
    vector_PRECISION_copy( p->r, p->b, start, end, l ); // compute r = b - w
  }

  beta = global_norm_PRECISION( p->r, p->v_start, p->v_end, l, threading ); // gamma_0 = norm(r)

  START_MASTER(threading);
  p->gcrodr_PRECISION.norm_r0 = beta;
  // setting the following line for the upcoming call to fgmresx_PRECISION(...)
  p->gamma[0] = beta;
  END_MASTER(threading);
  SYNC_MASTER_TO_ALL(threading)

  if (!p->initial_guess_zero) {
    beta = global_norm_PRECISION( p->b, p->v_start, p->v_end, l, threading );
  } else {
    START_MASTER(threading)
    p->gcrodr_PRECISION.b_norm = beta;
    END_MASTER(threading);
    SYNC_MASTER_TO_ALL(threading);
  }

  START_MASTER(threading)
  p->gcrodr_PRECISION.b_norm = beta;
  p->gcrodr_PRECISION.finish = 0;
  END_MASTER(threading);

  SYNC_MASTER_TO_ALL(threading);

  if ( p->gcrodr_PRECISION.CU_usable==1 ) {
    vector_PRECISION *Uk;

    if ( p->initial_guess_zero == 1 )
      vector_PRECISION_define( p->x, 0, start, end, l );

    SYNC_MASTER_TO_ALL(threading);

    // IMPORTANT : this if statement is not being accessed at the moment ...
    if ( p->gcrodr_PRECISION.update_CU == 1 ) {

      // Yk = copy(Uk)
      START_MASTER(threading);
      complex_PRECISION **tmp_ptr = p->gcrodr_PRECISION.U;
      p->gcrodr_PRECISION.U = p->gcrodr_PRECISION.Yk;
      p->gcrodr_PRECISION.Yk = tmp_ptr;
      for ( i=0; i<k; i++ ) {
        p->gcrodr_PRECISION.hatZ[i] = p->gcrodr_PRECISION.U[i];
      }
      END_MASTER(threading);
      SYNC_MASTER_TO_ALL(threading);

      // ------ QR = A*Yk

      // QR = A*Yk
      for ( i=0; i<k; i++ ) {
        apply_operator_PRECISION( p->gcrodr_PRECISION.C[i], p->gcrodr_PRECISION.Yk[i], p, l, threading );
      }

      //int i_length = p->v_end - p->v_start;
      int i_length = end-start;
      pqr_PRECISION( i_length, k, p->gcrodr_PRECISION.C, p->gcrodr_PRECISION.R, p, l, threading );

      SYNC_MASTER_TO_ALL(threading);

      START_MASTER(threading);
      inv_tri_PRECISION( &(p->gcrodr_PRECISION.eigslvr) );
      END_MASTER(threading);
      SYNC_MASTER_TO_ALL(threading);
      complex_PRECISION **Rinv = p->gcrodr_PRECISION.Rinv;

      // by this point : Ck = Q

      // Uk = Yk * R^{-1}
      Uk = p->gcrodr_PRECISION.U;
      vector_PRECISION *Yk = p->gcrodr_PRECISION.Yk;
      for ( i=0; i<k; i++ ) {
        // set all vectors in Yk to zero, to accumulate
        vector_PRECISION_define( Uk[i], 0, start, end, l );
        // and then, multi saxpy to obtain Yk
        // (the <i+1> in the 5th parameter is due to the triangular nature of Rinv)
        vector_PRECISION_multi_saxpy( Uk[i], Yk, Rinv[i], 1, i+1, start, end, l );
      }

      SYNC_MASTER_TO_ALL(threading);
      SYNC_CORES(threading)

      START_MASTER(threading)
      p->gcrodr_PRECISION.update_CU = 0;
      END_MASTER(threading)
      SYNC_MASTER_TO_ALL(threading)

#if defined(SINGLE_ALLREDUCE_ARNOLDI) && defined(PIPELINED_ARNOLDI)
      p->gcrodr_PRECISION.recompute_DPCk_poly = 1;
      p->gcrodr_PRECISION.recompute_DPCk_plain = 1;
#endif

      //printf0("UPDATED C AND U (INITIAL, USABLE ALREADY) ***\n");
    }

    // x  +=  Uk * Ck^{H} * r
    // r  -=  Ck * Ck^{H} * r

    complex_PRECISION *bf = p->gcrodr_PRECISION.Bbuff[0];
    complex_PRECISION *buffer = p->gcrodr_PRECISION.Bbuff[0] + k;
    vector_PRECISION *Ck = p->gcrodr_PRECISION.C;

    Uk = p->gcrodr_PRECISION.U;

    complex_PRECISION tmpx[k];
    process_multi_inner_product_PRECISION( k, tmpx, Ck, p->r, p->v_start, p->v_end, l, threading );
    START_MASTER(threading)
    // buffer is of length m, and k<m
    for ( i=0; i<k; i++ )
      buffer[i] = tmpx[i];
    if ( g.num_processes > 1 ) {
      PROF_PRECISION_START( _ALLR );
      MPI_Allreduce( buffer, bf, k, MPI_COMPLEX_PRECISION, MPI_SUM, (l->depth==0)?g.comm_cart:l->gs_PRECISION.level_comm );
      PROF_PRECISION_STOP( _ALLR, 1 );
    } else {
      for( i=0; i<k; i++ )
        bf[i] = buffer[i];
    }
    END_MASTER(threading)
    SYNC_MASTER_TO_ALL(threading);

    for( i=0; i<k; i++ )
      vector_PRECISION_saxpy( p->x, p->x, Uk[i], bf[i], start, end, l );

    apply_operator_PRECISION( p->w, p->x, p, l, threading ); // compute w = D*x
    vector_PRECISION_minus( p->r, p->b, p->w, start, end, l ); // compute r = b - w

    SYNC_MASTER_TO_ALL(threading);
  } else if ( p->gcrodr_PRECISION.CU_usable==0 ) {
  
    //printf0( "IN!\n" );
  
    // call one cycle of FGMRES

    if ( p->initial_guess_zero == 1 )
      vector_PRECISION_define( p->x, 0, start, end, l );

    int buff1x = p->restart_length;
    START_MASTER(threading)
    l->dup_H = 1;
    p->restart_length = k+1;
    END_MASTER(threading)
    SYNC_MASTER_TO_ALL(threading)
    m = fgmresx_PRECISION(p, l, threading);
    START_MASTER(threading)
    l->dup_H = 0;
    p->restart_length = buff1x;
    END_MASTER(threading)
    SYNC_MASTER_TO_ALL(threading);

    //printf0("OUT OF INITIAL GMRES, m = %d ***\n", m);

    //if ( m>20 && m<k ) {
    //if ( m>5 && m<k ) {
    if ( m>k ) {

      //double t0=0, t1=0;
      //START_MASTER(threading)
      //t0 = MPI_Wtime();
      //END_MASTER(threading)

      //START_MASTER(threading)
      //printf0("Quite a lot of iterations. Let's try and construct a deflation/recycling subspace\n");
      //END_MASTER(threading)
      //SYNC_MASTER_TO_ALL(threading)

      {

        START_MASTER(threading)
        p->initial_guess_zero = 0;
        END_MASTER(threading)
        SYNC_MASTER_TO_ALL(threading)

        START_MASTER(threading)
        vector_PRECISION_define_random( p->x, p->v_start, p->v_end, l );
        END_MASTER(threading)
        SYNC_MASTER_TO_ALL(threading)

        // compute initial residual
        apply_operator_PRECISION( p->w, p->x, p, l, threading ); // compute w = D*x
        vector_PRECISION_minus( p->r, p->b, p->w, start, end, l ); // compute r = b - w

        beta = global_norm_PRECISION( p->r, p->v_start, p->v_end, l, threading ); // gamma_0 = norm(r)

        START_MASTER(threading);
        p->gcrodr_PRECISION.norm_r0 = beta;
        // setting the following line for the upcoming call to fgmresx_PRECISION(...)
        p->gamma[0] = beta;
        END_MASTER(threading);
        SYNC_MASTER_TO_ALL(threading)

        beta = global_norm_PRECISION( p->b, p->v_start, p->v_end, l, threading );

        START_MASTER(threading)
        p->gcrodr_PRECISION.b_norm = beta;
        p->gcrodr_PRECISION.finish = 0;
        END_MASTER(threading);

        SYNC_MASTER_TO_ALL(threading);

      }

      double buff1x = p->tol;
      double buff2x = g.coarse_tol;
      int buff3x = p->restart_length;
      START_MASTER(threading)
      //if ( g.on_solve==1 ) {
        p->tol = 1.0e-20;
        g.coarse_tol = 1.0e-20;
        p->restart_length = k+1;
      //}
      l->dup_H = 1;
      END_MASTER(threading)
      SYNC_MASTER_TO_ALL(threading)

      m = fgmresx_PRECISION(p, l, threading);
      fgmresx_iter += m;

      START_MASTER(threading)
      p->tol = buff1x;
      g.coarse_tol = buff2x;
      p->restart_length = buff3x;
      l->dup_H = 0;
      END_MASTER(threading)
      SYNC_MASTER_TO_ALL(threading);

      //START_MASTER(threading)
      //t1 = MPI_Wtime();
      ////printf0("Arnoldi time : %.10f seconds\n", t1-t0);
      //END_MASTER(threading)
      //SYNC_MASTER_TO_ALL(threading)

    }
    else {
      fgmresx_iter += m;
    }

    // update the solution p->x (this, from the inside, applies back-substitution on p->y)
    compute_solution_PRECISION( p->x, (p->preconditioner&&p->kind==_RIGHT)?p->Z:p->V,
                                p->y, p->gamma, p->H, m-1, 1, p, l, threading );

//#ifdef BLOCK_JACOBI
#if 0

    // computing the actual residual in case of Block Jacobi
    {
      p->eval_operator( p->w, p->x, p->op, l, threading ); // compute w = D*x
      vector_PRECISION_minus( p->gcrodr_PRECISION.r_aux, p->block_jacobi_PRECISION.b_backup, p->w, start, end, l ); // compute r = b - w

      PRECISION norm_r0xx = global_norm_PRECISION( p->block_jacobi_PRECISION.b_backup, start, end, l, threading );
      PRECISION betaxx = global_norm_PRECISION( p->gcrodr_PRECISION.r_aux, start, end, l, threading );

      //printf0("OUT OF INITIAL GMRES, rel res = %f, finish = %d, tol = %f, marg tol = %f ***\n", betaxx/norm_r0xx, p->gcrodr_PRECISION.finish, p->tol, p->tol+0.2*p->tol);

      START_MASTER(threading)
      // hardcoding a 20% marging here
      if ( (betaxx/norm_r0xx) > (p->tol + 0.5*p->tol) ) {
        p->gcrodr_PRECISION.finish = 0;
      } else {
        p->gcrodr_PRECISION.finish = 1;
      }
      END_MASTER(threading)
      SYNC_MASTER_TO_ALL(threading)

      //printf0("OUT OF INITIAL GMRES, finish = %d ***\n", p->gcrodr_PRECISION.finish);
    }

    if ( l->level==0 && p->block_jacobi_PRECISION.BJ_usable==1 ) {
      block_jacobi_apply_PRECISION( l->p_PRECISION.block_jacobi_PRECISION.xtmp, p->w, p, l, threading );
      vector_PRECISION_minus( p->r, p->b, l->p_PRECISION.block_jacobi_PRECISION.xtmp, start, end, l ); // compute r = b - w
    } else {
      apply_operator_PRECISION( p->w, p->x, p, l, threading ); // compute w = D*x
      vector_PRECISION_minus( p->r, p->b, p->w, start, end, l ); // compute r = b - w
    }

#else

    apply_operator_PRECISION( p->w, p->x, p, l, threading ); // compute w = D*x
    vector_PRECISION_minus( p->r, p->b, p->w, start, end, l ); // compute r = b - w

#endif

    SYNC_MASTER_TO_ALL(threading);

    // if m<k, there's not enough information to build the recycling subspace
    if ( m<k ) {
      START_MASTER(threading)
      p->initial_guess_zero = buff_init_guess;
      END_MASTER(threading)
      SYNC_MASTER_TO_ALL(threading)
      return m;
    }

    //double t0=0, t1=0;
    //START_MASTER(threading)
    //t0 = MPI_Wtime();
    //END_MASTER(threading)
    //SYNC_MASTER_TO_ALL(threading)

    if ( p->preconditioner==NULL ) {
      // build the matrices A and B used for generalized-eigensolving
      gev_buildAB_PRECISION( p->gcrodr_PRECISION.gev_A, p->gcrodr_PRECISION.gev_B, p->gcrodr_PRECISION.eigslvr.Hc,
                             p->V, p->V, m, p, l, threading );

      SYNC_MASTER_TO_ALL(threading);
      SYNC_CORES(threading)

      // build C and U
      build_CU_PRECISION( p->gcrodr_PRECISION.eigslvr.Hc, p->V, p->V, p, l, threading, m );
    } else {
      // build the matrices A and B used for generalized-eigensolving
      gev_buildAB_PRECISION( p->gcrodr_PRECISION.gev_A, p->gcrodr_PRECISION.gev_B, p->gcrodr_PRECISION.eigslvr.Hc,
                             p->V, p->Z, m, p, l, threading );

      SYNC_MASTER_TO_ALL(threading);
      SYNC_CORES(threading)

      // build C and U
      build_CU_PRECISION( p->gcrodr_PRECISION.eigslvr.Hc, p->V, p->Z, p, l, threading, m );

      // IMPORTANT : this is an unconventional use of GCRO-DR, tailored to our case
      START_MASTER(threading)
      p->gcrodr_PRECISION.update_CU = 0;
      //printf0("SETTING FLAG TO NOT UPDATE C AND U ***\n");
      END_MASTER(threading)
      SYNC_MASTER_TO_ALL(threading)

      //printf0("COMPLETED INITIAL CONSTRUCTION OF C AND U ***\n");
    }

    //START_MASTER(threading)
    //t1 = MPI_Wtime();
    ////printf0("GEVP time : %.10f\n", t1-t0);
    //END_MASTER(threading)
    //SYNC_MASTER_TO_ALL(threading)

    // FIXME : issue when disabling this ...
    START_MASTER(threading)
    p->gcrodr_PRECISION.CU_usable=1;
    END_MASTER(threading)
    SYNC_MASTER_TO_ALL(threading);

#if defined(SINGLE_ALLREDUCE_ARNOLDI) && defined(PIPELINED_ARNOLDI)
      p->gcrodr_PRECISION.recompute_DPCk_poly = 1;
      p->gcrodr_PRECISION.recompute_DPCk_plain = 1;
#endif

    // check if this first call to fgmresx_PRECISION was enough
    if ( p->gcrodr_PRECISION.finish==1 ) {
      START_MASTER(threading)
      p->initial_guess_zero = buff_init_guess;
      END_MASTER(threading)
      SYNC_MASTER_TO_ALL(threading)
      return m;
    }

    SYNC_MASTER_TO_ALL(threading);
  } else{ error0("Invalid value for p->gcrodr_PRECISION.CU_usable \n"); }

//#ifdef BLOCK_JACOBI
#if 0
  PRECISION norm_r0xx = global_norm_PRECISION( p->block_jacobi_PRECISION.b_backup, start, end, l, threading );
#endif

  if ( g.gcrodr_calling_from_setup==1 ) {
    START_MASTER(threading)
    p->initial_guess_zero = buff_init_guess;
    END_MASTER(threading)
    SYNC_MASTER_TO_ALL(threading)
    return m;
  }

  for ( ol=0; ol < p->num_restart; ol++ )  {

    beta = global_norm_PRECISION( p->r, p->v_start, p->v_end, l, threading ); // gamma_0 = norm(r)

    START_MASTER(threading)
    // setting the following line for the upcoming call to fgmresx_PRECISION(...)
    p->gamma[0] = beta;
    l->dup_H = 1;
    p->gcrodr_PRECISION.orth_against_Ck = 1;
    END_MASTER(threading);
    SYNC_MASTER_TO_ALL(threading);

    // NOTE the value of m from here onwards
    m = fgmresx_PRECISION(p, l, threading);
    fgmresx_iter += m;

    //printf0("OUT OF INNER GMRES, m = %d ***\n", m);

    complex_PRECISION *bf = p->gcrodr_PRECISION.Bbuff[0];

    START_MASTER(threading)
    l->dup_H = 0;
    p->gcrodr_PRECISION.orth_against_Ck = 0;

    // After calling fgmresx_PRECISION(...), G is of
    // size ( k+(m+1) ) x ( k+m )
    // set B within G
    for ( j=0; j<m; j++ ) {
      memcpy( p->gcrodr_PRECISION.G[k+j] , p->gcrodr_PRECISION.ort_B[j], sizeof(complex_PRECISION)*k );
      memcpy( p->gcrodr_PRECISION.Gc[k+j], p->gcrodr_PRECISION.ort_B[j], sizeof(complex_PRECISION)*k );
    }
    // set H within G
    for ( j=0; j<m; j++ ) {
      memcpy( p->gcrodr_PRECISION.G[k+j]+k , p->gcrodr_PRECISION.eigslvr.Hc[j], sizeof(complex_PRECISION)*(j+2) );
      memcpy( p->gcrodr_PRECISION.Gc[k+j]+k, p->gcrodr_PRECISION.eigslvr.Hc[j], sizeof(complex_PRECISION)*(j+2) );
    }

    // and, the last ingredient for the least squares problem is : <bf> as the rhs, <G> as the matrix
    memset(bf, 0.0, sizeof(complex_PRECISION)*(k+m+1));
    bf[k] = beta;

    for ( i=0; i<k; i++ ) { p->gcrodr_PRECISION.lsp_diag_G[i] = p->gcrodr_PRECISION.G[i][i]; }

    g.gcrodr_LSP_time -= MPI_Wtime();
    //gels_PRECISION( LAPACK_COL_MAJOR, 'N', k+m+1, k+m, 1, p->gcrodr_PRECISION.G[0], k+p->restart_length+1, bf, k+p->restart_length+1);
    gels_via_givens_PRECISION( k+m+1, k+m, p->gcrodr_PRECISION.G[0], k+p->restart_length+1, bf, k+p->restart_length+1, p->gcrodr_PRECISION.lsp_x, k, m, p );
    g.gcrodr_LSP_time += MPI_Wtime();

    // restoring G from Gc
    {
      int g_ln = k + p->restart_length;
      memset( p->gcrodr_PRECISION.G[0], 0.0, sizeof(complex_PRECISION)*( g_ln*(g_ln+1) ) );
    }
    for ( i=0; i<k; i++ ) { p->gcrodr_PRECISION.G[i][i] = p->gcrodr_PRECISION.Gc[i][i]; }
    END_MASTER(threading)

    SYNC_MASTER_TO_ALL(threading)

    // update the solution
    for( i=0; i<k; i++ ) {
      vector_PRECISION_saxpy( p->x, p->x, p->gcrodr_PRECISION.hatZ[i], bf[i]*p->gcrodr_PRECISION.Gc[i][i], start, end, l );
    }
    for( i=k; i<(k+m); i++ )
      vector_PRECISION_saxpy( p->x, p->x, p->gcrodr_PRECISION.hatZ[i], bf[i], start, end, l );

    // updating p->r

//#ifdef BLOCK_JACOBI
#if 0

    // computing the actual residual in case of Block Jacobi
    {
      p->eval_operator( p->w, p->x, p->op, l, threading ); // compute w = D*x
      vector_PRECISION_minus( p->gcrodr_PRECISION.r_aux, p->block_jacobi_PRECISION.b_backup, p->w, start, end, l ); // compute r = b - w

      PRECISION betaxx = global_norm_PRECISION( p->gcrodr_PRECISION.r_aux, start, end, l, threading );

      //printf0("OUT OF INNER GMRES, rel res = %f, finish = %d, tol = %f, marg tol = %f ***\n", betaxx/norm_r0xx, p->gcrodr_PRECISION.finish, p->tol, p->tol+0.2*p->tol);

      // hardcoding a 20% marging here
      START_MASTER(threading)
      if ( (betaxx/norm_r0xx) > (p->tol + 0.2*p->tol) ) {
        p->gcrodr_PRECISION.finish = 0;
      } else { 
        p->gcrodr_PRECISION.finish = 1;
      }
      END_MASTER(threading)
      SYNC_MASTER_TO_ALL(threading)

      //printf0("OUT OF INNER GMRES, finish = %d ***\n", p->gcrodr_PRECISION.finish);
    }

    if ( l->level==0 && p->block_jacobi_PRECISION.BJ_usable==1 ) {
      block_jacobi_apply_PRECISION( l->p_PRECISION.block_jacobi_PRECISION.xtmp, p->w, p, l, threading );
      vector_PRECISION_minus( p->r, p->b, l->p_PRECISION.block_jacobi_PRECISION.xtmp, start, end, l ); // compute r = b - w
    } else {
      apply_operator_PRECISION( p->w, p->x, p, l, threading ); // compute w = D*x
      vector_PRECISION_minus( p->r, p->b, p->w, start, end, l ); // compute r = b - w
    }

#else

    apply_operator_PRECISION( p->w, p->x, p, l, threading ); // compute w = D*x
    vector_PRECISION_minus( p->r, p->b, p->w, start, end, l ); // compute r = b - w

#endif

    int upd_itrs;
    if ( g.on_solve==1 ) { upd_itrs = g.gcrodr_upd_itrs_solve; }
    else { upd_itrs = g.gcrodr_upd_itrs_setup; }

    if ( (m>20) && (p->gcrodr_PRECISION.upd_ctr < upd_itrs) ) {
      // build the matrices A and B used for generalized-eigensolving

      START_MASTER(threading)
      g.gcrodr_buildAB_time -= MPI_Wtime();
      END_MASTER(threading)
      gev_buildAB_PRECISION( p->gcrodr_PRECISION.gev_A, p->gcrodr_PRECISION.gev_B, p->gcrodr_PRECISION.Gc,
                             p->gcrodr_PRECISION.hatW, p->gcrodr_PRECISION.hatZ, k+m, p, l, threading );
      START_MASTER(threading)
      g.gcrodr_buildAB_time += MPI_Wtime();
      END_MASTER(threading)

      SYNC_MASTER_TO_ALL(threading);

      // build C and U
      START_MASTER(threading)
      g.gcrodr_buildCU_time -= MPI_Wtime();
      END_MASTER(threading)
      build_CU_PRECISION( p->gcrodr_PRECISION.Gc, p->gcrodr_PRECISION.hatW, p->gcrodr_PRECISION.hatZ, p, l, threading, k+m );
      START_MASTER(threading)
      g.gcrodr_buildCU_time += MPI_Wtime();
      END_MASTER(threading)

      START_MASTER(threading)
      p->gcrodr_PRECISION.upd_ctr++;
      END_MASTER(threading)
      SYNC_MASTER_TO_ALL(threading)

      //printf0("UPDATED C AND U (INNER, USABLE ALREADY) ***\n");
    }

    SYNC_MASTER_TO_ALL(threading);

    // check if tolerance has been reached
    if ( p->gcrodr_PRECISION.finish==1 ) {
      START_MASTER(threading)
      p->initial_guess_zero = buff_init_guess;
      END_MASTER(threading)
      SYNC_MASTER_TO_ALL(threading)
      return fgmresx_iter;
    }

    SYNC_MASTER_TO_ALL(threading);
  }

  SYNC_MASTER_TO_ALL(threading);

  START_MASTER(threading)
  p->initial_guess_zero = buff_init_guess;
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)
  return fgmresx_iter;
}



// ---------- AUXILIARY FUNCTIONS



// build A and B for the generalized eigenvalue problem within FL-GCRO-DR
void gev_buildAB_PRECISION( complex_PRECISION **A, complex_PRECISION **B, complex_PRECISION **G,
                            vector_PRECISION *W, vector_PRECISION *Z, int mk, gmres_PRECISION_struct *p,
                            level_struct *l, struct Thread *threading ){

  int start, end, i, j, k;
  int g_ln = p->restart_length + p->gcrodr_PRECISION.k;

  complex_PRECISION **Bbuff = p->gcrodr_PRECISION.Bbuff;
  complex_PRECISION *Bbuffx = Bbuff[0] + g_ln*(g_ln+1);
  // tmpy is stored in column-major
  complex_PRECISION tmpy[(mk+1)*(mk)];

  compute_core_start_end(p->v_start, p->v_end, &start, &end, l, threading);

  // -------- building B

  for ( j=0; j<mk; j++ ) {
    process_multi_inner_product_PRECISION( mk+1, tmpy+j*(mk+1), W, Z[j], p->v_start, p->v_end, l, threading );
  }

  START_MASTER(threading)
  if ( g.num_processes > 1 ) {
    PROF_PRECISION_START( _ALLR );
    MPI_Allreduce( tmpy, Bbuffx, (mk+1)*mk, MPI_COMPLEX_PRECISION, MPI_SUM, (l->depth==0)?g.comm_cart:l->gs_PRECISION.level_comm );
    PROF_PRECISION_STOP( _ALLR, 1 );
  } else {
    for( i=0; i<(mk+1)*mk; i++ )
      Bbuffx[i] = tmpy[i];
  }

  // from Bbuffx to Bbuff
  for ( j=0; j<mk; j++ ) {
    for ( i=0; i<(mk+1); i++ ) {
      Bbuff[j][i] = (Bbuffx+j*(mk+1))[i];
    }
  }

  for ( j=0; j<mk; j++ ) {
    if (p->gcrodr_PRECISION.CU_usable == 1) {
      if (j<p->gcrodr_PRECISION.k) for (i=0; i<(mk+1); i++) Bbuff[j][i] *= G[j][j];
    }
  }
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)

  // FIXME : improve the following matrix-matrix multiplication by using more threads than <master>
  START_MASTER(threading)
  for ( j=0; j<mk; j++ ) {
    for ( i=0; i<mk; i++ ) {
      B[j][i] = 0.0;
      for ( k=0; k<mk+1; k++ ) {
        B[j][i] += conj_PRECISION(G[i][k])*(Bbuff[j][k]);
      }
    }
  }
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading);

  // -------- building A

  // FIXME : improve the following matrix-matrix multiplication by using more threads than <master>
  START_MASTER(threading)
  for ( j=0; j<mk; j++ ) {
    for ( i=0; i<mk; i++ ) {
      A[j][i] = 0.0;
      for ( k=0; k<mk+1; k++ ) {
        A[j][i] += conj_PRECISION(G[i][k])*(G[j][k]);
      }
    }
  }
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading);
}


int fgmresx_PRECISION( gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading ) {

/*********************************************************************************
* This is a trimmed version of FGMRES, as a helper to flgcrodr_PRECISION(...)
* This function assumes:
*	-- p->r already contains the computed residual
*********************************************************************************/  

  SYNC_MASTER_TO_ALL(threading);
  SYNC_CORES(threading)

  // start and end indices for vector functions depending on thread
  int start, end, j=-1, finish=0, iter=0, il;

  //int was_there_stagnation = 0;

  PRECISION norm_r0=1, gamma_jp1=1;

  START_LOCKED_MASTER(threading)
  if ( l->level==0 && g.num_levels > 1 && g.interpolation ) p->tol = g.coarse_tol;
  END_LOCKED_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)

  // compute start and end indices for core
  // this puts zero for all other hyperthreads, so we can call functions below with all hyperthreads
  compute_core_start_end(p->v_start, p->v_end, &start, &end, l, threading);
  
  norm_r0 = p->gcrodr_PRECISION.norm_r0;

  SYNC_MASTER_TO_ALL(threading)
  SYNC_CORES(threading)

  vector_PRECISION_real_scale( p->V[0], p->r, 1/p->gamma[0], start, end, l ); // v_0 = r / gamma_0

#if defined(SINGLE_ALLREDUCE_ARNOLDI) && defined(PIPELINED_ARNOLDI)
  if ( l->level == 0 && l->depth > 0 ) {
    arnoldi_step_PRECISION( p->V, p->Z, p->w, p->H, p->y, 0, p->preconditioner, p, l, threading );
  }
#endif

  PRECISION prev_res=-1.0, curr_res=-1.0;

//#ifdef BLOCK_JACOBI
//  int rewinder = p->restart_length/10;
//  if (rewinder<6) rewinder = 6;
//  if (rewinder>10) rewinder = 10;
//  // two more iterations might be enough to cover the difference between 'real' and 'fake' residuals
//  for( il=0; (il < rewinder) || (il<p->restart_length && finish==0); il++) {
//#else
  for( il=0; il<p->restart_length && finish==0; il++) {
//#endif

    j = il; iter++;

    //START_MASTER(threading)
    //printf0( "IL : j = %d\n",j );
    //END_MASTER(threading)

    // one step of Arnoldi
#if defined(SINGLE_ALLREDUCE_ARNOLDI) && defined(PIPELINED_ARNOLDI)
    if ( l->level == 0 && l->depth > 0 ) {
      if ( !arnoldi_step_PRECISION( p->V, p->Z, p->w, p->H, p->y, j+1, p->preconditioner, p, l, threading ) ) {
        printf0("| -------------- iteration %d, restart due to H(%d,%d) < 0 |\n", iter, j+2, j+1 );
        break;
      }
    } else {
      if ( !arnoldi_step_PRECISION( p->V, p->Z, p->w, p->H, p->y, j, p->preconditioner, p, l, threading ) ) {
        printf0("| -------------- iteration %d, restart due to H(%d,%d) < 0 |\n", iter, j+1, j );
        break;
      }
    }
#else
    if ( !arnoldi_step_PRECISION( p->V, p->Z, p->w, p->H, p->y, j, p->preconditioner, p->shift, p, l, threading ) ) {
      printf0("| -------------- iteration %d, restart due to H(%d,%d) < 0 |\n", iter, j+1, j );
      break;
    }
#endif

    SYNC_MASTER_TO_ALL(threading)

    //printf0("WITHIN INNER GMRES, H[x][x] = %f ***\n", cabs( p->H[j][j+1] ));

    if ( cabs( p->H[j][j+1] ) > p->tol/10 ) {
      qr_update_PRECISION( p->H, p->s, p->c, p->gamma, j, l, threading );
      SYNC_MASTER_TO_ALL(threading)
      gamma_jp1 = cabs( p->gamma[j+1] );

      START_MASTER(threading)
      //printf0("g (proc=%d,j=%d) rel residual (gcro-dr) = %f\n\n", g.my_rank, j, gamma_jp1/norm_r0);
      END_MASTER(threading)

      //printf0("WITHIN INNER GMRES, inner rel res = %.8f ***\n", cabs( p->gamma[j+1] )/norm_r0);

      // check if the four most significant digits of the residual haven't changed
      if ( j%5==0 ) {
        prev_res = curr_res;
        curr_res = cabs( p->gamma[j+1] )/norm_r0;

        if ( j>0 ) {
          // extract the three most significant digits of prev_res and curr_res
          int nr1_i, nr2_i;
          {
            PRECISION nr1_f = prev_res, nr2_f = curr_res;
            int fctr = 1;
            while ( nr1_f<1000 ) {
              nr1_f *= 10;
              fctr *= 10;
            }
            nr2_f *= fctr;
            nr1_i = (int)nr1_f;
            nr2_i = (int)nr2_f;
          }

          // if the residual hasn't changed, exit
          if ( nr1_i == nr2_i ) {

            //printf0( "WARNING : stagnation to three significant digits in the residual, the recycling subspace needs to be rebuilt\n" );

            finish = 1;

            START_MASTER(threading)
            p->was_there_stagnation = 1;
            END_MASTER(threading)
            SYNC_MASTER_TO_ALL(threading)
          }
        }
      }

      if( gamma_jp1/norm_r0 < p->tol || gamma_jp1/norm_r0 > 1E+5 ) { // if satisfied ... stop

        START_MASTER(threading)
        p->gcrodr_PRECISION.finish = 1;
        END_MASTER(threading)
        SYNC_MASTER_TO_ALL(threading)
        
        finish = 1;
        
        START_MASTER(threading)
        if ( gamma_jp1/norm_r0 > 1E+5 ) printf0("Divergence of fgmresx_PRECISION, iter = %d, level=%d\n", iter, l->level );
        END_MASTER(threading)

      }
    } else {
      //START_MASTER(threading)
      //printf0("from gcrodr : depth: %d, iter: %d, p->H(%d,%d) = %+lf+%lfi\n", l->depth, iter, j+1, j, CSPLIT( p->H[j][j+1] ) );
      //END_MASTER(threading)
      finish = 1;
    }
  } // end of the (only and) single restart

  if ( l->level == 0 ) {
    START_LOCKED_MASTER(threading)
    g.coarse_iter_count += iter;
    END_LOCKED_MASTER(threading)
  }

  SYNC_MASTER_TO_ALL(threading)
  SYNC_CORES(threading)

  //if ( was_there_stagnation==1 ) printf0("WARNING : there was stagnation of the residual\n");

  return iter;
}


// order the eigenpairs returned by the generalized eigenvalue solver from LAPACK
void order_pairs_PRECISION( vector_PRECISION keys, complex_PRECISION *keys_cpy, int *output, int n ){
  int i, j, buff1;

  complex_PRECISION buff2;

  memcpy(keys_cpy, keys, sizeof(complex_PRECISION)*n);

  for ( i=0; i<n; i++ ) {
    output[i] = i;
  }

  for ( i=0; i<n; i++ ) {
    for ( j=i+1; j<n; j++ ) {
      if ( cabs(keys_cpy[i]) > cabs(keys_cpy[j]) ) {
        buff2 =  keys_cpy[i];
        keys_cpy[i] = keys_cpy[j];
        keys_cpy[j] = buff2;

        buff1 =  output[i];
        output[i] = output[j];
        output[j] = buff1;
      }
    }
  }
}


void build_CU_PRECISION( complex_PRECISION **G, vector_PRECISION *W, vector_PRECISION *Z,
                         gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading, int m ){

  // --------- eigensolving first

  // FIXME : improve the following eigensolve by using more threads than <master>
  START_MASTER(threading)
  // calling LAPACK's generalized eigenvalue solver through LAPACKE
  p->gcrodr_PRECISION.eigslvr.N = m;
  gen_eigslvr_PRECISION( &(p->gcrodr_PRECISION.eigslvr) );
  // the actual eigenvalues coming of LAPACK's gen-eigensolver are w/beta
  for (int i=0; i<m; i++) {
    p->gcrodr_PRECISION.eigslvr.w[i] /= p->gcrodr_PRECISION.eigslvr.beta[i];
  }
  // p->gcrodr_PRECISION.eigslvr.ordr_idxs contains the indices to access w and vr in ascending magnitude
  order_pairs_PRECISION( p->gcrodr_PRECISION.eigslvr.w, p->gcrodr_PRECISION.eigslvr.ordr_keyscpy,
                         p->gcrodr_PRECISION.eigslvr.ordr_idxs, m );
  END_MASTER(threading)

  SYNC_MASTER_TO_ALL(threading)

  // ---------------- then, computing C and U

  int i, j, kl, start, end, g_ln=p->restart_length + p->gcrodr_PRECISION.k, k=p->gcrodr_PRECISION.k;

  compute_core_start_end(p->v_start, p->v_end, &start, &end, l, threading);

  complex_PRECISION **QR = p->gcrodr_PRECISION.QR;
  complex_PRECISION **Q = p->gcrodr_PRECISION.Q;
  complex_PRECISION **R = p->gcrodr_PRECISION.R;
  complex_PRECISION **Rinv = p->gcrodr_PRECISION.eigslvr.qr_Rinv;

  vector_PRECISION *Yk = p->gcrodr_PRECISION.Yk;
  vector_PRECISION *Ck2 = p->gcrodr_PRECISION.Cc;
  vector_PRECISION *Uk = p->gcrodr_PRECISION.U;
  vector_PRECISION vr  = p->gcrodr_PRECISION.eigslvr.vr;

  int *idxs = p->gcrodr_PRECISION.eigslvr.ordr_idxs;

  // for each new eigensolution, we have a new mapping for Pk
  vector_PRECISION *Pk = p->gcrodr_PRECISION.Pk;
  START_MASTER(threading)
  for ( i=0; i<k; i++ ) {
    Pk[i] = vr + idxs[i] * g_ln;
  }
  END_MASTER(threading)

  SYNC_MASTER_TO_ALL(threading)

  // compute Yk
  for ( i=0; i<k; i++ ) {
    // set all vectors in Yk to zero, to accumulate
    vector_PRECISION_define( Yk[i], 0, start, end, l );
    // and then, multi saxpy to obtain Yk

    if (p->gcrodr_PRECISION.CU_usable == 1) {
      for (j=0; j<k; j++) vector_PRECISION_saxpy( Yk[i], Yk[i], Z[j], Pk[i][j] * G[j][j], start, end, l );
      vector_PRECISION_multi_saxpy( Yk[i], Z+k, Pk[i]+k, 1, m-k, start, end, l );
    } else {
      vector_PRECISION_multi_saxpy( Yk[i], Z, Pk[i], 1, m, start, end, l );
    }
  }

  // build the matrix for QR
  // FIXME : improve the following matrix-matrix multiplication by using more threads than <master>
  START_MASTER(threading)
  for ( j=0; j<k; j++ ) {
    // set all column j of QR to zero
    memset( QR[j], 0.0, sizeof(complex_PRECISION)*(m+1) );
    // and then do accumulations over that column
    for ( i=0; i<m; i++ ) {
      for ( kl=0; kl<(m+1); kl++ ) {
        QR[j][kl] += G[i][kl] * Pk[j][i];
      }
    }
  }

  // QR decomposition of G*Pk
  // FIXME : improve the following matrix-matrix multiplication by using more threads than <master>
  // QR
  p->gcrodr_PRECISION.eigslvr.qr_m = m+1;
  p->gcrodr_PRECISION.eigslvr.qr_n = k;
  qr_PRECISION( &(p->gcrodr_PRECISION.eigslvr) );
  p->gcrodr_PRECISION.eigslvr.qr_k = k;
  memset( R[0], 0.0, sizeof(complex_PRECISION)*k*k );
  // compute R^{-1}
  for ( j=0; j<k; j++ ) {
    for ( i=0; i<k; i++ ) {
      R[j][i] = QR[j][i];
    }
  }
  inv_tri_PRECISION( &(p->gcrodr_PRECISION.eigslvr) );
  // compute Q
  q_from_qr_PRECISION( &(p->gcrodr_PRECISION.eigslvr) );
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)

  // compute Ck
  for ( i=0; i<k; i++ ) {
    // set all vectors in Yk to zero, to accumulate
    vector_PRECISION_define( Ck2[i], 0, start, end, l );
    // and then, multi saxpy to obtain Yk
    vector_PRECISION_multi_saxpy( Ck2[i], W, Q[i], 1, m+1, start, end, l );
  }

  SYNC_MASTER_TO_ALL(threading)

  // and then swap pointers
  START_MASTER(threading)
  complex_PRECISION **tmp_ptr;
  tmp_ptr = p->gcrodr_PRECISION.C;
  p->gcrodr_PRECISION.C = p->gcrodr_PRECISION.Cc;
  p->gcrodr_PRECISION.Cc = tmp_ptr;
  for ( i=0; i<k; i++ ) {
    p->gcrodr_PRECISION.hatW[i] = p->gcrodr_PRECISION.C[i];
  }
  END_MASTER(threading)

  SYNC_MASTER_TO_ALL(threading)

  // compute Uk
  for ( i=0; i<k; i++ ) {
    // set all vectors in Yk to zero, to accumulate
    vector_PRECISION_define( Uk[i], 0, start, end, l );
    // and then, multi saxpy to obtain Yk
    // (the <i+1> in the 5th parameter is due to the triangular nature of Rinv)
    vector_PRECISION_multi_saxpy( Uk[i], Yk, Rinv[i], 1, i+1, start, end, l );
  }

  // compute \tilde{ Uk }
  complex_PRECISION tmpy[k+1];
  complex_PRECISION **Bbuff = p->gcrodr_PRECISION.Bbuff;

  for ( i=0; i<k; i++ ) {
    process_multi_inner_product_PRECISION( 1, tmpy+i, &(Uk[i]), Uk[i], p->v_start, p->v_end, l, threading );
  }

  START_MASTER(threading)
  if ( g.num_processes > 1 ) {
    PROF_PRECISION_START( _ALLR );
    MPI_Allreduce( tmpy, Bbuff[0], k, MPI_COMPLEX_PRECISION, MPI_SUM, (l->depth==0)?g.comm_cart:l->gs_PRECISION.level_comm );
    PROF_PRECISION_STOP( _ALLR, 1 );
  } else {
    for( i=0; i<k; i++ )
      Bbuff[0][i] = tmpy[i];
  }

  for ( i=0; i<k; i++ ) {
    complex_PRECISION diag_term = sqrt( creal(Bbuff[0][i]) );
    START_MASTER(threading)
    diag_term = 1.0 / diag_term;
    p->gcrodr_PRECISION.G[i][i]  = diag_term;
    p->gcrodr_PRECISION.Gc[i][i] = diag_term;
    END_MASTER(threading)
  }
  END_MASTER(threading)

  SYNC_MASTER_TO_ALL(threading)

#if defined(SINGLE_ALLREDUCE_ARNOLDI) && defined(PIPELINED_ARNOLDI)
  p->gcrodr_PRECISION.recompute_DPCk_poly = 1;
  p->gcrodr_PRECISION.recompute_DPCk_plain = 1;
  SYNC_MASTER_TO_ALL(threading)
#endif
}


int arnoldix_step_PRECISION( vector_PRECISION *V, vector_PRECISION *Z, vector_PRECISION w,
                             complex_PRECISION **H, complex_PRECISION* buffer, int j, void (*prec)(),
                             gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading ) {

  int return_val;
  return_val = arnoldi_step_PRECISION( V, Z, w, H, buffer, j, prec, p->shift, p, l, threading );
  return return_val;

}

void re_scale_Uk_PRECISION( gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading ){
  int i;
  int k = p->gcrodr_PRECISION.k;
  int start, end;
  compute_core_start_end( p->v_start, p->v_end, &start, &end, l, threading );

  vector_PRECISION *Uk = p->gcrodr_PRECISION.U;

  SYNC_MASTER_TO_ALL(threading)
  SYNC_CORES(threading)

  for ( i=0; i<k; i++ ) {
    complex_PRECISION diag_term = 1.0 / p->gcrodr_PRECISION.Gc[i][i];
    vector_PRECISION_scale( Uk[i], Uk[i], diag_term, start, end, l );
  }

  SYNC_MASTER_TO_ALL(threading)
  SYNC_CORES(threading)

}


#endif
