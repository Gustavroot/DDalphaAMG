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

void local_fgmres_PRECISION_struct_init( local_gmres_PRECISION_struct *p ) {

  p->Z = NULL;
  p->V = NULL;
  p->H = NULL;
  p->x = NULL;
  p->b = NULL;
  p->r = NULL;
  p->w = NULL;
  p->y = NULL;
  p->gamma = NULL;
  p->c = NULL;
  p->s = NULL;
  //p->preconditioner = NULL;
  p->eval_operator = NULL;

  // copy of Hesselnberg matrix
#if defined(POLYPREC)
  p->polyprec_PRECISION.eigslvr.Hc = NULL;
#endif

#ifdef POLYPREC
  p->polyprec_PRECISION.Hcc = NULL; 
  p->polyprec_PRECISION.L = NULL;
  p->polyprec_PRECISION.col_prods = NULL;
  p->polyprec_PRECISION.accum_prod = NULL;
  p->polyprec_PRECISION.product = NULL;
  p->polyprec_PRECISION.temp = NULL;
  p->polyprec_PRECISION.h_ritz = NULL;
  p->polyprec_PRECISION.lejas = NULL;
  p->polyprec_PRECISION.random_rhs = NULL;
  p->polyprec_PRECISION.xtmp = NULL;

  p->polyprec_PRECISION.eigslvr.vl = NULL;
  p->polyprec_PRECISION.eigslvr.vr = NULL;
  p->polyprec_PRECISION.dirctslvr.ipiv = NULL;
  p->polyprec_PRECISION.dirctslvr.x = NULL;
  p->polyprec_PRECISION.dirctslvr.b = NULL;
#endif
}

void local_fgmres_PRECISION_struct_alloc( int m, int n, long int vl, PRECISION tol, const int type, const int prec_kind,
                                          void (*precond)(), void (*eval_op)(), local_gmres_PRECISION_struct *p, level_struct *l ) {


  long int total=0; 
  int i, k=0;
  
  p->restart_length = m;
  p->num_restart = n;

  //p->preconditioner = precond;

  p->eval_operator = eval_op; 
  p->tol = tol;
  p->kind = prec_kind;

#ifdef HAVE_TM1p1
  //vl*=2;
#endif
  
  if(m > 0) {
  total += (m+1)*m; // Hessenberg matrix
  MALLOC( p->H, complex_PRECISION*, m );
  
  total += (5+m)*vl; // x, r, b, w, V
  MALLOC( p->V, complex_PRECISION*, m+1 );

  // no preconditioner
  k = 0;

  total += 4*(m+1); // y, gamma, c, s
  
  p->H[0] = NULL; // allocate connected memory
  MALLOC( p->H[0], complex_PRECISION, total );
  
  p->total_storage = total;
  total = 0;
  
  // ordering: H, y, gamma, c, s, w, V, Z, x, r, b
  // H
  for ( i=1; i<m; i++ )
    p->H[i] = p->H[0] + i*(m+1);
  total += m*(m+1);
  
  // y
  p->y = p->H[0] + total; total += m+1;
  // gamma
  p->gamma = p->H[0] + total; total += m+1;
  // c
  p->c = p->H[0] + total; total += m+1;
  // s
  p->s = p->H[0] + total; total += m+1;
  // w
  p->w = p->H[0] + total; total += vl;
  // V
  for ( i=0; i<m+1; i++ ) {
    p->V[i] = p->H[0] + total; total += vl;
  }
  // Z
  for ( i=0; i<k; i++ ) {
    p->Z[i] = p->H[0] + total; total += vl;
  }

  // x
  p->x = p->H[0] + total; total += vl;
  // r
  p->r = p->H[0] + total; total += vl;
  // b
  p->b = p->H[0] + total; total += vl;
  
  ASSERT( p->total_storage == total );
  }

  if ( type == _GLOBAL_FGMRES ) {    
    p->timing = 1;
    p->print = g.vt.evaluation?0:1;
    p->initial_guess_zero = 1;
    p->v_start = 0;
    p->v_end = l->inner_vector_size;
    p->op = &(g.op_PRECISION);
  } else if ( type == _K_CYCLE ) {
    // these settings also work for GMRES as a smoother
    p->timing = 0;
    p->print = 0;
    p->initial_guess_zero = 1;
    p->v_start = 0;
    p->v_end = l->inner_vector_size;
    p->op = &(l->s_PRECISION.op);
  } else if ( type == _COARSE_GMRES ) {
    p->timing = 0;
    p->print = 0;
    p->initial_guess_zero = 1;
    p->layout = -1;
    p->v_start = 0;
    p->v_end = l->inner_vector_size;
    if ( g.odd_even )
      p->op = &(l->oe_op_PRECISION);
    else  
      p->op = &(l->s_PRECISION.op);
  } else {
    ASSERT( type < 3 );
  }

#if defined(GCRODR) || defined(POLYPREC)
  if (l->level==0) {
#endif

  // copy of Hesselnberg matrix
#if defined(POLYPREC)
  MALLOC(p->polyprec_PRECISION.eigslvr.Hc, complex_PRECISION*, m);
  p->polyprec_PRECISION.eigslvr.Hc[0] = NULL; // allocate connected memory
  MALLOC( p->polyprec_PRECISION.eigslvr.Hc[0], complex_PRECISION, m*(m+1) );
  for ( i=1; i<m; i++ )
    p->polyprec_PRECISION.eigslvr.Hc[i] = p->polyprec_PRECISION.eigslvr.Hc[0] + i*(m+1);
#endif

#ifdef POLYPREC
  //p->polyprec_PRECISION.d_poly = g.polyprec_d;
  int d_poly=p->polyprec_PRECISION.d_poly;

  MALLOC( p->polyprec_PRECISION.col_prods, complex_PRECISION, d_poly);
  MALLOC( p->polyprec_PRECISION.h_ritz, complex_PRECISION, d_poly);
  MALLOC( p->polyprec_PRECISION.lejas, complex_PRECISION, d_poly);
  MALLOC( p->polyprec_PRECISION.random_rhs, complex_PRECISION, vl );
  MALLOC( p->polyprec_PRECISION.accum_prod, complex_PRECISION, vl );
  MALLOC( p->polyprec_PRECISION.product, complex_PRECISION, vl );
  MALLOC( p->polyprec_PRECISION.temp, complex_PRECISION, vl );

  MALLOC( p->polyprec_PRECISION.xtmp, complex_PRECISION, vl );

  MALLOC( p->polyprec_PRECISION.Hcc, complex_PRECISION, d_poly*d_poly );
  MALLOC( p->polyprec_PRECISION.L, complex_PRECISION*, d_poly+ 1);

  p->polyprec_PRECISION.L[0] = NULL;

  MALLOC( p->polyprec_PRECISION.L[0], complex_PRECISION, (d_poly+1)*d_poly );

  for (i=1; i<d_poly+1; i++)
  {
    p->polyprec_PRECISION.L[i] = p->polyprec_PRECISION.L[0] + i*d_poly;
  }

  MALLOC( p->polyprec_PRECISION.dirctslvr.ipiv, int, d_poly);
  MALLOC( p->polyprec_PRECISION.dirctslvr.x, complex_PRECISION, d_poly);
  MALLOC( p->polyprec_PRECISION.dirctslvr.b, complex_PRECISION, d_poly);

  p->polyprec_PRECISION.dirctslvr.N = d_poly;
  p->polyprec_PRECISION.dirctslvr.lda = d_poly; // m here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  p->polyprec_PRECISION.dirctslvr.ldb = d_poly;
  p->polyprec_PRECISION.dirctslvr.nrhs = 1;
  p->polyprec_PRECISION.dirctslvr.Hcc = p->polyprec_PRECISION.Hcc;
  p->polyprec_PRECISION.dirctslvr.dirctslvr_PRECISION = dirctslvr_PRECISION;

  MALLOC( p->polyprec_PRECISION.eigslvr.vl, complex_PRECISION, d_poly*d_poly );
  MALLOC( p->polyprec_PRECISION.eigslvr.vr, complex_PRECISION, d_poly*d_poly );

  p->polyprec_PRECISION.eigslvr.jobvl = 'N';
  p->polyprec_PRECISION.eigslvr.jobvr = 'N';

  p->polyprec_PRECISION.eigslvr.N = d_poly;
  p->polyprec_PRECISION.eigslvr.lda = p->restart_length + 1;
  p->polyprec_PRECISION.eigslvr.ldvl = d_poly;
  p->polyprec_PRECISION.eigslvr.ldvr = d_poly;
  p->polyprec_PRECISION.eigslvr.w = p->polyprec_PRECISION.h_ritz;
  p->polyprec_PRECISION.Hc = p->polyprec_PRECISION.eigslvr.Hc;
  p->polyprec_PRECISION.eigslvr.eigslvr_PRECISION = eigslvr_PRECISION;    

  p->polyprec_PRECISION.update_lejas = 1;
  //p->polyprec_PRECISION.preconditioner = NULL;
  //p->polyprec_PRECISION.preconditioner_bare = p->preconditioner;
  p->polyprec_PRECISION.syst_size = vl;

  p->polyprec_PRECISION.eigslvr.A = p->polyprec_PRECISION.Hc[0];
#endif

#if defined(POLYPREC)
  }
#endif
}


void local_fgmres_PRECISION_struct_free( local_gmres_PRECISION_struct *p, level_struct *l ) {

  int k=0;

  //int m = p->restart_length;
  // no preconditioner
  k = 0;

  if(p->restart_length > 0) {
    FREE( p->H[0], complex_PRECISION, p->total_storage );
    FREE( p->H, complex_PRECISION*, p->restart_length );
    FREE( p->V, complex_PRECISION*, p->restart_length+1 );
  
    if ( p->Z != NULL )
      FREE( p->Z, complex_PRECISION*, k );
  }
  
  p->D = NULL;
  p->clover = NULL;

#if defined(GCRODR) || defined(POLYPREC)
  if (l->level==0) {
#endif

  // copy of Hesselnberg matrix
#if defined(POLYPREC)
  int m = p->restart_length;
  FREE( p->polyprec_PRECISION.eigslvr.Hc[0], complex_PRECISION, m*(m+1) );
  FREE(p->polyprec_PRECISION.eigslvr.Hc, complex_PRECISION*, m);
#endif

#ifdef POLYPREC
  int d_poly = 10;
  int vl = p->polyprec_PRECISION.syst_size;
  FREE( p->polyprec_PRECISION.Hcc, complex_PRECISION, d_poly*d_poly );
  FREE( p->polyprec_PRECISION.L[0], complex_PRECISION, (d_poly+1)*d_poly );
  FREE( p->polyprec_PRECISION.L, complex_PRECISION*, d_poly+1 );
  FREE( p->polyprec_PRECISION.h_ritz,complex_PRECISION, d_poly );
  FREE( p->polyprec_PRECISION.lejas,complex_PRECISION, d_poly );
  FREE( p->polyprec_PRECISION.accum_prod, complex_PRECISION, vl );
  FREE( p->polyprec_PRECISION.product, complex_PRECISION, vl );    
  FREE( p->polyprec_PRECISION.temp, complex_PRECISION, vl );    
  FREE( p->polyprec_PRECISION.xtmp, complex_PRECISION, vl );
  FREE( p->polyprec_PRECISION.random_rhs, complex_PRECISION, vl );
  FREE( p->polyprec_PRECISION.col_prods, complex_PRECISION, d_poly );

  FREE( p->polyprec_PRECISION.eigslvr.vl,complex_PRECISION, d_poly*d_poly );
  FREE( p->polyprec_PRECISION.eigslvr.vr,complex_PRECISION, d_poly*d_poly );  

  FREE( p->polyprec_PRECISION.dirctslvr.ipiv, int, d_poly );
  FREE( p->polyprec_PRECISION.dirctslvr.x, complex_PRECISION, d_poly );
  FREE( p->polyprec_PRECISION.dirctslvr.b, complex_PRECISION, d_poly );
#endif

#if defined(GCRODR) || defined(POLYPREC)
  }
#endif
}


void local_process_multi_inner_product_PRECISION( int count, complex_PRECISION *results, vector_PRECISION *phi, vector_PRECISION psi,
                                                  int start, int end, level_struct *l, struct Thread *threading ) {

  int i;
  for(int c=0; c<count; c++)
    results[c] = 0.0;

  int thread_start=start;
  int thread_end=end;

#ifdef _M10TV
  //compute_core_start_end_custom(start, end, &thread_start, &thread_end, l, threading, 20);
  for(int c=0; c<count; c++)
    for ( i=thread_start; i<thread_end; )
      FOR20( results[c] += conj_PRECISION(phi[c][i])*psi[i]; i++; )
#else
  //compute_core_start_end_custom(start, end, &thread_start, &thread_end, l, threading, 2);
  for(int c=0; c<count; c++)
    for ( i=thread_start; i<thread_end; )
      FOR2( results[c] += conj_PRECISION(phi[c][i])*psi[i]; i++; )
#endif
}


void local_qr_update_PRECISION( complex_PRECISION **H, complex_PRECISION *s,
                                complex_PRECISION *c, complex_PRECISION *gamma, int j,
                                level_struct *l, struct Thread *threading ) {

  SYNC_HYPERTHREADS(threading)
  SYNC_CORES(threading)

  START_MASTER(threading)

  int i;
  complex_PRECISION beta;
  
  // update QR factorization
  // apply previous Givens rotation
  for ( i=0; i<j; i++ ) {
    beta = (-s[i])*H[j][i] + (c[i])*H[j][i+1];
    H[j][i] = conj_PRECISION(c[i])*H[j][i] + conj_PRECISION(s[i])*H[j][i+1];
    H[j][i+1] = beta;
  }
  // compute current Givens rotation
  beta = (complex_PRECISION) sqrt( NORM_SQUARE_PRECISION(H[j][j]) + NORM_SQUARE_PRECISION(H[j][j+1]) );
  s[j] = H[j][j+1]/beta; c[j] = H[j][j]/beta;
  // update right column
  gamma[j+1] = (-s[j])*gamma[j]; gamma[j] = conj_PRECISION(c[j])*gamma[j];
  // apply current Givens rotation
  H[j][j] = beta; H[j][j+1] = 0;

  END_MASTER(threading)

  SYNC_MASTER_TO_ALL(threading);
  SYNC_CORES(threading)
}


int local_fgmres_PRECISION( local_gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading ) {

  int j=-1, finish=0, iter=0, il;
  complex_PRECISION gamma0 = 0;

  PRECISION norm_r0=1, gamma_jp1=1;

  if ( l->depth > 0 ) p->timing = 1;

  int start, end;
  compute_core_start_end(p->v_start, p->v_end, &start, &end, l, threading);

  // towards main loop
  // always assume initial guess set to zero
  vector_PRECISION_copy( p->r, p->b, start, end, l );

  gamma0 = process_norm_PRECISION( p->r, start, end, l, threading );

  START_MASTER(threading)
  p->gamma[0] = gamma0;
  END_MASTER(threading)
  norm_r0 = creal(gamma0);
  vector_PRECISION_real_scale( p->V[0], p->r, 1.0/gamma0, start, end, l ); // v_0 = r / gamma_0

  gamma_jp1 = cabs( gamma0 );

  for( il=0; il<p->restart_length && finish==0; il++) {

    j = il; iter++;

    if ( !local_arnoldi_step_PRECISION( p->V, p->Z, p->w, p->H, p->y, j, NULL, p, l, threading ) ) {
      printf0("| -------------- iteration %d, restart due to H(%d,%d) < 0 |\n", iter, j+1, j );
      break;
    }

    if ( cabs( p->H[j][j+1] ) > p->tol/10 ) {
      local_qr_update_PRECISION( p->H, p->s, p->c, p->gamma, j, l, threading );
      gamma_jp1 = cabs( p->gamma[j+1] );

      //printf0("l (proc=%d) rel residual = %f\n", g.my_rank, gamma_jp1/norm_r0);

      if( gamma_jp1/norm_r0 < p->tol || gamma_jp1/norm_r0 > 1E+5 ) { // if satisfied ... stop
        finish = 1;
        if ( gamma_jp1/norm_r0 > 1E+5 ) printf0("Divergence of fgmres_PRECISION, iter = %d, level=%d\n", iter, l->level );
      }
    } else {
      printf0("from fgmres : depth: %d, iter: %d, p->H(%d,%d) = %+lf+%lfi\n", l->depth, iter, j+1, j, CSPLIT( p->H[j][j+1] ) );
      finish = 1;
      break;
    }
  } // end of a single restart

  return 0;
}


int local_arnoldi_step_PRECISION( vector_PRECISION *V, vector_PRECISION *Z, vector_PRECISION w,
                                  complex_PRECISION **H, complex_PRECISION* buffer, int j, void (*prec)(),
                                  local_gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading ) {

  SYNC_MASTER_TO_ALL(threading);
  SYNC_CORES(threading)

  int i;
  int start, end;
  compute_core_start_end(p->v_start, p->v_end, &start, &end, l, threading);

  p->eval_operator( w, V[j], p->op, l, threading );

  SYNC_MASTER_TO_ALL(threading);
  SYNC_CORES(threading)

  // orthogonalization
  complex_PRECISION tmp[j+1];
  process_multi_inner_product_PRECISION( j+1, tmp, V, w, start, end, l, threading );

  START_MASTER(threading)
  for( i=0; i<=j; i++ )
    H[j][i] = tmp[i];
  END_MASTER(threading)

  SYNC_MASTER_TO_ALL(threading);
  SYNC_CORES(threading)

  for( i=0; i<=j; i++ )
    vector_PRECISION_saxpy( w, w, V[i], -H[j][i], start, end, l );

  PRECISION tmp2 = process_norm_PRECISION( w, start, end, l, threading );

  START_MASTER(threading)
  H[j][j+1] = tmp2;
  END_MASTER(threading)

  // V_j+1 = w / H_j+1,j
  if ( cabs_PRECISION( tmp2 ) > 1e-15 )
    vector_PRECISION_real_scale( V[j+1], w, 1/tmp2, start, end, l );

  int jx = j;

  // copy of Hessenberg matrix (only level=0 currently)
  START_MASTER(threading)
  if (l->dup_H==1 && l->level==0)
  {
    memcpy( p->polyprec_PRECISION.eigslvr.Hc[jx], H[jx], sizeof(complex_PRECISION)*(jx+2) );
    memset( p->polyprec_PRECISION.eigslvr.Hc[jx]+jx+2, 0.0, sizeof(complex_PRECISION)*(p->restart_length + 1 - (jx+2)) );
  }
  END_MASTER(threading)

  SYNC_MASTER_TO_ALL(threading);
  SYNC_CORES(threading)

  return 1;
}


void local_set_ghost_PRECISION( vector_PRECISION phi, const int mu, const int dir,
                                comm_PRECISION_struct *c, const int amount, level_struct *l ) {

  // does not allow sending in both directions at the same time
  if( l->global_splitting[mu] > 1 ) {
    
    int i, j, *table=NULL, mu_dir = 2*mu-MIN(dir,0), offset = c->offset,
        length[2] = {0,0}, comm_start = 0, table_start = 0;
    vector_PRECISION buffer, phi_pt;
    
    if ( amount == _FULL_SYSTEM ) {
      length[0] = (c->num_boundary_sites[2*mu])*offset;
      length[1] = (c->num_boundary_sites[2*mu+1])*offset;
      comm_start = c->comm_start[mu_dir];
      table_start = 0;
    } else if ( amount == _EVEN_SITES ) {
      length[0] = c->num_even_boundary_sites[2*mu]*offset;
      length[1] = c->num_even_boundary_sites[2*mu+1]*offset;
      comm_start = c->comm_start[mu_dir];
      table_start = 0;
    } else if ( amount == _ODD_SITES ) {
      length[0] = c->num_odd_boundary_sites[2*mu]*offset;
      length[1] = c->num_odd_boundary_sites[2*mu+1]*offset;
      comm_start = c->comm_start[mu_dir]+c->num_even_boundary_sites[mu_dir]*offset;
      table_start = c->num_even_boundary_sites[mu_dir];
    }
    
#ifdef HAVE_TM1p1
    if ( g.n_flavours == 2 ) {
      length[0] *= 2;
      length[1] *= 2;
      comm_start *= 2;
      offset *= 2;
    }
#endif

    if ( MAX(length[0],length[1]) > c->max_length[mu] ) {
      printf("CAUTION: my_rank: %d, not enough comm buffer\n", g.my_rank ); fflush(0);
      ghost_free_PRECISION( c, l );
      ghost_alloc_PRECISION( MAX(length[0],length[1]), c, l );
    }
    
    buffer = (vector_PRECISION)c->buffer[mu_dir];

    // dir = senddir
    if ( dir == 1 ) {
      memset( buffer, 0.0, length[1]*sizeof(complex_PRECISION) );
    } else if ( dir == -1 ) {
      phi_pt = phi + comm_start;
      memset( phi_pt, 0.0, length[0]*sizeof(complex_PRECISION) );
    } else ASSERT( dir == 1 || dir == -1 );

    // this second part mimics the ghost_wait
    if ( dir == 1 ) {
      int num_boundary_sites = length[0]/offset;

      buffer = (vector_PRECISION)c->buffer[mu_dir];      
      table = c->boundary_table[2*mu+1] + table_start;

      if ( l->depth == 0 ) {
        for ( j=0; j<num_boundary_sites; j++ ) {
          phi_pt = phi + table[j]*offset;
          for ( i=0; i<offset; i++ )
            phi_pt[i] = buffer[i];
          buffer += offset;
        }
      } else {
        for ( j=0; j<num_boundary_sites; j++ ) {
          phi_pt = phi + table[j]*offset;
          for ( i=0; i<offset; i++ )
            phi_pt[i] += buffer[i];
          buffer += offset;
        }
      }
    } else if ( dir == -1 ) {
      // do nothing
    } else ASSERT( dir == 1 || dir == -1 );
  }
}


void coarse_local_n_hopping_term_PRECISION( vector_PRECISION out, vector_PRECISION in, operator_PRECISION_struct *op,
                                            const int amount, level_struct *l, struct Thread *threading ){

  //START_NO_HYPERTHREADS(threading)

  int mu, i, index, num_site_var=l->num_lattice_site_var,
      num_4link_var=4*4*(l->num_lattice_site_var/2)*(l->num_lattice_site_var/2),
      num_link_var=4*(l->num_lattice_site_var/2)*(l->num_lattice_site_var/2),
      start=0, num_lattice_sites=l->num_inner_lattice_sites,
      plus_dir_param=_FULL_SYSTEM, minus_dir_param=_FULL_SYSTEM;
  vector_PRECISION in_pt, out_pt;
  config_PRECISION D_pt;

  int core_start;
  int core_end;

  // assumptions (1) self coupling has already been performed
  //          OR (2) "out" is initialized with zeros
  //set_boundary_PRECISION( out, 0, l, threading );
  memset( out+l->inner_vector_size, 0.0, (l->vector_size-l->inner_vector_size)*sizeof(complex_PRECISION) );

  if ( amount == _EVEN_SITES ) {
    minus_dir_param = _ODD_SITES;
    plus_dir_param = _EVEN_SITES;
  } else if ( amount == _ODD_SITES ) {
    minus_dir_param = _EVEN_SITES;
    plus_dir_param = _ODD_SITES;
  }

  if ( amount == _EVEN_SITES ) {
    start = op->num_even_sites, num_lattice_sites = op->num_odd_sites;
  } else if ( amount == _ODD_SITES ) {
    start = 0; num_lattice_sites = op->num_even_sites;
  }
  // no cores here, everything is per-process
  core_start = start;
  core_end = start+num_lattice_sites;

  // compute U_mu^dagger coupling
  for ( i=core_start; i<core_end; i++ ) {
    index = 5*i;
    in_pt = in + num_site_var*op->neighbor_table[index];
    D_pt = op->D + num_4link_var*op->neighbor_table[index] + 0*num_link_var;
    index++;
    out_pt = out + num_site_var*op->neighbor_table[index+T];
    coarse_n_daggered_hopp_PRECISION( out_pt, in_pt, D_pt, l );
  }
  for ( i=core_start; i<core_end; i++ ) {
    index = 5*i;
    in_pt = in + num_site_var*op->neighbor_table[index];
    D_pt = op->D + num_4link_var*op->neighbor_table[index] + 1*num_link_var;
    index++;
    out_pt = out + num_site_var*op->neighbor_table[index+Z];
    coarse_n_daggered_hopp_PRECISION( out_pt, in_pt, D_pt, l );
  }
  for ( i=core_start; i<core_end; i++ ) {
    index = 5*i;
    in_pt = in + num_site_var*op->neighbor_table[index];
    D_pt = op->D + num_4link_var*op->neighbor_table[index] + 2*num_link_var;
    index++;
    out_pt = out + num_site_var*op->neighbor_table[index+Y];
    coarse_n_daggered_hopp_PRECISION( out_pt, in_pt, D_pt, l );
  }
  for ( i=core_start; i<core_end; i++ ) {
    index = 5*i;
    in_pt = in + num_site_var*op->neighbor_table[index];
    D_pt = op->D + num_4link_var*op->neighbor_table[index] + 3*num_link_var;
    index++;
    out_pt = out + num_site_var*op->neighbor_table[index+X];
    coarse_n_daggered_hopp_PRECISION( out_pt, in_pt, D_pt, l );
  }

  // instead of ghost exchanges, set &(op->c) to zero
  if ( op->c.comm ) {
    for ( mu=0; mu<4; mu++ ) {
      local_set_ghost_PRECISION( in, mu, -1, &(op->c), minus_dir_param, l );
    }
  }
  if ( op->c.comm ) {
    for ( mu=0; mu<4; mu++ ) {
      local_set_ghost_PRECISION( out, mu, +1, &(op->c), plus_dir_param, l );    
    }
  }

  //memset( in+l->inner_vector_size, 0.0, (l->vector_size-l->inner_vector_size)*sizeof(complex_PRECISION) );
  //memset( out+l->inner_vector_size, 0.0, (l->vector_size-l->inner_vector_size)*sizeof(complex_PRECISION) );

  if ( amount == _EVEN_SITES ) {
    start = 0; num_lattice_sites = op->num_even_sites;
  } else if ( amount == _ODD_SITES ) {
    start = op->num_even_sites, num_lattice_sites = op->num_odd_sites;
  }
  // no cores here, everything is per-process
  core_start = start;
  core_end = start+num_lattice_sites;

  // compute U_mu couplings
  for ( i=core_start; i<core_end; i++ ) {
    index = 5*i;
    out_pt = out + num_site_var*op->neighbor_table[index];
    D_pt = op->D + num_4link_var*op->neighbor_table[index];
    index++;
    in_pt = in + num_site_var*op->neighbor_table[index+T];
    coarse_n_hopp_PRECISION( out_pt, in_pt, D_pt, l );
    
    D_pt += num_link_var;
    in_pt = in + num_site_var*op->neighbor_table[index+Z];
    coarse_n_hopp_PRECISION( out_pt, in_pt, D_pt, l );
    
    D_pt += num_link_var;
    in_pt = in + num_site_var*op->neighbor_table[index+Y];
    coarse_n_hopp_PRECISION( out_pt, in_pt, D_pt, l );
    
    D_pt += num_link_var;
    in_pt = in + num_site_var*op->neighbor_table[index+X];
    coarse_n_hopp_PRECISION( out_pt, in_pt, D_pt, l );
  }

  //END_NO_HYPERTHREADS(threading)
}


void coarse_local_hopping_term_PRECISION( vector_PRECISION out, vector_PRECISION in, operator_PRECISION_struct *op,
                                          const int amount, level_struct *l, struct Thread *threading ) {

  //START_NO_HYPERTHREADS(threading)

  int mu, i, index, num_site_var=l->num_lattice_site_var,
      num_4link_var=4*4*(l->num_lattice_site_var/2)*(l->num_lattice_site_var/2),
      num_link_var=4*(l->num_lattice_site_var/2)*(l->num_lattice_site_var/2),
      start=0, num_lattice_sites=l->num_inner_lattice_sites,
      plus_dir_param=_FULL_SYSTEM, minus_dir_param=_FULL_SYSTEM;
  vector_PRECISION in_pt, out_pt;
  config_PRECISION D_pt;

  int core_start;
  int core_end;

  // assumptions (1) self coupling has already been performed
  //          OR (2) "out" is initialized with zeros
  //set_boundary_PRECISION( out, 0, l, threading );
  memset( out+l->inner_vector_size, 0.0, (l->vector_size-l->inner_vector_size)*sizeof(complex_PRECISION) );

  if ( amount == _EVEN_SITES ) {
    minus_dir_param = _ODD_SITES;
    plus_dir_param = _EVEN_SITES;
  } else if ( amount == _ODD_SITES ) {
    minus_dir_param = _EVEN_SITES;
    plus_dir_param = _ODD_SITES;
  }

  if ( amount == _EVEN_SITES ) {
    start = op->num_even_sites, num_lattice_sites = op->num_odd_sites;
  } else if ( amount == _ODD_SITES ) {
    start = 0; num_lattice_sites = op->num_even_sites;
  }
  // no cores here, everything is per-process
  core_start = start;
  core_end = start+num_lattice_sites;

  // compute U_mu^dagger coupling

  for ( i=core_start; i<core_end; i++ ) {
    index = 5*i;
    in_pt = in + num_site_var*op->neighbor_table[index];
    D_pt = op->D + num_4link_var*op->neighbor_table[index] + 0*num_link_var;
    index++;
    out_pt = out + num_site_var*op->neighbor_table[index+T];
    coarse_daggered_hopp_PRECISION( out_pt, in_pt, D_pt, l );
  }

  for ( i=core_start; i<core_end; i++ ) {
    index = 5*i;
    in_pt = in + num_site_var*op->neighbor_table[index];
    D_pt = op->D + num_4link_var*op->neighbor_table[index] + 1*num_link_var;
    index++;
    out_pt = out + num_site_var*op->neighbor_table[index+Z];
    coarse_daggered_hopp_PRECISION( out_pt, in_pt, D_pt, l );
  }
  for ( i=core_start; i<core_end; i++ ) {
    index = 5*i;
    in_pt = in + num_site_var*op->neighbor_table[index];
    D_pt = op->D + num_4link_var*op->neighbor_table[index] + 2*num_link_var;
    index++;
    out_pt = out + num_site_var*op->neighbor_table[index+Y];
    coarse_daggered_hopp_PRECISION( out_pt, in_pt, D_pt, l );
  }
  for ( i=core_start; i<core_end; i++ ) {
    index = 5*i;
    in_pt = in + num_site_var*op->neighbor_table[index];
    D_pt = op->D + num_4link_var*op->neighbor_table[index] + 3*num_link_var;
    index++;
    out_pt = out + num_site_var*op->neighbor_table[index+X];
    coarse_daggered_hopp_PRECISION( out_pt, in_pt, D_pt, l );
  }

  // instead of ghost exchanges, set &(op->c) to zero
  if ( op->c.comm ) {
    for ( mu=0; mu<4; mu++ ) {
      local_set_ghost_PRECISION( in, mu, -1, &(op->c), minus_dir_param, l );
    }
  }
  if ( op->c.comm ) {
    for ( mu=0; mu<4; mu++ ) {
      local_set_ghost_PRECISION( out, mu, +1, &(op->c), plus_dir_param, l );    
    }
  }

  //memset( in+l->inner_vector_size, 0.0, (l->vector_size-l->inner_vector_size)*sizeof(complex_PRECISION) );
  //memset( out+l->inner_vector_size, 0.0, (l->vector_size-l->inner_vector_size)*sizeof(complex_PRECISION) );

  if ( amount == _EVEN_SITES ) {
    start = 0; num_lattice_sites = op->num_even_sites;
  } else if ( amount == _ODD_SITES ) {
    start = op->num_even_sites, num_lattice_sites = op->num_odd_sites;
  }
  // no cores here, everything is per-process
  core_start = start;
  core_end = start+num_lattice_sites;

  // compute U_mu couplings
  for ( i=core_start; i<core_end; i++ ) {
    index = 5*i;
    out_pt = out + num_site_var*op->neighbor_table[index];
    D_pt = op->D + num_4link_var*op->neighbor_table[index];
    index++;
    in_pt = in + num_site_var*op->neighbor_table[index+T];
    coarse_hopp_PRECISION( out_pt, in_pt, D_pt, l );
    
    D_pt += num_link_var;
    in_pt = in + num_site_var*op->neighbor_table[index+Z];
    coarse_hopp_PRECISION( out_pt, in_pt, D_pt, l );
    
    D_pt += num_link_var;
    in_pt = in + num_site_var*op->neighbor_table[index+Y];
    coarse_hopp_PRECISION( out_pt, in_pt, D_pt, l );
    
    D_pt += num_link_var;
    in_pt = in + num_site_var*op->neighbor_table[index+X];
    coarse_hopp_PRECISION( out_pt, in_pt, D_pt, l );
  }

  //END_NO_HYPERTHREADS(threading)
}


// CHANGED WRT TM CODE
/*
void coarse_local_diag_oo_inv_PRECISION( vector_PRECISION y, vector_PRECISION x, operator_PRECISION_struct *op, 
                                         level_struct *l, struct Thread *threading ) {

  int start, end;
  start = 0;
  end = op->num_odd_sites;

  // odd sites
  int num_site_var = l->num_lattice_site_var,
    oo_inv_size = SQUARE(num_site_var);

#ifndef OPTIMIZED_COARSE_SELF_COUPLING_PRECISION
#ifdef HAVE_TM1p1
  config_PRECISION sc = (g.n_flavours==2) ? op->clover_doublet_oo_inv:op->clover_oo_inv;
#else
  config_PRECISION sc = op->clover_oo_inv;
#endif
#else
  int lda = SIMD_LENGTH_PRECISION*((num_site_var+SIMD_LENGTH_PRECISION-1)/SIMD_LENGTH_PRECISION);
  oo_inv_size = 2*num_site_var*lda;
#ifdef HAVE_TM1p1
  OPERATOR_TYPE_PRECISION *sc = (g.n_flavours==2) ? op->clover_doublet_oo_inv_vectorized:op->clover_oo_inv_vectorized;
#else
  OPERATOR_TYPE_PRECISION *sc = op->clover_oo_inv_vectorized;
#endif
#endif

  x += num_site_var*(op->num_even_sites+start);
  y += num_site_var*(op->num_even_sites+start);  
  sc += oo_inv_size*start;

  for ( int i=start; i<end; i++ ) {
#ifndef OPTIMIZED_COARSE_SELF_COUPLING_PRECISION
    coarse_perform_fwd_bwd_subs_PRECISION( y, x, sc, l );
#else
    for(int j=0; j<num_site_var; j++)
      y[j] = _COMPLEX_PRECISION_ZERO;
    cgemv( num_site_var, sc, lda, (float *)x, (float *)y);
#endif
    x += num_site_var;
    y += num_site_var;
    sc += oo_inv_size;
  }
}
*/

// CHANGED WRT TM CODE
/*
void coarse_local_diag_ee_PRECISION( vector_PRECISION y, vector_PRECISION x, operator_PRECISION_struct *op, level_struct *l, struct Thread *threading ) {
  
  int start, end;
  start = 0;
  end = op->num_even_sites;

  // even sites
#ifndef OPTIMIZED_COARSE_SELF_COUPLING_PRECISION
  coarse_self_couplings_PRECISION( y, x, op, start, end, l );
#else
  coarse_self_couplings_PRECISION_vectorized( y, x, op, start, end, l );
#endif
}
*/


void coarse_local_apply_schur_complement_PRECISION( vector_PRECISION out, vector_PRECISION in,
                                                    operator_PRECISION_struct *op, level_struct *l,
                                                    struct Thread *threading ) {
  SYNC_MASTER_TO_ALL(threading);
  SYNC_CORES(threading)

  int start, end;
  compute_core_start_end(op->num_even_sites*l->num_lattice_site_var, l->inner_vector_size, &start, &end, l, threading);

  vector_PRECISION *tmp = op->buffer;

  //coarse_local_diag_ee_PRECISION( out, in, op, l, threading );
  coarse_diag_ee_PRECISION( out, in, op, l, threading );

  vector_PRECISION_define( tmp[0], 0, start, end, l );

  //coarse_local_hopping_term_PRECISION( tmp[0], in, op, _ODD_SITES, l, threading );
  //coarse_local_diag_oo_inv_PRECISION( tmp[1], tmp[0], op, l, threading );
  //coarse_local_n_hopping_term_PRECISION( out, tmp[1], op, _EVEN_SITES, l, threading );

  //coarse_local_diag_oo_inv_PRECISION( tmp[0], in, op, l, threading );
  coarse_diag_oo_inv_PRECISION( tmp[0], in, op, l, threading );

  vector_PRECISION_minus( out, out, tmp[0], start, end, l );

  SYNC_MASTER_TO_ALL(threading);
  SYNC_CORES(threading)
}


// ------------------------- LOCAL POLYPREC -------------------------


void local_harmonic_ritz_PRECISION( local_gmres_PRECISION_struct *p )
{
  int i, j, d;
  complex_PRECISION h_dd;

  d = p->polyprec_PRECISION.d_poly;
  h_dd = p->polyprec_PRECISION.Hc[d-1][d];
  memset(p->polyprec_PRECISION.dirctslvr.b, 0.0, sizeof(complex_PRECISION)*(d-1));
  p->polyprec_PRECISION.dirctslvr.b[d-1] = 1.;

  for (i=0; i<d; i++)
    for (j=0; j<d; j++)
      p->polyprec_PRECISION.Hcc[i*d + j ] = conj(p->polyprec_PRECISION.Hc[j][i]);

  p->polyprec_PRECISION.dirctslvr.dirctslvr_PRECISION(&p->polyprec_PRECISION.dirctslvr);

  for (i=0; i<d; i++)
    p->polyprec_PRECISION.Hc[d-1][i] += h_dd*h_dd*p->polyprec_PRECISION.dirctslvr.x[i];
    
  p->polyprec_PRECISION.eigslvr.eigslvr_PRECISION(&p->polyprec_PRECISION.eigslvr);
}


void local_leja_ordering_PRECISION( local_gmres_PRECISION_struct *p )
{

  int i, j, ii, d_poly;
  int max_j, exchange_cols;
  complex_PRECISION tmp, leja;

  complex_PRECISION** L;
  complex_PRECISION* col_prods;

  d_poly = p->polyprec_PRECISION.d_poly;
  L = p->polyprec_PRECISION.L;
  col_prods = p->polyprec_PRECISION.col_prods;

  // Create a matrix made of n+1 rows, each row is x (all rows equal).
  for (i=0; i<d_poly+1; i++ )
    memcpy( L[i], p->polyprec_PRECISION.h_ritz, sizeof(complex_PRECISION)*(d_poly) );

  leja = 0;

  for (i=0; i < d_poly-1; i++)
  {

    for (j=i; j<d_poly; j++ ) 
      L[i][j] = cabs( L[i][j] - leja );

    for (j = i; j < d_poly; j++)
    {
      col_prods[j] = 1.;
      for (ii = 0; ii <= i; ii++)
        col_prods[j] *= L[ii][j];
    }

    exchange_cols = 0;
    max_j = i;
    for (j=i+1; j<d_poly; j++ )
    {
      if ( creal(col_prods[j]) > creal(col_prods[max_j]) )
      {
        max_j = j; 
        exchange_cols = 1;
      }
    }
        
    if (exchange_cols)
    {
      for (ii=0; ii<d_poly+1; ii++ )
      {
        tmp = L[ii][i];
        L[ii][i] = L[ii][max_j];
        L[ii][max_j] = tmp;
      } 
    }

    leja = L[d_poly][i];

  }

  memcpy( p->polyprec_PRECISION.lejas, p->polyprec_PRECISION.L[d_poly], sizeof(complex_PRECISION)*(d_poly) );
}



void local_update_lejas_PRECISION( local_gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading )
{
  int start, end;
  compute_core_start_end(p->v_start, p->v_end, &start, &end, l, threading);

  vector_PRECISION random_rhs, buff0;
  random_rhs = p->polyprec_PRECISION.random_rhs;
  PRECISION buff3;
  vector_PRECISION buff4;

  int buff1, buff2;

  buff0 = p->b;
  buff2 = p->num_restart;
  buff1 = p->restart_length;
  buff3 = p->tol;
  buff4 = p->x;

  START_MASTER(threading)
  p->b = random_rhs;
  p->num_restart = 1;
  p->restart_length = p->polyprec_PRECISION.d_poly;
  p->tol = 1E-20;
  p->x = p->polyprec_PRECISION.xtmp;

  l->dup_H = 1;
  END_MASTER(threading)

  SYNC_MASTER_TO_ALL(threading);
  SYNC_CORES(threading)

  START_MASTER(threading)
  vector_PRECISION_define_random( random_rhs, p->v_start, p->v_end, l );
  END_MASTER(threading)

  SYNC_MASTER_TO_ALL(threading);
  SYNC_CORES(threading)

  local_fgmres_PRECISION(p, l, threading);

  SYNC_MASTER_TO_ALL(threading);
  SYNC_CORES(threading)

  START_MASTER(threading)
  l->dup_H = 0;

  p->b = buff0;
  p->num_restart = buff2;
  p->restart_length = buff1;
  p->tol = buff3;
  p->x = buff4;

  // TODO : re-enable this
  p->polyprec_PRECISION.update_lejas = 0;
  l->p_PRECISION.block_jacobi_PRECISION.BJ_usable = 1;
  END_MASTER(threading)

  START_MASTER(threading)
  local_harmonic_ritz_PRECISION(p);
  local_leja_ordering_PRECISION(p);
  END_MASTER(threading)
}


void local_re_construct_lejas_PRECISION( level_struct *l, struct Thread *threading ) {

  local_update_lejas_PRECISION(&(l->p_PRECISION.block_jacobi_PRECISION.local_p), l, threading);

  //local_gmres_PRECISION_struct *p = &(l->p_PRECISION.block_jacobi_PRECISION.local_p);
  //START_MASTER(threading)
  //int d_poly = p->polyprec_PRECISION.d_poly;
  //printf0("lejas (proc=%d) =  ", g.my_rank);
  //for ( int i=0; i<d_poly; i++ ) {
  //  printf0( "%f+%fj  ", creal(p->polyprec_PRECISION.lejas[i]), cimag(p->polyprec_PRECISION.lejas[i]) );
  //}
  //printf0("\n");
  //END_MASTER(threading)

  SYNC_MASTER_TO_ALL(threading);
  SYNC_CORES(threading)
}



void local_apply_polyprec_PRECISION( vector_PRECISION phi, vector_PRECISION Dphi, vector_PRECISION eta,
                                     int res, level_struct *l, struct Thread *threading )
{
  SYNC_MASTER_TO_ALL(threading);
  SYNC_CORES(threading)

  int i;

  int start, end;
  compute_core_start_end(l->p_PRECISION.block_jacobi_PRECISION.local_p.v_start, l->p_PRECISION.block_jacobi_PRECISION.local_p.v_end, &start, &end, l, threading);

  local_gmres_PRECISION_struct *p = &( l->p_PRECISION.block_jacobi_PRECISION.local_p );

  int d_poly = p->polyprec_PRECISION.d_poly;
  vector_PRECISION accum_prod = p->polyprec_PRECISION.accum_prod;
  vector_PRECISION product = p->polyprec_PRECISION.product;
  vector_PRECISION temp = p->polyprec_PRECISION.temp;
  vector_PRECISION lejas = p->polyprec_PRECISION.lejas;

  vector_PRECISION_copy( product, eta, start, end, l );
  vector_PRECISION_define(accum_prod, 0.0, start, end, l);

  vector_PRECISION_saxpy(accum_prod, accum_prod, product, 1./lejas[0], start, end, l);
  for (i = 1; i < d_poly; i++)
  {
    p->eval_operator(temp, product, p->op, l, threading);

    vector_PRECISION_saxpy(product, product, temp, -1./lejas[i-1], start, end, l);
    vector_PRECISION_saxpy(accum_prod, accum_prod, product, 1./lejas[i], start, end, l);
  }
  vector_PRECISION_copy( phi, accum_prod, start, end, l );

  SYNC_MASTER_TO_ALL(threading);
  SYNC_CORES(threading)
}


#endif
