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

#ifdef POLYPREC


/*-----------------------------------------------*/

void print_matrix_PRECISION(complex_PRECISION* A, int mv, int mh )
{
  int i,j;

  // printf("\n\n");
  // for (i=0; i < mv; i++)
  // {
  //     for(j=0; j < mh; j++)
  //     {
  //             fprintf(stdout, "%6.6f +i%6.6f\t", creal(A[i*mh + j]), cimag(A[i*mh+j]));
  //     }
  //     fprintf(stdout, "\n");
  // }
  // printf("--\n");

  printf("\n\n");
  for (i=0; i < mh; i++)
  {
    for(j=0; j < mv; j++)
    {
      fprintf(stdout, "%6.6f +i%6.6f\t", creal(A[j*mh + i]), cimag(A[j*mh+i]));
    }
    fprintf(stdout, "\n");
  }
  printf("--\n");
  printf("\n\n");
}


void print_vector_PRECISION( char* desc, vector_PRECISION w, int n)
{
  int j;
  printf0( "\n %s\n", desc );
  for( j = 0; j < n; j++ ) printf0( " (%6.6f,%6.6f)", creal(w[j]), cimag(w[j]) );
  printf0( "\n" );
}


void harmonic_ritz_PRECISION( gmres_PRECISION_struct *p )
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



/*-----------------------------------------------*/



void leja_ordering_PRECISION( gmres_PRECISION_struct *p )
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



int update_lejas_PRECISION( gmres_PRECISION_struct *p, level_struct *l, struct Thread *threading )
{
  int start, end;
  compute_core_start_end(p->v_start, p->v_end, &start, &end, l, threading);

  vector_PRECISION random_rhs, buff0;
  random_rhs = p->polyprec_PRECISION.random_rhs;
  PRECISION buff3, buff5;
  vector_PRECISION buff4;

  int buff1, buff2;
  int fgmres_itersx;

  buff0 = p->b;
  buff2 = p->num_restart;
  buff1 = p->restart_length;
  buff3 = p->tol;
  buff4 = p->x;
  buff5 = g.coarse_tol;

  SYNC_MASTER_TO_ALL(threading)
  SYNC_CORES(threading)

  START_MASTER(threading)
  p->b = random_rhs;
  p->num_restart = 1;
  p->restart_length = p->polyprec_PRECISION.d_poly;
  p->preconditioner = NULL;
  p->tol = 1E-20;
  g.coarse_tol = 1E-20;
  p->x = p->polyprec_PRECISION.xtmp;
  l->dup_H = 1;
  vector_PRECISION_define_random( random_rhs, p->v_start, p->v_end, l );
  END_MASTER(threading)

  SYNC_MASTER_TO_ALL(threading)
  SYNC_CORES(threading)

  fgmres_itersx = fgmres_PRECISION(p, l, threading);

  //printf0( "FROM WITHIN POLYPREC SETUP : %d, d POLY = %d\n",fgmres_itersx,p->polyprec_PRECISION.d_poly );

  SYNC_MASTER_TO_ALL(threading)
  SYNC_CORES(threading)

  START_MASTER(threading)
  l->dup_H = 0;
  p->b = buff0;
  p->num_restart = buff2;
  p->restart_length = buff1;
  p->tol = buff3;
  g.coarse_tol = buff5;
  p->x = buff4;
  END_MASTER(threading)

  SYNC_MASTER_TO_ALL(threading);
  SYNC_CORES(threading);

  if ( fgmres_itersx == p->polyprec_PRECISION.d_poly ) {
    START_MASTER(threading)
    p->polyprec_PRECISION.preconditioner = p->polyprec_PRECISION.preconditioner_bare;
    l->p_PRECISION.polyprec_PRECISION.update_lejas = 0;
    END_MASTER(threading)

    SYNC_MASTER_TO_ALL(threading);
    SYNC_CORES(threading);

  } else { return -1; }

  START_MASTER(threading)
  harmonic_ritz_PRECISION(p);
  leja_ordering_PRECISION(p);
  END_MASTER(threading)

  SYNC_MASTER_TO_ALL(threading)
  SYNC_CORES(threading)

  return 1;
}



int re_construct_lejas_PRECISION( level_struct *l, struct Thread *threading ) {

  //printf0("UPDATED LEJAS\n");

  return update_lejas_PRECISION(&(l->p_PRECISION), l, threading);

}


void apply_polyprec_PRECISION( vector_PRECISION phi, vector_PRECISION Dphi, vector_PRECISION eta,
                               int res, level_struct *l, struct Thread *threading )
{

  //printf0("APPLYING POLYNOMIAL\n");

  int i, start, end;

  compute_core_start_end(l->p_PRECISION.v_start, l->p_PRECISION.v_end, &start, &end, l, threading);

  int d_poly = l->p_PRECISION.polyprec_PRECISION.d_poly;
  vector_PRECISION accum_prod = l->p_PRECISION.polyprec_PRECISION.accum_prod;
  vector_PRECISION product = l->p_PRECISION.polyprec_PRECISION.product;
  vector_PRECISION temp = l->p_PRECISION.polyprec_PRECISION.temp;
  vector_PRECISION lejas = l->p_PRECISION.polyprec_PRECISION.lejas;

  vector_PRECISION_copy( product, eta, start, end, l );
  vector_PRECISION_define(accum_prod, 0.0, start, end, l);
  vector_PRECISION_saxpy(accum_prod, accum_prod, product, 1./lejas[0], start, end, l);

  for (i = 1; i < d_poly; i++)
  {
#ifdef PERS_COMMS
    g.pers_comms_id2 = l->p_PRECISION.restart_length + g.pers_comms_nrZxs;
    g.use_pers_comms1 = 1;
#endif

    SYNC_MASTER_TO_ALL(threading)
    SYNC_CORES(threading)

    apply_operator_PRECISION(temp, product, &l->p_PRECISION, l, threading);

#ifdef PERS_COMMS
    g.pers_comms_id2 = -1;
    g.use_pers_comms1 = 0;
#endif

    vector_PRECISION_saxpy(product, product, temp, -1./lejas[i-1], start, end, l);
    vector_PRECISION_saxpy(accum_prod, accum_prod, product, 1./lejas[i], start, end, l);
  }

  vector_PRECISION_copy( phi, accum_prod, start, end, l );

  SYNC_MASTER_TO_ALL(threading)
  SYNC_CORES(threading)
}

#endif
