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

#include "main.h"
#include "vcycle_PRECISION.h"
#ifdef RICHARDSON_SMOOTHER
#include "proxies/linsolve_proxy_PRECISION.h"
#endif
#include "dirac_PRECISION.h"
#include "proxies/dirac_proxy_PRECISION.h"
#include "proxies/oddeven_proxy_PRECISION.h"

void smoother_PRECISION( vector_PRECISION phi, vector_PRECISION Dphi, vector_PRECISION eta,
                         int n, const int res, complex_PRECISION shift, level_struct *l, struct Thread *threading ) {

  ASSERT( phi != eta );

  START_MASTER(threading);
  PROF_PRECISION_START( _SM );
  END_MASTER(threading);

  if ( g.method == 1 ) {
    additive_schwarz_PRECISION( phi, Dphi, eta, n, res, &(l->s_PRECISION), l, threading );
  } else if ( g.method == 2 ) {

#ifdef CUDA_OPT
    if(l->depth==0){
      START_LOCKED_MASTER(threading)
      schwarz_PRECISION_CUDA( phi, Dphi, eta, n, res, &(l->s_PRECISION), l, threading );
      END_LOCKED_MASTER(threading)

      SYNC_CORES(threading)

      //START_LOCKED_MASTER(threading)
      //printf("(%d) right after smoother...\n", g.my_rank);
      //END_LOCKED_MASTER(threading)
    }
    else{
      //schwarz_PRECISION( phi, Dphi, eta, n, res, &(l->s_PRECISION), l, threading );
      red_black_schwarz_PRECISION( phi, Dphi, eta, n, res, &(l->s_PRECISION), l, threading );
    }
#else
    //schwarz_PRECISION( phi, Dphi, eta, n, res, &(l->s_PRECISION), l, threading );
    red_black_schwarz_PRECISION( phi, Dphi, eta, n, res, &(l->s_PRECISION), l, threading );
#endif
  } else if ( g.method == 3 ) {
    sixteen_color_schwarz_PRECISION( phi, Dphi, eta, n, res, &(l->s_PRECISION), l, threading );
  } else {
    int start = threading->start_index[l->depth];
    int end   = threading->end_index[l->depth];
    START_LOCKED_MASTER(threading)
    l->sp_PRECISION.shift = shift;
    l->sp_PRECISION.initial_guess_zero = res;
    l->sp_PRECISION.num_restart = n;
    END_LOCKED_MASTER(threading)
    if ( g.method == 4 || g.method == 6 ) {
      if ( g.odd_even ) {
        if ( res == _RES ) {
#ifdef CUDA_OPT
          // FIXME : this has to be fixed : forcing the double-precision Dirac operator to
          // be done on CPUs, as things are not prepared properly currently for running
          // it on GPUs
          START_MASTER(threading)
          l->p_PRECISION.eval_operator = d_plus_clover_PRECISION_cpu;
          END_MASTER(threading)
          SYNC_CORES(threading)
          apply_operator_PRECISION( l->sp_PRECISION.x, phi, &(l->p_PRECISION), l, threading );
          START_MASTER(threading)
          l->p_PRECISION.eval_operator = d_plus_clover_PRECISION;
          END_MASTER(threading)
          SYNC_CORES(threading)
#else
          apply_operator_PRECISION( l->sp_PRECISION.x, phi, &(l->p_PRECISION), l, threading );
#endif
          vector_PRECISION_minus( l->sp_PRECISION.x, eta, l->sp_PRECISION.x, start, end, l );
        }
        block_to_oddeven_PRECISION( l->sp_PRECISION.b, res==_RES?l->sp_PRECISION.x:eta, l, threading );
        START_LOCKED_MASTER(threading)
        l->sp_PRECISION.initial_guess_zero = _NO_RES;
        END_LOCKED_MASTER(threading)
        if ( g.method == 6 ) {
          if ( l->depth == 0 ) g5D_solve_oddeven_PRECISION( &(l->sp_PRECISION), &(l->oe_op_PRECISION), l, threading );
          else g5D_coarse_solve_odd_even_PRECISION( &(l->sp_PRECISION), &(l->oe_op_PRECISION), l, threading );
        } else {
          if ( l->depth == 0 ) {
#ifdef GCR_SMOOTHER
            START_MASTER(threading)
            l->sp_PRECISION.use_gcr = 1;
            END_MASTER(threading)
            SYNC_CORES(threading)
#endif
#ifdef RICHARDSON_SMOOTHER
            START_MASTER(threading)
            l->sp_PRECISION.use_richardson = 1;
            END_MASTER(threading)
            SYNC_CORES(threading)
#endif

            START_MASTER(threading);
            PROF_PRECISION_START( _SM_OE );
            END_MASTER(threading);

#if defined(RICHARDSON_SMOOTHER)
            // only Richardson enabled as GPU odd-even finest-level smoother at the moment
            solve_oddeven_PRECISION( &(l->sp_PRECISION), &(l->oe_op_PRECISION), l, threading );
#else
            solve_oddeven_PRECISION_cpu( &(l->sp_PRECISION), &(l->oe_op_PRECISION), l, threading );
#endif

            START_MASTER(threading);
            PROF_PRECISION_STOP( _SM_OE, 1 );
            END_MASTER(threading);

#ifdef GCR_SMOOTHER
            START_MASTER(threading)
            l->sp_PRECISION.use_gcr = 0;
            END_MASTER(threading)
            SYNC_CORES(threading)
#endif
#ifdef RICHARDSON_SMOOTHER
            START_MASTER(threading)
            l->sp_PRECISION.use_richardson = 0;
            END_MASTER(threading)
            SYNC_CORES(threading)
#endif
          } else {
#ifdef GCR_SMOOTHER
            START_MASTER(threading)
            l->sp_PRECISION.use_gcr = 1;
            END_MASTER(threading)
            SYNC_CORES(threading)
#endif
#ifdef RICHARDSON_SMOOTHER
            START_MASTER(threading)
            l->sp_PRECISION.use_richardson = 1;
            END_MASTER(threading)
            SYNC_CORES(threading)
#endif

            coarse_solve_odd_even_PRECISION( &(l->sp_PRECISION), &(l->oe_op_PRECISION), l, threading );

#ifdef GCR_SMOOTHER
            START_MASTER(threading)
            l->sp_PRECISION.use_gcr = 0;
            END_MASTER(threading)
            SYNC_CORES(threading)
#endif
#ifdef RICHARDSON_SMOOTHER
            START_MASTER(threading)
            l->sp_PRECISION.use_richardson = 0;
            END_MASTER(threading)
            SYNC_CORES(threading)
#endif
          }
        }
        if ( res == _NO_RES ) {
          oddeven_to_block_PRECISION( phi, l->sp_PRECISION.x, l, threading );
        } else {
          oddeven_to_block_PRECISION( l->sp_PRECISION.b, l->sp_PRECISION.x, l, threading );
          vector_PRECISION_plus( phi, phi, l->sp_PRECISION.b, start, end, l );
        }
      } else {
        START_LOCKED_MASTER(threading)
        l->sp_PRECISION.x = phi; l->sp_PRECISION.b = eta;
        END_LOCKED_MASTER(threading)
#ifdef GCR_SMOOTHER
        fgcr_PRECISION( &(l->sp_PRECISION), l, threading );
#elif RICHARDSON_SMOOTHER
        richardson_PRECISION( &(l->sp_PRECISION), l, threading );
#else
        fgmres_PRECISION( &(l->sp_PRECISION), l, threading );
#endif
      }
    } else if ( g.method == 5 ) {
      vector_PRECISION_copy( l->sp_PRECISION.b, eta, start, end, l );
      bicgstab_PRECISION( &(l->sp_PRECISION), l, threading );
      vector_PRECISION_copy( phi, l->sp_PRECISION.x, start, end, l );
    }
    ASSERT( Dphi == NULL );
  }
  
  START_MASTER(threading);
  PROF_PRECISION_STOP( _SM, n );
  END_MASTER(threading);
}


void vcycle_PRECISION( vector_PRECISION phi, vector_PRECISION Dphi, vector_PRECISION eta,
                       int res, level_struct *l, struct Thread *threading ) {

  int fgmres_ctr = 0;

  if ( g.interpolation && l->level>0 ) {
    for ( int i=0; i<l->n_cy; i++ ) {
      if ( i==0 && res == _NO_RES ) {
        restrict_PRECISION( l->next_level->p_PRECISION.b, eta, l, threading );
      } else {
        int start = threading->start_index[l->depth];
        int end   = threading->end_index[l->depth];
#ifdef CUDA_OPT
        // FIXME : this has to be fixed : forcing the double-precision Dirac operator to
        // be done on CPUs, as things are not prepared properly currently for running
        // it on GPUs. Note, though, that entering this condition i.e. with a non-zero
        // initial guess and more than one V-cycle applications is quite unusual when calling
        // this function vcycle_PRECISION(...)
        START_MASTER(threading)
        l->p_PRECISION.eval_operator = d_plus_clover_PRECISION_cpu;
        END_MASTER(threading)
        SYNC_CORES(threading)
        apply_operator_PRECISION( l->vbuf_PRECISION[2], phi, &(l->p_PRECISION), l, threading );
        START_MASTER(threading)
        l->p_PRECISION.eval_operator = d_plus_clover_PRECISION;
        END_MASTER(threading)
        SYNC_CORES(threading)
#else
        apply_operator_PRECISION( l->vbuf_PRECISION[2], phi, &(l->p_PRECISION), l, threading );
#endif
        vector_PRECISION_minus( l->vbuf_PRECISION[3], eta, l->vbuf_PRECISION[2], start, end, l );
        restrict_PRECISION( l->next_level->p_PRECISION.b, l->vbuf_PRECISION[3], l, threading );
      }
      if ( !l->next_level->idle ) {
        START_MASTER(threading)
        if ( l->depth == 0 )
          g.coarse_time -= MPI_Wtime();
        END_MASTER(threading)
        if ( l->level > 1 ) {
          if ( g.kcycle ){
            fgmres_ctr = fgmres_PRECISION( &(l->next_level->p_PRECISION), l->next_level, threading );
            //fgmres_PRECISION( &(l->next_level->p_PRECISION), l->next_level, threading );
            if(l->depth == 1){
              printf0("%d, %d -- ", l->level, fgmres_ctr);
            }
          }
          else{
            vcycle_PRECISION( l->next_level->p_PRECISION.x, NULL, l->next_level->p_PRECISION.b, _NO_RES, l->next_level, threading );
          }
        } else {
          if ( g.odd_even ) {
            if ( g.method == 6 ) {
              g5D_coarse_solve_odd_even_PRECISION( &(l->next_level->p_PRECISION), &(l->next_level->oe_op_PRECISION), l->next_level, threading );
            } else {

              START_MASTER(threading)
              g.coarsest_time -= MPI_Wtime();
              END_MASTER(threading)

#ifdef GCRODR
              // NOTE : something that shouldn't be happening here happens, namely the RHS is changed
              //        by the function coarse_solve_odd_even_PRECISION(...). So, we back it up and restore
              //        it as necessary

              int start,end;
              compute_core_start_end( l->next_level->p_PRECISION.v_start, l->next_level->p_PRECISION.v_end, &start, &end, l->next_level, threading );
              vector_PRECISION_copy( l->next_level->p_PRECISION.rhs_bk, l->next_level->p_PRECISION.b, start, end, l->next_level );

              START_MASTER(threading)
              l->next_level->p_PRECISION.was_there_stagnation = 0;
              END_MASTER(threading)
              SYNC_MASTER_TO_ALL(threading)

              while( 1 ) {
                coarse_solve_odd_even_PRECISION( &(l->next_level->p_PRECISION), &(l->next_level->oe_op_PRECISION), l->next_level, threading );
                if ( l->next_level->p_PRECISION.was_there_stagnation==0 ) { break; }
                else if ( l->next_level->p_PRECISION.was_there_stagnation==1 && l->next_level->p_PRECISION.gcrodr_PRECISION.CU_usable==1 ) {
                  // in case there was stagnation, we need to rebuild the coarsest-level data
                  double time_bk = g.coarsest_time;
                  coarsest_level_resets_PRECISION( l->next_level, threading );
                  START_MASTER(threading)
                  l->next_level->p_PRECISION.was_there_stagnation = 0;
                  g.coarsest_time = time_bk;
                  END_MASTER(threading)
                  SYNC_MASTER_TO_ALL(threading)
                  vector_PRECISION_copy( l->next_level->p_PRECISION.b, l->next_level->p_PRECISION.rhs_bk, start, end, l->next_level );
                }
                else {
                  // in this case, there was stagnation but no deflation/recycling subspace is being used
                  break;
                }
              }
#else
              coarse_solve_odd_even_PRECISION( &(l->next_level->p_PRECISION), &(l->next_level->oe_op_PRECISION), l->next_level, threading );
#endif

              START_MASTER(threading)
              g.coarsest_time += MPI_Wtime();
              END_MASTER(threading)

              //fgmres_PRECISION( &(l->next_level->p_PRECISION), l->next_level, threading );
            }
          } else {
            fgmres_PRECISION( &(l->next_level->p_PRECISION), l->next_level, threading );
          }
        }
        START_MASTER(threading)
        if ( l->depth == 0 )
          g.coarse_time += MPI_Wtime();
        END_MASTER(threading)
      }

      if( i == 0 && res == _NO_RES )
        interpolate3_PRECISION( phi, l->next_level->p_PRECISION.x, l, threading );
      else
        interpolate_PRECISION( phi, l->next_level->p_PRECISION.x, l, threading );

      smoother_PRECISION( phi, Dphi, eta, l->post_smooth_iter, _RES, _NO_SHIFT, l, threading );

      res = _RES;
    }
  } else {
    smoother_PRECISION( phi, Dphi, eta, (l->depth==0)?l->n_cy:l->post_smooth_iter, res, _NO_SHIFT, l, threading );
  }
}
