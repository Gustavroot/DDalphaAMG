/*
 * Copyright (C) 2016, Matthias Rottmann, Artur Strebel, Simon Heybrock, Simone Bacchio, Bjoern Leder, Issaku Kanamori.
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
    if( g.doing_setup==1 ){
      //schwarz_PRECISION( phi, Dphi, eta, n, res, &(l->s_PRECISION), l, threading );
      schwarz_PRECISION_CUDA( phi, Dphi, eta, n, res, &(l->s_PRECISION), l, threading );
    }
    else{
      // Enabled for now on the finest grid only
      if( l->depth==0 && g.odd_even ){
        //if( g.my_rank==0 && l->depth==0 ){ printf( "Applying CUDA smoother for solving, at depth=%d\n", l->depth ); }
        schwarz_PRECISION_CUDA( phi, Dphi, eta, n, res, &(l->s_PRECISION), l, threading );
        //schwarz_PRECISION( phi, Dphi, eta, n, res, &(l->s_PRECISION), l, threading );
        //additive_schwarz_PRECISION( phi, Dphi, eta, n, res, &(l->s_PRECISION), l, threading );
      }
      else{
        schwarz_PRECISION( phi, Dphi, eta, n, res, &(l->s_PRECISION), l, threading );
        //schwarz_PRECISION_CUDA( phi, Dphi, eta, n, res, &(l->s_PRECISION), l, threading );
      }
    }

    /*

      struct timeval start, end;
      long start_us, end_us;
      long speedp, dt_gpu, dt_cpu;

      gettimeofday(&start, NULL);

      //red_black_schwarz_PRECISION( phi, Dphi, eta, n, res, &(l->s_PRECISION), l, threading );
      schwarz_PRECISION( phi, Dphi, eta, n, res, &(l->s_PRECISION), l, threading );

      gettimeofday(&end, NULL);

      start_us = start.tv_sec * (int)1e6 + start.tv_usec;
      end_us = end.tv_sec * (int)1e6 + end.tv_usec;

      dt_cpu = (end_us-start_us);

      printf("\n(proc=%d)Time (in us) for computing one call (& level = %d) of schwarz_PRECISION(...) (according to gettimeofday): %ld\n",
             g.my_rank, l->level, dt_cpu);

      // Emulating the amount of time it would take to transfer spinors

      vector_PRECISION r = (l->s_PRECISION).buf1;
      vector_PRECISION x = (l->s_PRECISION).buf3;
      vector_PRECISION latest_iter = (l->s_PRECISION).buf2;

      cuda_vector_PRECISION r_dev = ((l->s_PRECISION).cu_s).buf1,
                            x_dev = ((l->s_PRECISION).cu_s).buf3,
                            latest_iter_dev = ((l->s_PRECISION).cu_s).buf2;

      cudaStream_t *streams_schwarz = (l->s_PRECISION).streams;

      if( g.my_rank==0 ){

      gettimeofday(&start, NULL);

      cuda_vector_PRECISION_copy((void*)x_dev, (void*)x, 0, (l->s_PRECISION).num_blocks*(l->s_PRECISION).block_vector_size, l, _H2D, _CUDA_SYNC, 0, streams_schwarz );
      cuda_vector_PRECISION_copy((void*)r_dev, (void*)r, 0, (l->s_PRECISION).num_blocks*(l->s_PRECISION).block_vector_size, l, _H2D, _CUDA_SYNC, 0, streams_schwarz );
      //cuda_vector_PRECISION_copy((void*)latest_iter_dev, (void*)latest_iter, 0, s->num_blocks*s->block_vector_size, l, _H2D, _CUDA_ASYNC, 0, streams_schwarz );

      //cuda_safe_call( cudaDeviceSynchronize() );

      cuda_vector_PRECISION_copy((void*)x, (void*)x_dev, 0, (l->s_PRECISION).num_blocks*(l->s_PRECISION).block_vector_size, l, _D2H, _CUDA_SYNC, 0, streams_schwarz );
      cuda_vector_PRECISION_copy((void*)r, (void*)r_dev, 0, (l->s_PRECISION).num_blocks*(l->s_PRECISION).block_vector_size, l, _D2H, _CUDA_SYNC, 0, streams_schwarz );
      cuda_vector_PRECISION_copy((void*)latest_iter, (void*)latest_iter_dev, 0, (l->s_PRECISION).num_blocks*(l->s_PRECISION).block_vector_size, l, _D2H, _CUDA_SYNC, 0, streams_schwarz );

      gettimeofday(&end, NULL);

      start_us = start.tv_sec * (int)1e6 + start.tv_usec;
      end_us = end.tv_sec * (int)1e6 + end.tv_usec;

      dt_gpu = (end_us-start_us);

      printf("\n(proc=%d)Time (in us) for transferring data (& level = %d) to GPU, associated to schwarz_PRECISION(...) (according to gettimeofday): %ld\n",
             g.my_rank, l->level, dt_gpu);

      speedp = dt_cpu/dt_gpu;

      printf("\n(proc=%d)Attainable speedup for schwarz_PRECISION(...) (according to gettimeofday): %ld\n",
             g.my_rank, speedp);

      }

      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Finalize();
      exit(0);

    */

#else
    //red_black_schwarz_PRECISION( phi, Dphi, eta, n, res, &(l->s_PRECISION), l, threading );
    schwarz_PRECISION( phi, Dphi, eta, n, res, &(l->s_PRECISION), l, threading );
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
          apply_operator_PRECISION( l->sp_PRECISION.x, phi, &(l->p_PRECISION), l, threading );
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
          if ( l->depth == 0 ) solve_oddeven_PRECISION( &(l->sp_PRECISION), &(l->oe_op_PRECISION), l, threading );
          else coarse_solve_odd_even_PRECISION( &(l->sp_PRECISION), &(l->oe_op_PRECISION), l, threading );
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
        fgmres_PRECISION( &(l->sp_PRECISION), l, threading );
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

  if ( g.interpolation && l->level>0 ) {
    for ( int i=0; i<l->n_cy; i++ ) {
      if ( i==0 && res == _NO_RES ) {
        restrict_PRECISION( l->next_level->p_PRECISION.b, eta, l, threading );
      } else {
        int start = threading->start_index[l->depth];
        int end   = threading->end_index[l->depth];
        apply_operator_PRECISION( l->vbuf_PRECISION[2], phi, &(l->p_PRECISION), l, threading );
        vector_PRECISION_minus( l->vbuf_PRECISION[3], eta, l->vbuf_PRECISION[2], start, end, l );
        restrict_PRECISION( l->next_level->p_PRECISION.b, l->vbuf_PRECISION[3], l, threading );
      }
      if ( !l->next_level->idle ) {
        START_MASTER(threading)
        if ( l->depth == 0 )
          g.coarse_time -= MPI_Wtime();
        END_MASTER(threading)
        if ( l->level > 1 ) {
          if ( g.kcycle )
            fgmres_PRECISION( &(l->next_level->p_PRECISION), l->next_level, threading );
          else
            vcycle_PRECISION( l->next_level->p_PRECISION.x, NULL, l->next_level->p_PRECISION.b, _NO_RES, l->next_level, threading );
        } else {
          if ( g.odd_even ) {
            if ( g.method == 6 ) {
              g5D_coarse_solve_odd_even_PRECISION( &(l->next_level->p_PRECISION), &(l->next_level->oe_op_PRECISION), l->next_level, threading );
            } else {
              coarse_solve_odd_even_PRECISION( &(l->next_level->p_PRECISION), &(l->next_level->oe_op_PRECISION), l->next_level, threading );
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
