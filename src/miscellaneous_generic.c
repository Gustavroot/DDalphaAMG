#include "main.h"


void coarsest_level_resets_PRECISION( level_struct* l, struct Thread* threading ) {

  START_MASTER(threading)
  g.coarsest_time = 0.0;
  END_MASTER(threading)

#ifdef POLYPREC
  {

    SYNC_MASTER_TO_ALL(threading)
    SYNC_CORES(threading)

    START_MASTER(threading)

    // setting flag to re-update lejas
    level_struct *lx = l;
    while (1) {
      if ( lx->level==0 ) {
        lx->p_PRECISION.polyprec_PRECISION.update_lejas = 1;
        lx->p_PRECISION.polyprec_PRECISION.preconditioner = NULL;
        break;
      }
      else { lx = lx->next_level; }
    }

    END_MASTER(threading)

    SYNC_MASTER_TO_ALL(threading)
    SYNC_CORES(threading)

  }
#endif

#ifdef GCRODR
  START_MASTER(threading)
  {
    // setting flag to re-update recycling subspace
    level_struct *lx = l;
    while (1) {
      if ( lx->level==0 ) {
        lx->p_PRECISION.gcrodr_PRECISION.CU_usable = 0;
        lx->p_PRECISION.gcrodr_PRECISION.update_CU = 1;
        lx->p_PRECISION.gcrodr_PRECISION.upd_ctr = 0;
        break;
      }
      else { lx = lx->next_level; }
    }
  }
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)
#endif

  // calling the coarsest-level solver once on setup
#if defined(GCRODR)
      {
        level_struct *lx = l;

        START_MASTER(threading)
        printf0( "\nPre-constructing coarsest-level data ...\n" );
        END_MASTER(threading)

        while (1) {
          if ( lx->level==0 ) {

            if ( !(lx->idle) ) {

              gmres_PRECISION_struct* px = &(lx->p_PRECISION);

              START_MASTER(threading)
              g.gcrodr_calling_from_setup = 1;
              END_MASTER(threading)
              SYNC_MASTER_TO_ALL(threading)

              double buff1x = px->tol;
              double buff2x = g.coarse_tol;
              START_MASTER(threading)
              px->tol = 1.0e-20;
              g.coarse_tol = 1.0e-20;
              END_MASTER(threading)
              SYNC_MASTER_TO_ALL(threading)
              // call the coarsest-level solver
              int try_ctr = 0;
              while ( px->gcrodr_PRECISION.CU_usable==0 ) {
                // set RHS to random
                START_MASTER(threading)
                vector_PRECISION_define_random( px->b, px->v_start, px->v_end, lx );
                END_MASTER(threading)
                SYNC_MASTER_TO_ALL(threading)

                coarse_solve_odd_even_PRECISION( px, &(lx->oe_op_PRECISION), lx, threading );
                try_ctr++;
                if ( try_ctr>=2 && px->gcrodr_PRECISION.CU_usable==0 ) {
                  START_MASTER(threading)
                  printf0( "Tried 2 times to construct a recycling/deflation subspace, failed\n" );
                  END_MASTER(threading)
                  break;
                }
              }
              START_MASTER(threading)
              px->tol = buff1x;
              g.coarse_tol = buff2x;
              END_MASTER(threading)
              SYNC_MASTER_TO_ALL(threading)

              START_MASTER(threading)
              g.gcrodr_calling_from_setup = 0;
              END_MASTER(threading)
              SYNC_MASTER_TO_ALL(threading)

            } // end of !idle if

            break;
          }
          else { lx = lx->next_level; }
        }

        START_MASTER(threading)
        printf0( "... done\n\n" );
        END_MASTER(threading)

      }
#endif

  START_MASTER(threading)
  g.avg_b1 = 0.0;
  g.avg_b2 = 0.0;
  g.avg_crst = 0.0;
  END_MASTER(threading)

}
