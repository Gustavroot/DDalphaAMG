#include "main.h"

#ifdef CUDA_OPT

void field_saver( void* phi, int length, char* datatype, char* filename ){
  int i;
  FILE *f;

  f = fopen(filename, "w");
  if(f == NULL){
    printf("Error opening file!\n");
    return;
  }

  for(i=0; i<length; i++){

    if(strcmp(datatype, "float")){
      float buf = ((float*)phi)[i];
      fprintf(f, "[%d]-th entry = %f+i%f\n", i, creal(buf), cimag(buf));
    }
    else{
      double buf = ((double*)phi)[i];
      fprintf(f, "[%d]-th entry = %lf+i%lf\n", i, creal(buf), cimag(buf));
    }

  }

  fclose(f);
}
#endif


void coarsest_level_resets( level_struct* l, struct Thread* threading ) {

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
        if ( g.mixed_precision==0 ) {
          lx->p_double.polyprec_double.update_lejas = 1;
          lx->p_double.polyprec_double.preconditioner = NULL;
        }
        else {
          lx->p_float.polyprec_float.update_lejas = 1;
          lx->p_float.polyprec_float.preconditioner = NULL;
        }
        break;
      }
      else { lx = lx->next_level; }
    }

    END_MASTER(threading)

    SYNC_MASTER_TO_ALL(threading)
    SYNC_CORES(threading)

  }
#endif

//#ifdef BLOCK_JACOBI
#if 0
  {
    // setting flag to re-update lejas
    level_struct *lx = l;
    while (1) {
      if ( lx->level==0 ) {
        if ( g.mixed_precision==0 ) {
          lx->p_double.block_jacobi_double.local_p.polyprec_double.update_lejas = 1;
          lx->p_double.block_jacobi_double.BJ_usable = 0;
        }
        else {
          lx->p_float.block_jacobi_float.local_p.polyprec_float.update_lejas = 1;
          lx->p_float.block_jacobi_float.BJ_usable = 0;
        }
        break;
      }
      else { lx = lx->next_level; }
    }
  }
#endif

#ifdef GCRODR
  START_MASTER(threading)
  {
    // setting flag to re-update recycling subspace
    level_struct *lx = l;
    while (1) {
      if ( lx->level==0 ) {
        if ( g.mixed_precision==0 ) {
          //lx->p_double.gcrodr_double.CU_usable = 0;
          lx->p_double.gcrodr_double.update_CU = 1;
          lx->p_double.gcrodr_double.upd_ctr = 0;
        }
        else {
          //lx->p_float.gcrodr_float.CU_usable = 0;
          lx->p_float.gcrodr_float.update_CU = 1;
          lx->p_float.gcrodr_float.upd_ctr = 0;
        }
        break;
      }
      else { lx = lx->next_level; }
    }
  }
  END_MASTER(threading)
  SYNC_MASTER_TO_ALL(threading)
#endif

      // calling the coarsest-level solver once on setup
//#if defined(GCRODR) || defined(POLYPREC) || defined(BLOCK_JACOBI)
#if defined(GCRODR)
//#if 0
      {
        level_struct *lx = l;

        START_MASTER(threading)
        printf0( "\nPre-constructing coarsest-level data ...\n" );
        END_MASTER(threading)

        while (1) {
          if ( lx->level==0 ) {

            if ( !(lx->idle) ) {

            if ( g.mixed_precision==0 ) {

              gmres_double_struct* px = &(lx->p_double);

              //// set RHS to random
              //START_MASTER(threading)
              //vector_double_define_random( px->b, px->v_start, px->v_end, lx );
              //END_MASTER(threading)
              //SYNC_MASTER_TO_ALL(threading)

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
              while ( px->gcrodr_double.CU_usable==0 ) {
                // set RHS to random
                START_MASTER(threading)
                vector_double_define_random( px->b, px->v_start, px->v_end, lx );
                END_MASTER(threading)
                SYNC_MASTER_TO_ALL(threading)

                coarse_solve_odd_even_double( px, &(lx->oe_op_double), lx, threading );
                try_ctr++;
                if ( try_ctr>=5 ) {
                  printf0( "Tried 5 times to construct a recycling/deflation subspace, failed\n" );
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

            }
            else {

              gmres_float_struct* px = &(lx->p_float);

              //// set RHS to random
              //START_MASTER(threading)
              //vector_float_define_random( px->b, px->v_start, px->v_end, lx );
              //END_MASTER(threading)
              //SYNC_MASTER_TO_ALL(threading)

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
              while ( px->gcrodr_float.CU_usable==0 ) {
                // set RHS to random
                START_MASTER(threading)
                vector_float_define_random( px->b, px->v_start, px->v_end, lx );
                END_MASTER(threading)
                SYNC_MASTER_TO_ALL(threading)

                coarse_solve_odd_even_float( px, &(lx->oe_op_float), lx, threading );
                try_ctr++;
                if ( try_ctr>=5 ) {
                  printf0( "Tried 5 times to construct a recycling/deflation subspace, failed\n" );
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

            }

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


void set_some_coarsest_level_improvs_params_for_setup( level_struct* l, struct Thread* threading ) {

#if defined(POLYPREC) || defined(GCRODR)
    START_MASTER(threading)
    {
      level_struct *lx = l;
      while (1) {
        if ( lx->level==0 ) {
          if ( g.mixed_precision==0 ) {
#ifdef GCRODR
            lx->p_double.gcrodr_double.k = g.gcrodr_k_setup;
#endif
#ifdef POLYPREC
            lx->p_float.polyprec_float.d_poly = g.polyprec_d_setup;
#endif
          }
          else {
#ifdef GCRODR
            lx->p_float.gcrodr_float.k = g.gcrodr_k_setup;
#endif
#ifdef POLYPREC
            lx->p_float.polyprec_float.d_poly = g.polyprec_d_setup;
#endif
          }
          break;
        }
        else { lx = lx->next_level; }
      }
    }
    END_MASTER(threading)
    SYNC_MASTER_TO_ALL(threading)
#endif

}


void set_some_coarsest_level_improvs_params_for_solve( level_struct* l, struct Thread* threading ) {

#if defined(POLYPREC) || defined(GCRODR)
    START_MASTER(threading)
    {
      level_struct *lx = l;
      while (1) {
        if ( lx->level==0 ) {
          if ( g.mixed_precision==0 ) {
#ifdef POLYPREC
            lx->p_double.polyprec_double.d_poly = g.polyprec_d_solve;
#endif
#ifdef GCRODR
            lx->p_double.gcrodr_double.k = g.gcrodr_k_solve;
#endif
          }
          else {
#ifdef POLYPREC
            lx->p_float.polyprec_float.d_poly = g.polyprec_d_solve;
#endif
#ifdef GCRODR
            lx->p_float.gcrodr_float.k = g.gcrodr_k_solve;
#endif
          }
          break;
        }
        else { lx = lx->next_level; }
      }
    }
    END_MASTER(threading)
    SYNC_MASTER_TO_ALL(threading)
#endif

}
