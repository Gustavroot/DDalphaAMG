#include "data_layout_PRECISION.h"
#include "data_layout.h"
#include "main.h"

void define_nt_bt_tt_PRECISION_cpu(operator_PRECISION_struct *op, int **bt, int *dt, level_struct *l ) {
  int *nt = op->neighbor_table,
      *backward_nt = op->backward_neighbor_table,
      *tt = op->translation_table,
      *it = op->index_table;
  ASSERT( dt != NULL && it != NULL );
  
  int i, mu, pos, t, z, y, x, ls[4], le[4], l_st[4], l_en[4],
      offset, stride, *bt_mu, *gs = l->global_splitting;
  
  for ( mu=0; mu<4; mu++ ) {
    ls[mu] = 0;
    le[mu] = ls[mu] + l->local_lattice[mu];
    l_st[mu] = ls[mu];
    l_en[mu] = le[mu];
  }
  
  // define neighbor table
  stride = (l->depth==0)?4:5; offset = (l->depth==0)?0:1;
  for ( t=ls[T]; t<le[T]; t++ )
    for ( z=ls[Z]; z<le[Z]; z++ )
      for ( y=ls[Y]; y<le[Y]; y++ )
        for ( x=ls[X]; x<le[X]; x++ ) {
          pos = site_index( t, z, y, x, dt, it );
          if ( offset )
            nt[stride*pos  ] = pos;
          nt[stride*pos+offset+T] = site_index( (gs[T]>1)?t+1:(t+1)%le[T], z, y, x, dt, it ); // T dir
          nt[stride*pos+offset+Z] = site_index( t, (gs[Z]>1)?z+1:(z+1)%le[Z], y, x, dt, it ); // Z dir
          nt[stride*pos+offset+Y] = site_index( t, z, (gs[Y]>1)?y+1:(y+1)%le[Y], x, dt, it ); // Y dir
          nt[stride*pos+offset+X] = site_index( t, z, y, (gs[X]>1)?x+1:(x+1)%le[X], dt, it ); // X dir
        }

  // define backward neighbor table
  stride = (l->depth==0)?4:5; offset = (l->depth==0)?0:1;
  for ( t=ls[T]; t<le[T]; t++ )
    for ( z=ls[Z]; z<le[Z]; z++ )
      for ( y=ls[Y]; y<le[Y]; y++ )
        for ( x=ls[X]; x<le[X]; x++ ) {
          pos = site_index( t, z, y, x, dt, it );
          if ( offset )
            backward_nt[stride*pos  ] = pos;
          backward_nt[stride*pos+offset+T] = site_index( (gs[T]>1)?(t-1+dt[T])%dt[T]:(t-1+le[T])%le[T], z, y, x, dt, it ); // T dir
          backward_nt[stride*pos+offset+Z] = site_index( t, (gs[Z]>1)?(z-1+dt[Z])%dt[Z]:(z-1+le[Z])%le[Z], y, x, dt, it ); // Z dir
          backward_nt[stride*pos+offset+Y] = site_index( t, z, (gs[Y]>1)?(y-1+dt[Y])%dt[Y]:(y-1+le[Y])%le[Y], x, dt, it ); // Y dir
          backward_nt[stride*pos+offset+X] = site_index( t, z, y, (gs[X]>1)?(x-1+dt[X])%dt[X]:(x-1+le[X])%le[X], dt, it ); // X dir
        }
        
  if ( bt != NULL ) {
    for ( mu=0; mu<4; mu++ ) {
      // define negative boundary table for communication
      l_en[mu] = l_st[mu]+1;
      bt_mu = bt[2*mu+1];
      i = 0;
      for ( t=l_st[T]; t<l_en[T]; t++ )
        for ( z=l_st[Z]; z<l_en[Z]; z++ )
          for ( y=l_st[Y]; y<l_en[Y]; y++ )
            for ( x=l_st[X]; x<l_en[X]; x++ ) {
              bt_mu[i] = site_index( t, z, y, x, dt, it );
              i++;
            }
      l_en[mu] = le[mu];
      
      // define positive boundary table for communication (if desired)
      if ( bt[2*mu] != bt[2*mu+1] ) {
        l_st[mu] = le[mu]-1;
        bt_mu = bt[2*mu];
        i = 0;
        for ( t=l_st[T]; t<l_en[T]; t++ )
          for ( z=l_st[Z]; z<l_en[Z]; z++ )
            for ( y=l_st[Y]; y<l_en[Y]; y++ )
              for ( x=l_st[X]; x<l_en[X]; x++ ) {
                bt_mu[i] = site_index( t, z, y, x, dt, it );
                i++;
              }

        l_st[mu] = ls[mu];
      }
    }
  }
  
  // define layout translation table
  // for translation to lexicographical site ordering
  if ( tt != NULL ) {
    i = 0;
    for ( t=0; t<l->local_lattice[T]; t++ )
      for ( z=0; z<l->local_lattice[Z]; z++ )
        for ( y=0; y<l->local_lattice[Y]; y++ )
          for ( x=0; x<l->local_lattice[X]; x++ ) {
            tt[i] = site_index( t, z, y, x, dt, it );
            i++;
          }
  }
}