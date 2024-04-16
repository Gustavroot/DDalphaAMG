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

void data_layout_init( level_struct *l ) {
  
  int i, j;
  
  l->num_inner_lattice_sites = 1;
  for ( i=0; i<4; i++ )
    l->num_inner_lattice_sites *= l->local_lattice[i];  
  l->num_lattice_sites = l->num_inner_lattice_sites;
  l->inner_vector_size = l->num_inner_lattice_sites * l->num_lattice_site_var;
  
  j = l->num_lattice_sites;
  for ( i=0; i<4; i++ ) 
    l->num_lattice_sites += j / l->local_lattice[i];
  
  l->vector_size = l->num_lattice_sites * l->num_lattice_site_var;
  l->schwarz_vector_size = 2*l->vector_size - l->inner_vector_size;
}


void define_eot( int *eot, int *N, level_struct *l ) {
  
  int i, mu, t, z, y, x, ls[4], le[4], oe_offset=0;
      
  for ( mu=0; mu<4; mu++ )
    oe_offset += (l->local_lattice[mu]*(g.my_coords[mu]/l->comm_offset[mu]))%2;
  oe_offset = oe_offset%2;
  
  for ( mu=0; mu<4; mu++ ) {
    ls[mu] = 0;
    le[mu] = ls[mu] + l->local_lattice[mu];
  }
  
  i = 0;
  for ( t=0; t<le[T]; t++ )
    for ( z=0; z<le[Z]; z++ )
      for ( y=0; y<le[Y]; y++ )
        for ( x=0; x<le[X]; x++ )
          if ( (t+z+y+x+oe_offset)%2 == 0 ) {
            eot[ lex_index( t, z, y, x, N ) ] = i;
            i++;
          }
          
  for ( t=0; t<le[T]; t++ )
    for ( z=0; z<le[Z]; z++ )
      for ( y=0; y<le[Y]; y++ )
        for ( x=0; x<le[X]; x++ )
          if ( (t+z+y+x+oe_offset)%2 == 1 ) {
            eot[ lex_index( t, z, y, x, N ) ] = i;
            i++;
          }
                  
  for ( mu=0; mu<4; mu++ ) {
    ls[mu] = le[mu];
    le[mu]++;
    
    for ( t=ls[T]; t<le[T]; t++ )
      for ( z=ls[Z]; z<le[Z]; z++ )
        for ( y=ls[Y]; y<le[Y]; y++ )
          for ( x=ls[X]; x<le[X]; x++ )
            if ( (t+z+y+x+oe_offset)%2 == 0 ) {
              eot[ lex_index( t, z, y, x, N ) ] = i;
              i++;
            }
            
    for ( t=ls[T]; t<le[T]; t++ )
      for ( z=ls[Z]; z<le[Z]; z++ )
        for ( y=ls[Y]; y<le[Y]; y++ )
          for ( x=ls[X]; x<le[X]; x++ )
            if ( (t+z+y+x+oe_offset)%2 == 1 ) {
              eot[ lex_index( t, z, y, x, N ) ] = i;
              i++;
            }
    
    ls[mu] = 0;
    le[mu]--;    
  }
}


void define_eo_bt( int **bt, int *eot, int *n_ebs, int *n_obs, int *n_bs, int *N, level_struct *l ) {
  
  int i, t, z, y, x, mu, nu, le[4], bs, oe_offset=0, *bt_mu;
  
  for ( mu=0; mu<4; mu++ ) {
    le[mu] = l->local_lattice[mu];
  }
  
  for ( mu=0; mu<4; mu++ )
    oe_offset += (l->local_lattice[mu]*(g.my_coords[mu]/l->comm_offset[mu]))%2;
  oe_offset = oe_offset%2;
  
  for ( mu=0; mu<4; mu++ ) {
    bt_mu = bt[2*mu];
    bs = 1;
    le[mu] = 1;
    for ( nu=0; nu<4; nu++ )
      bs *= le[nu];
     
    i = 0;
    for ( t=0; t<le[T]; t++ )
      for ( z=0; z<le[Z]; z++ )
        for ( y=0; y<le[Y]; y++ )
          for ( x=0; x<le[X]; x++ )
            if ( (t+z+y+x+oe_offset)%2 == 0 ) {
              bt_mu[i] = site_index( t, z, y, x, N, eot );
              i++;
            }
    n_ebs[2*mu] = i;
    n_ebs[2*mu+1] = i;
    
    for ( t=0; t<le[T]; t++ )
      for ( z=0; z<le[Z]; z++ )
        for ( y=0; y<le[Y]; y++ )
          for ( x=0; x<le[X]; x++ )
            if ( (t+z+y+x+oe_offset)%2 == 1 ) {
              bt_mu[i] = site_index( t, z, y, x, N, eot );
              i++;
            }
            
    n_obs[2*mu] = i - n_ebs[2*mu];
    n_obs[2*mu+1] = i - n_ebs[2*mu+1];
    n_bs[2*mu] = i;
    n_bs[2*mu+1] = i;
    le[mu] = l->local_lattice[mu];
  }
}
