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

void smoother_PRECISION_def( level_struct *l ) {
  
  if ( g.method >= 0 )
    schwarz_PRECISION_def( &(l->s_PRECISION), &(g.op_double), l );
  
  l->p_PRECISION.op = &(l->s_PRECISION.op);
  if ( g.method == 6 ) {
    l->p_PRECISION.eval_operator = (l->depth > 0)?g5D_apply_coarse_operator_PRECISION:g5D_plus_clover_PRECISION;
  } else {
    l->p_PRECISION.eval_operator = (l->depth > 0)?apply_coarse_operator_PRECISION:d_plus_clover_PRECISION;
  }
}


// TODO
#ifdef CUDA_OPT
void smoother_PRECISION_def_CUDA( level_struct *l ) { }
#endif


void smoother_PRECISION_free( level_struct *l ) {
  
  if ( g.method >= 0 )
    schwarz_PRECISION_free( &(l->s_PRECISION), l );
#ifdef CUDA_OPT
  if( l->depth==0 && g.odd_even ){
    schwarz_PRECISION_free_CUDA( &(l->s_PRECISION), l );
  }
#endif
}


// TODO
#ifdef CUDA_OPT
void smoother_PRECISION_free_CUDA( level_struct *l ) { }
#endif


void schwarz_PRECISION_init( schwarz_PRECISION_struct *s, level_struct *l ) {

  operator_PRECISION_init( &(s->op) );
  
  s->index[T] = NULL;
  s->oe_index[T] = NULL;
  s->block = NULL;
  s->bbuf1 = NULL;
  s->buf1 = NULL;
  s->buf2 = NULL;
  s->buf3 = NULL;
  s->buf4 = NULL;
  s->buf5 = NULL;
  l->sbuf_PRECISION[0] = NULL;
  s->oe_bbuf[0] = NULL;
  s->oe_bbuf[1] = NULL;
  s->oe_buf[0] = NULL;
  s->oe_buf[1] = NULL;
  s->oe_buf[2] = NULL;
  s->oe_buf[3] = NULL;
  s->local_minres_buffer[0] = NULL;
  s->local_minres_buffer[1] = NULL;
  s->local_minres_buffer[2] = NULL;
  s->block_list = NULL;
  s->block_list_length = NULL;
  s->num_colors = 0;
}


// TODO
#ifdef CUDA_OPT
void schwarz_PRECISION_init_CUDA( schwarz_PRECISION_struct *s, level_struct *l ) { }
#endif


void schwarz_PRECISION_alloc( schwarz_PRECISION_struct *s, level_struct *l ) {
  
  int i, j, n, mu, nu, *bl = l->block_lattice, vs = (l->depth==0)?l->inner_vector_size:l->vector_size;
  
  if ( g.method == 4 ) {
    fgmres_PRECISION_struct_alloc( l->block_iter, 1, (l->depth==0)?l->inner_vector_size:l->vector_size,
                                   EPS_PRECISION, _COARSE_GMRES, _NOTHING, NULL,
                                   (l->depth==0)?(g.odd_even?apply_schur_complement_PRECISION:d_plus_clover_PRECISION):
                                   (g.odd_even?coarse_apply_schur_complement_PRECISION:apply_coarse_operator_PRECISION),
                                   &(l->sp_PRECISION), l );
  } else if ( g.method == 5 ) {
    fgmres_PRECISION_struct_alloc( 5, 1, (l->depth==0)?l->inner_vector_size:l->vector_size,
                                   EPS_PRECISION, _COARSE_GMRES, _NOTHING, NULL,
                                   (l->depth==0)?(g.odd_even?apply_schur_complement_PRECISION:d_plus_clover_PRECISION):
                                   (g.odd_even?coarse_apply_schur_complement_PRECISION:apply_coarse_operator_PRECISION),
                                   &(l->sp_PRECISION), l );
  } else if ( g.method == 6 ) {
    fgmres_PRECISION_struct_alloc( l->block_iter, 1, (l->depth==0)?l->inner_vector_size:l->vector_size,
                                   EPS_PRECISION, _COARSE_GMRES, _NOTHING, NULL,
                                   (l->depth==0)?(g.odd_even?g5D_apply_schur_complement_PRECISION:g5D_plus_clover_PRECISION):
                                   (g.odd_even?g5D_coarse_apply_schur_complement_PRECISION:g5D_apply_coarse_operator_PRECISION),
                                   &(l->sp_PRECISION), l );
  }
  
  operator_PRECISION_alloc( &(s->op), _SCHWARZ, l );
  if ( l->level > 0 && l->depth > 0 ) l->p_PRECISION.op = &(s->op);
  
  s->dir_length[T] = (bl[T]-1)*bl[Z]*bl[Y]*bl[X];
  s->dir_length[Z] = bl[T]*(bl[Z]-1)*bl[Y]*bl[X];
  s->dir_length[Y] = bl[T]*bl[Z]*(bl[Y]-1)*bl[X];
  s->dir_length[X] = bl[T]*bl[Z]*bl[Y]*(bl[X]-1);
  
  MALLOC( s->index[T], int, MAX(1,s->dir_length[T]+s->dir_length[Z]+s->dir_length[Y]+s->dir_length[X]) );
  s->index[Z] = s->index[T]+s->dir_length[T];
  s->index[Y] = s->index[Z]+s->dir_length[Z];
  s->index[X] = s->index[Y]+s->dir_length[Y];
  
  if ( l->depth == 0 && g.odd_even ) {
    MALLOC( s->oe_index[T], int, MAX(1,s->dir_length[T]+s->dir_length[Z]+s->dir_length[Y]+s->dir_length[X]) );
    s->oe_index[Z] = s->oe_index[T]+s->dir_length[T];
    s->oe_index[Y] = s->oe_index[Z]+s->dir_length[Z];
    s->oe_index[X] = s->oe_index[Y]+s->dir_length[Y];
  }
  
  s->num_blocks = 1;
  s->num_block_sites = 1;
  for ( mu=0; mu<4; mu++ ) {
    s->num_block_sites *= bl[mu];
    s->num_blocks *= l->local_lattice[mu]/bl[mu];
  }
  s->block_vector_size = s->num_block_sites*l->num_lattice_site_var;
  
  if ( g.method == 3 ) {
    MALLOC( s->block_list, int*, 16 );
    s->block_list[0] = NULL;
    MALLOC( s->block_list[0], int, s->num_blocks );
    j = s->num_blocks/16;
    for ( i=1; i<16; i++ )
      s->block_list[i] = s->block_list[0]+i*j;
  } else if ( g.method == 2 ) {
    MALLOC( s->block_list_length, int, 8 );
    MALLOC( s->block_list, int*, 8 );
    for ( i=0; i<8; i++ ) {
      s->block_list[i] = NULL;
      MALLOC( s->block_list[i], int, s->num_blocks );
    }
  }
  
  MALLOC( s->block, block_struct, s->num_blocks );
  MALLOC( s->bbuf1, complex_PRECISION, (l->depth==0&&g.odd_even?9:3)*s->block_vector_size );
#ifndef EXTERNAL_DD
  if ( l->depth == 0 ) {
    MALLOC( s->oe_buf[0], complex_PRECISION, 4*l->inner_vector_size );
    s->oe_buf[1] = s->oe_buf[0] + l->inner_vector_size;
    s->oe_buf[2] = s->oe_buf[1] + l->inner_vector_size;
    s->oe_buf[3] = s->oe_buf[2] + l->inner_vector_size;
  }
#endif
  s->bbuf2 = s->bbuf1 + s->block_vector_size;
  s->bbuf3 = s->bbuf2 + s->block_vector_size;
  if ( l->depth == 0 && g.odd_even ) {
    s->oe_bbuf[0] = s->bbuf3 + s->block_vector_size;
    s->oe_bbuf[1] = s->oe_bbuf[0] + s->block_vector_size;
    s->oe_bbuf[2] = s->oe_bbuf[1] + s->block_vector_size;
    s->oe_bbuf[3] = s->oe_bbuf[2] + s->block_vector_size;
    s->oe_bbuf[4] = s->oe_bbuf[3] + s->block_vector_size;
    s->oe_bbuf[5] = s->oe_bbuf[4] + s->block_vector_size;
  }
  
  n = 0;
  for ( mu=0; mu<4; mu++ ) {
    i = 1;
    for ( nu=0; nu<4; nu++ ) {
      if ( mu != nu ) {
        i *= bl[nu];
      }
    }
    s->block_boundary_length[2*mu] = n;
    s->block_boundary_length[2*mu+1] = n+2*i;
    n += 4*i;
  }
  s->block_boundary_length[8] = n;
  
  for ( i=0; i<s->num_blocks; i++ ) {
    s->block[i].bt = NULL;
    MALLOC( s->block[i].bt, int, n );
  }

#ifdef CUDA_OPT
  MALLOC( s->buf1, complex_PRECISION, vs+3*l->schwarz_vector_size );
#else
  cuda_safe_call( cudaMallocHost( (void**)&(s->buf1), (vs+3*l->schwarz_vector_size)*sizeof(complex_PRECISION) ) );
#endif
  s->buf2 = s->buf1 + vs;
  s->buf3 = s->buf2 + l->schwarz_vector_size;
  s->buf4 = s->buf3 + l->schwarz_vector_size;
  
  if ( g.method == 1 )
    MALLOC( s->buf5, complex_PRECISION, l->schwarz_vector_size );
  
  MALLOC( l->sbuf_PRECISION[0], complex_PRECISION, 2*vs );
  l->sbuf_PRECISION[1] = l->sbuf_PRECISION[0] + vs;

#ifdef EXTERNAL_DD
  if ( l->depth > 0 )
#endif
  {
    // these buffers are introduced to make local_minres_PRECISION thread-safe
    MALLOC( s->local_minres_buffer[0], complex_PRECISION, l->schwarz_vector_size );
    MALLOC( s->local_minres_buffer[1], complex_PRECISION, l->schwarz_vector_size );
    MALLOC( s->local_minres_buffer[2], complex_PRECISION, l->schwarz_vector_size );
  }

#ifdef OPTIMIZED_NEIGHBOR_COUPLING_PRECISION
  if ( l->depth == 0 ) {
    MALLOC_HUGEPAGES( s->op.D_vectorized, PRECISION, 2*4*(2*l->vector_size-l->inner_vector_size), 4*SIMD_LENGTH_PRECISION );
    MALLOC_HUGEPAGES( s->op.D_transformed_vectorized, PRECISION, 2*4*(2*l->vector_size-l->inner_vector_size), 4*SIMD_LENGTH_PRECISION );
  }
#endif
#ifdef OPTIMIZED_SELF_COUPLING_PRECISION
  if ( l->depth == 0 ) {
    MALLOC_HUGEPAGES( s->op.clover_vectorized, PRECISION, 2*6*l->inner_vector_size, 4*SIMD_LENGTH_PRECISION );
  }
#endif

#ifdef CUDA_OPT
  // Streams for pipelining the computations in SAP
  //int nr_streams = ( s->num_blocks/96*96 )/30/2 + 2;
  // TODO: use safe malloc here
  s->streams = (cudaStream_t*) malloc( 1 * sizeof(cudaStream_t) );
  cuda_safe_call( cudaStreamCreate( &(s->streams[0]) ) );
#endif
}


// TODO
#ifdef CUDA_OPT
void schwarz_PRECISION_alloc_CUDA( schwarz_PRECISION_struct *s, level_struct *l ) {

  int vs, i;

  vs = (l->depth==0)?l->inner_vector_size:l->vector_size;

  // FIXME: what is the role of 'vs' here ?
  //MALLOC( s->buf1, complex_PRECISION, vs+3*l->schwarz_vector_size );
  // Buffers for the computations in SAP
  cuda_safe_call( cudaMalloc( (void**) (&( (s->cu_s).buf1 )), vs*sizeof(cu_cmplx_PRECISION) ) );
  cuda_safe_call( cudaMalloc( (void**) (&( (s->cu_s).buf2 )), l->schwarz_vector_size*sizeof(cu_cmplx_PRECISION) ) );
  cuda_safe_call( cudaMalloc( (void**) (&( (s->cu_s).buf3 )), l->schwarz_vector_size*sizeof(cu_cmplx_PRECISION) ) );
  cuda_safe_call( cudaMalloc( (void**) (&( (s->cu_s).buf4 )), l->schwarz_vector_size*sizeof(cu_cmplx_PRECISION) ) );

  if ( l->depth == 0 ) {
    // TODO: is l->inner_vector_size the ACTUAL size in this allocation ?
    for( i=0; i<4; i++ ){
      cuda_safe_call( cudaMalloc( (void**)&( (s->s_on_gpu_cpubuff).oe_buf[i] ), l->inner_vector_size*sizeof(cu_cmplx_PRECISION) ) );
    }
  }

  cuda_safe_call( cudaMalloc( (void**) (&( s->s_on_gpu )), 1*sizeof(schwarz_PRECISION_struct_on_gpu) ) );


  // TODO: do this at all MG levels ?
  if( l->depth==0 ){
    // these buffers are introduced to make local_minres_PRECISION thread-safe
    cuda_safe_call( cudaMalloc( (void**) (&( (s->cu_s).local_minres_buffer[0] )), l->schwarz_vector_size*sizeof(cu_cmplx_PRECISION) ) );
    cuda_safe_call( cudaMalloc( (void**) (&( (s->cu_s).local_minres_buffer[1] )), l->schwarz_vector_size*sizeof(cu_cmplx_PRECISION) ) );
    cuda_safe_call( cudaMalloc( (void**) (&( (s->cu_s).local_minres_buffer[2] )), l->schwarz_vector_size*sizeof(cu_cmplx_PRECISION) ) );

    //MALLOC( s->local_minres_buffer[0], complex_PRECISION, l->schwarz_vector_size );
    //MALLOC( s->local_minres_buffer[1], complex_PRECISION, l->schwarz_vector_size );
    //MALLOC( s->local_minres_buffer[2], complex_PRECISION, l->schwarz_vector_size );
  }

}
#endif


void schwarz_PRECISION_free( schwarz_PRECISION_struct *s, level_struct *l ) {
  
  int i, n, mu, nu, *bl = l->block_lattice, vs = (l->depth==0)?l->inner_vector_size:l->vector_size;
  
  if ( g.method == 4 || g.method == 5 || g.method == 6 )
    fgmres_PRECISION_struct_free( &(l->sp_PRECISION), l );
  
  FREE( s->index[T], int, MAX(1,s->dir_length[T]+s->dir_length[Z]+s->dir_length[Y]+s->dir_length[X]) );
  s->index[Z] = NULL;
  s->index[Y] = NULL;
  s->index[X] = NULL;
  
  if ( l->depth == 0 && g.odd_even ) {
    FREE( s->oe_index[T], int, MAX(1,s->dir_length[T]+s->dir_length[Z]+s->dir_length[Y]+s->dir_length[X]) );
    s->oe_index[Z] = NULL;
    s->oe_index[Y] = NULL;
    s->oe_index[X] = NULL;
  }
  
  n = 0;
  for ( mu=0; mu<4; mu++ ) {
    i = 1;
    for ( nu=0; nu<4; nu++ ) {
      if ( mu != nu ) {
        i *= bl[nu];
      }
    }
    n += 4*i;
  }
  
  for ( i=0; i<s->num_blocks; i++ ) {
    FREE( s->block[i].bt, int, n );
  }
  
  if ( g.method == 3 ) {
    FREE( s->block_list[0], int, s->num_blocks );
    FREE( s->block_list, int*, 16 );
  } else if ( g.method == 2 ) {
    FREE( s->block_list_length, int, 8 );
    for ( i=0; i<8; i++ )
      FREE( s->block_list[i], int, s->num_blocks );
    FREE( s->block_list, int*, 8 );
  }
  
  FREE( s->block, block_struct, s->num_blocks );
  FREE( s->bbuf1, complex_PRECISION, (l->depth==0&&g.odd_even?9:3)*s->block_vector_size );
#ifndef EXTERNAL_DD
  if ( l->depth == 0 ) {
    s->oe_buf[1] = NULL;
    s->oe_buf[2] = NULL;
    s->oe_buf[3] = NULL;
    FREE( s->oe_buf[0], complex_PRECISION, 4*l->inner_vector_size );
    s->oe_buf[0] = NULL;
  }
#endif
  s->bbuf2 = NULL; s->bbuf3 = NULL; s->oe_bbuf[0] = NULL; s->oe_bbuf[1] = NULL;
  s->oe_bbuf[2] = NULL; s->oe_bbuf[3] = NULL; s->oe_bbuf[4] = NULL; s->oe_bbuf[5] = NULL;

#ifdef CUDA_OPT  
  FREE( s->buf1, complex_PRECISION, vs+3*l->schwarz_vector_size );
#else
  cuda_safe_call( cudaFreeHost(s->buf1) );
#endif
  s->buf2 = NULL; s->buf3 = NULL;
  s->buf4 = NULL;
  
  if ( g.method == 1 )
    FREE( s->buf5, complex_PRECISION, l->schwarz_vector_size );
  
  operator_PRECISION_free( &(s->op), _SCHWARZ, l );
  
  FREE( l->sbuf_PRECISION[0], complex_PRECISION, 2*vs );
  l->sbuf_PRECISION[1] = NULL;

#ifdef EXTERNAL_DD
  if ( l->depth > 0 )
#endif
  {
    FREE( s->local_minres_buffer[0], complex_PRECISION, l->schwarz_vector_size );
    FREE( s->local_minres_buffer[1], complex_PRECISION, l->schwarz_vector_size );
    FREE( s->local_minres_buffer[2], complex_PRECISION, l->schwarz_vector_size );
    s->local_minres_buffer[0] = NULL;
    s->local_minres_buffer[1] = NULL;
    s->local_minres_buffer[2] = NULL;
  }
#ifdef OPTIMIZED_NEIGHBOR_COUPLING_PRECISION
  if ( l->depth == 0 ) {
    FREE_HUGEPAGES( s->op.D_vectorized, PRECISION, 2*4*(2*l->vector_size-l->inner_vector_size) );
    FREE_HUGEPAGES( s->op.D_transformed_vectorized, PRECISION, 2*4*(2*l->vector_size-l->inner_vector_size) );
  }
#endif
#ifdef OPTIMIZED_SELF_COUPLING_PRECISION
  if ( l->depth == 0 ) {
    FREE_HUGEPAGES( s->op.clover_vectorized, PRECISION, 2*6*l->inner_vector_size );
  }
#endif

#ifdef CUDA_OPT
  free( s->streams );
#endif
}


// TODO
#ifdef CUDA_OPT
void schwarz_PRECISION_free_CUDA( schwarz_PRECISION_struct *s, level_struct *l ) {

  int i;

  cuda_safe_call( cudaFree( (s->cu_s).buf1 ) );
  cuda_safe_call( cudaFree( (s->cu_s).buf2 ) );
  cuda_safe_call( cudaFree( (s->cu_s).buf3 ) );
  cuda_safe_call( cudaFree( (s->cu_s).buf4 ) );

  if ( l->depth == 0 ) {
    // TODO: is l->inner_vector_size the ACTUAL size in this allocation ?
    for( i=0; i<4; i++ ){
      cuda_safe_call( cudaFree( (s->s_on_gpu_cpubuff).oe_buf[i] ) );
    }
  }

  cuda_safe_call( cudaFree( s->s_on_gpu ) );
}
#endif


void schwarz_layout_PRECISION_define( schwarz_PRECISION_struct *s, level_struct *l ) {

  int a0, b0, c0, d0, a1, b1, c1, d1, block_split[4], block_size[4], agg_split[4], 
      i, j, k, mu, index, x, y, z, t, ls[4], le[4], l_st[4], l_en[4], *dt = s->op.table_dim,
      *dt_mod = s->op.table_mod_dim, *it = s->op.index_table, *count[4];

  // Define coloring    
  if ( g.method == 1 )        // Additive
    s->num_colors = 1;
  else if ( g.method == 2 )   // Red-Black
    s->num_colors = 2;
  else if ( g.method == 3 ) { // 16 Color
    int flag = 0;
    for ( mu=0; mu<4; mu++ ) {
      if ( (l->local_lattice[mu]/l->block_lattice[mu]) % 2 == 1 )
        flag = 1;
    }
    if ( flag == 0 )
      s->num_colors = 16;
    else {
      s->num_colors = 2;
      printf0("depth: %d, switching to red black schwarz as smoother\n", l->depth );
    }
  }
  
  const int sigma[16] = {0,1,3,2,6,4,5,7,15,14,12,13,9,11,10,8};
  const int color_to_comm[16][2] = { {T,-1}, {X,+1}, {Y,+1}, {X,-1}, {Z,+1}, {Y,-1}, {X,+1}, {Y,+1},
                               {T,+1}, {X,-1}, {Y,-1}, {X,+1}, {Z,-1}, {Y,+1}, {X,-1}, {Y,-1}  };
  int color_counter[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  
  s->num_block_sites = 1;
  s->block_oe_offset = 0;
  s->num_aggregates = 1;
  
  if ( g.method == 2 ) {
    for ( i=0; i<8; i++ )
      s->block_list_length[i]=0;
  }
  
  for ( mu=0; mu<4; mu++ ) {
    s->num_block_sites *= l->block_lattice[mu];
    s->block_oe_offset += ((l->local_lattice[mu]/l->block_lattice[mu])*(g.my_coords[mu]/l->comm_offset[mu]))%2;
    ls[mu] = 0;
    le[mu] = ls[mu] + l->local_lattice[mu];
    dt[mu] = l->local_lattice[mu]+2;
    dt_mod[mu] = l->local_lattice[mu]+2;
    l_st[mu] = ls[mu];
    l_en[mu] = le[mu];
    agg_split[mu] = l->local_lattice[mu]/l->coarsening[mu];
    block_split[mu] = l->coarsening[mu]/l->block_lattice[mu];
    block_size[mu] = l->block_lattice[mu];
    s->num_aggregates *= agg_split[mu];
  }
  s->block_oe_offset = s->block_oe_offset%2;
  s->block_vector_size = s->num_block_sites*l->num_lattice_site_var;

  i = 0; j = 0;
  // inner hyper cuboid
  count[T] = &d1; count[Z] = &c1; count[Y] = &b1; count[X] = &a1;
  for ( d0=0; d0<agg_split[T]; d0++ )
    for ( c0=0; c0<agg_split[Z]; c0++ )
      for ( b0=0; b0<agg_split[Y]; b0++ )
        for ( a0=0; a0<agg_split[X]; a0++ ) {
          
          for ( d1=d0*block_split[T]; d1<(d0+1)*block_split[T]; d1++ )
            for ( c1=c0*block_split[Z]; c1<(c0+1)*block_split[Z]; c1++ )
              for ( b1=b0*block_split[Y]; b1<(b0+1)*block_split[Y]; b1++ )
                for ( a1=a0*block_split[X]; a1<(a0+1)*block_split[X]; a1++ ) {
                  
                  s->block[j].start = i;
                  s->block[j].no_comm = 1;
                  if ( s->num_colors == 1 ) {
                    s->block[j].color = 0;
                  } else if ( s->num_colors == 2 ) {
                    s->block[j].color = ( d1+c1+b1+a1+s->block_oe_offset )%2;
                  } else if ( s->num_colors == 16 ) {
                    for ( k=0; k<16; k++ )
                      if ( sigma[k] == 8*(d1%2)+4*(c1%2)+2*(b1%2)+1*(a1%2) ) {
                        s->block[j].color = k;
                        s->block_list[k][color_counter[k]] = j;
                        color_counter[k]++;
                        break;
                      }
                  }
                  
                  if ( s->num_colors == 1 || s->num_colors == 2 ) {
                    for ( mu=0; mu<4; mu++ ) {
                      if ( ( (*count[mu]) == 0 ) || ( (*count[mu]+1) == le[mu]/block_size[mu] ) )
                        s->block[j].no_comm = 0;
                    }
                    
                    if ( s->num_colors == 2 ) {
                      // calculate boundary correspondence of the block
                      int count_plus=0, count_minus=0, count_inner=0, index;
                      for ( mu=0; mu<4; mu++ ) {
                        if ( (*count[mu]) == 0 )
                          count_minus++;
                        if ( (*count[mu]+1) == le[mu]/block_size[mu] )
                          count_plus++;
                        if ( (*count[mu]) != 0 && (*count[mu]+1) != le[mu]/block_size[mu] )
                          count_inner++;
                      }
                      
                      if ( count_inner == 4 ) {
                        index = 4*s->block[j].color;
                      } else if ( count_minus == 0 ) {
                        if ( s->block[j].color == 0 ) index = 1;
                        else index = 7;
                      } else if ( count_plus == 0 ) {
                        if ( s->block[j].color == 0 ) index = 3;
                        else index = 5;
                      } else {
                        index = 2 + 4*s->block[j].color;
                      }
                      
                      s->block_list[index][s->block_list_length[index]] = j;
                      s->block_list_length[index]++;
                    }
                    
                  } else if ( s->num_colors == 16 ) {
                    k = s->block[j].color;
                    if ( k == 0 ) {
                      for ( mu=0; mu<4; mu++ ) {
                        if ( (*count[mu]) == 0 )
                          s->block[j].no_comm = 0;
                      }
                    } else {
                      mu = color_to_comm[k][0];
                      if ( (color_to_comm[k][1] == +1 && (*count[mu]+1) == le[mu]/block_size[mu]) ||
                           (color_to_comm[k][1] == -1 && (*count[mu]) == 0 ) )
                        s->block[j].no_comm = 0;
                    }
                  }
                  
                  j++;
                  
                  // set up index table
                  if ( l->depth == 0 && g.odd_even ) {
                    // odd even on the blocks
                    // even sites
                    for ( t=d1*block_size[T]; t<(d1+1)*block_size[T]; t++ )
                      for ( z=c1*block_size[Z]; z<(c1+1)*block_size[Z]; z++ )
                        for ( y=b1*block_size[Y]; y<(b1+1)*block_size[Y]; y++ )
                          for ( x=a1*block_size[X]; x<(a1+1)*block_size[X]; x++ ) {
                            if (((t-d1*block_size[T])+(z-c1*block_size[Z])+
                                 (y-b1*block_size[Y])+(x-a1*block_size[X]))%2 == 0 ) {
                              index = lex_index( t, z, y, x, dt );
                              it[index] = i;
                              i++;
                            }
                          }
                    // odd sites
                    for ( t=d1*block_size[T]; t<(d1+1)*block_size[T]; t++ )
                      for ( z=c1*block_size[Z]; z<(c1+1)*block_size[Z]; z++ )
                        for ( y=b1*block_size[Y]; y<(b1+1)*block_size[Y]; y++ )
                          for ( x=a1*block_size[X]; x<(a1+1)*block_size[X]; x++ ) {
                            if (((t-d1*block_size[T])+(z-c1*block_size[Z])+
                                  (y-b1*block_size[Y])+(x-a1*block_size[X]))%2 == 1 ) {
                              index = lex_index( t, z, y, x, dt );
                              it[index] = i;
                              i++;
                            }
                          }
                  } else {
                    // no odd even
                    for ( t=d1*block_size[T]; t<(d1+1)*block_size[T]; t++ )
                      for ( z=c1*block_size[Z]; z<(c1+1)*block_size[Z]; z++ )
                        for ( y=b1*block_size[Y]; y<(b1+1)*block_size[Y]; y++ )
                          for ( x=a1*block_size[X]; x<(a1+1)*block_size[X]; x++ ) {
                            index = lex_index( t, z, y, x, dt );
                            it[index] = i;
                            i++;
                          }
                  }
                }
        }
        
  

  // boundaries
  for ( mu=0; mu<4; mu++ ) {
    l_st[mu] = le[mu];
    l_en[mu] = le[mu]+1;
    for ( t=l_st[T]; t<l_en[T]; t++ )
      for ( z=l_st[Z]; z<l_en[Z]; z++ )
        for ( y=l_st[Y]; y<l_en[Y]; y++ )
          for ( x=l_st[X]; x<l_en[X]; x++ ) {
            index = lex_index( t, z, y, x, dt );
            it[index] = i;
            i++;
          }  
    l_st[mu] = ls[mu];
    l_en[mu] = le[mu];
  }
  
  // define negative boundaries
  for ( mu=0; mu<4; mu++ ) {
    l_st[mu] = ls[mu]-1;
    l_en[mu] = ls[mu];
    for ( t=l_st[T]; t<l_en[T]; t++ )
      for ( z=l_st[Z]; z<l_en[Z]; z++ )
        for ( y=l_st[Y]; y<l_en[Y]; y++ )
          for ( x=l_st[X]; x<l_en[X]; x++ ) {
            index = lex_mod_index( t, z, y, x, dt );
            it[index] = i;
            i++;
          }  
    l_st[mu] = ls[mu];
    l_en[mu] = le[mu];
  }

  i = 0; j = 0;
  // block boundary table
  for ( d0=0; d0<agg_split[T]; d0++ )
    for ( c0=0; c0<agg_split[Z]; c0++ )
      for ( b0=0; b0<agg_split[Y]; b0++ )
        for ( a0=0; a0<agg_split[X]; a0++ )
          
          for ( d1=d0*block_split[T]; d1<(d0+1)*block_split[T]; d1++ )
            for ( c1=c0*block_split[Z]; c1<(c0+1)*block_split[Z]; c1++ )
              for ( b1=b0*block_split[Y]; b1<(b0+1)*block_split[Y]; b1++ )
                for ( a1=a0*block_split[X]; a1<(a0+1)*block_split[X]; a1++ ) {
                  // for all blocks
                  i=0;
                  int block_start[4], block_end[4], tmp;
                  
                  block_start[T] = d1*block_size[T]; block_start[Z] = c1*block_size[Z];
                  block_start[Y] = b1*block_size[Y]; block_start[X] = a1*block_size[X];
                  block_end[T] = (d1+1)*block_size[T]; block_end[Z] = (c1+1)*block_size[Z];
                  block_end[Y] = (b1+1)*block_size[Y]; block_end[X] = (a1+1)*block_size[X];
                  
                  for ( mu=0; mu<4; mu++ ) {
                    tmp = block_start[mu];
                                        
                    // minus dir
                    block_start[mu] = block_end[mu]-1;
                    for ( t=block_start[T]; t<block_end[T]; t++ )
                      for ( z=block_start[Z]; z<block_end[Z]; z++ )
                        for ( y=block_start[Y]; y<block_end[Y]; y++ )
                          for ( x=block_start[X]; x<block_end[X]; x++ ) {
                            s->block[j].bt[i] = site_index( t, z, y, x, dt, it );
                            i++;
                            s->block[j].bt[i] = connect_link_PRECISION( t, z, y, x, mu, +1, dt, it, s, l );
                            i++;
                          }
                          
                    block_start[mu] = tmp;
                    tmp = block_end[mu];
                        
                    // plus dir
                    block_end[mu] = block_start[mu]+1;
                    for ( t=block_start[T]; t<block_end[T]; t++ )
                      for ( z=block_start[Z]; z<block_end[Z]; z++ )
                        for ( y=block_start[Y]; y<block_end[Y]; y++ )
                          for ( x=block_start[X]; x<block_end[X]; x++ ) {
                            s->block[j].bt[i] = site_index( t, z, y, x, dt, it );
                            i++;
                            s->block[j].bt[i] = connect_link_PRECISION( t, z, y, x, mu, -1, dt, it, s, l );
                            i++;
                          }
                          block_end[mu] = tmp;
                  }
                  j++;
                }
  
  // index table for block dirac operator
  if ( l->depth == 0 && g.odd_even ) {
    count[T] = &t; count[Z] = &z; count[Y] = &y; count[X] = &x;
    i=0; j=0;
    for ( t=0; t<block_size[T]; t++ )
      for ( z=0; z<block_size[Z]; z++ )
        for ( y=0; y<block_size[Y]; y++ )
          for ( x=0; x<block_size[X]; x++ ) {
            if ( (t+z+y+x)%2 == 0 ) {
              i++;
            } else {
              j++;
            }
          }
    s->num_block_even_sites = i;
    s->num_block_odd_sites = j;
    
    for ( mu=0; mu<4; mu++ ) {
      // even sites, plus dir ( = odd sites, minus dir )
      i=0; j=0;
      for ( t=0; t<block_size[T]; t++ )
        for ( z=0; z<block_size[Z]; z++ )
          for ( y=0; y<block_size[Y]; y++ )
            for ( x=0; x<block_size[X]; x++ ) {
              if ( (t+z+y+x)%2 == 0 ) {
                if ( *(count[mu]) < block_size[mu]-1 ) {
                  s->oe_index[mu][j] = i; 
                  j++;
                }
                i++;
              }
            }
      s->dir_length_even[mu] = j;
      // odd sites, plus dir ( = even sites, minus dir )
      j=0;
      for ( t=0; t<block_size[T]; t++ )
        for ( z=0; z<block_size[Z]; z++ )
          for ( y=0; y<block_size[Y]; y++ )
            for ( x=0; x<block_size[X]; x++ ) {
              if ( (t+z+y+x)%2 == 1 ) {
                if ( *(count[mu]) < block_size[mu]-1 ) {
                  s->oe_index[mu][s->dir_length_even[mu]+j] = i; 
                  j++;
                }
                i++;
              }
            }
      s->dir_length_odd[mu] = j;
    }
  }
  
  count[T] = &t; count[Z] = &z; count[Y] = &y; count[X] = &x;
  for ( mu=0; mu<4; mu++ ) {
    j=0;
    for ( t=0; t<block_size[T]; t++ )
      for ( z=0; z<block_size[Z]; z++ )
        for ( y=0; y<block_size[Y]; y++ )
          for ( x=0; x<block_size[X]; x++ ) {
            if ( *(count[mu]) < block_size[mu]-1 ) {
              s->index[mu][j] =  site_index( t, z, y, x, dt, it ); j++;
            }
          }
  }
  
  // define neighbor table (for the application of the entire operator),
  // negative inner boundary table (for communication),
  // translation table (for translation to lexicographical site ordnering)
  define_nt_bt_tt( s->op.neighbor_table, s->op.backward_neighbor_table, s->op.c.boundary_table, s->op.translation_table, it, dt, l );

#ifdef CUDA_OPT

  /*

  int color, comms_ctr, noncomms_ctr;

  // FIXME: call malloc with error check

  // TODO: should I move some of these lines to another schwarz_* function ?

  s->nr_DD_blocks_in_comms = (int*) malloc( s->num_colors*sizeof(int) );
  s->nr_DD_blocks_notin_comms = (int*) malloc( s->num_colors*sizeof(int) );

  s->DD_blocks_in_comms = (int**) malloc( s->num_colors*sizeof(int*) );
  s->DD_blocks_notin_comms = (int**) malloc( s->num_colors*sizeof(int*) );

  for(color=0; color<s->num_colors; color++){
    for(i=0; i<s->num_blocks; i++){
      if ( color == s->block[i].color && s->block[i].no_comm ) {
        s->nr_DD_blocks_notin_comms[color]++;
      }
      else if( color == s->block[i].color && !s->block[i].no_comm ){
        s->nr_DD_blocks_in_comms[color]++;
      }
    }
  }

  for(color=0; color<s->num_colors; color++){
    s->DD_blocks_in_comms[color] = (int*) malloc( s->nr_DD_blocks_in_comms[color]*sizeof(int) );
    s->DD_blocks_notin_comms[color] = (int*) malloc( s->nr_DD_blocks_notin_comms[color]*sizeof(int) );
  }

  for(color=0; color<s->num_colors; color++){
    comms_ctr = 0;
    noncomms_ctr = 0;
    for(i=0; i<s->num_blocks; i++){
      if ( color == s->block[i].color && s->block[i].no_comm ) {
        s->DD_blocks_notin_comms[color][noncomms_ctr] = i;
        noncomms_ctr++;
      }
      else if( color == s->block[i].color && !s->block[i].no_comm ){
        s->DD_blocks_in_comms[color][comms_ctr] = i;
        comms_ctr++;
      }
    }
  }

  (s->cu_s).DD_blocks_in_comms = (int**) malloc( s->num_colors*sizeof(int*) );
  (s->cu_s).DD_blocks_notin_comms = (int**) malloc( s->num_colors*sizeof(int*) );

  for(color=0; color<s->num_colors; color++){
    cuda_safe_call( cudaMalloc( (void**) (&( (s->cu_s).DD_blocks_in_comms[color] )), s->nr_DD_blocks_in_comms[color]*sizeof(int) ) );
    cuda_safe_call( cudaMalloc( (void**) (&( (s->cu_s).DD_blocks_notin_comms[color] )), s->nr_DD_blocks_notin_comms[color]*sizeof(int) ) );
  }

  for(color=0; color<s->num_colors; color++){
    cuda_safe_call( cudaMemcpy((s->cu_s).DD_blocks_in_comms[color], s->DD_blocks_in_comms[color], s->nr_DD_blocks_in_comms[color]*sizeof(int), cudaMemcpyHostToDevice) );
    cuda_safe_call( cudaMemcpy((s->cu_s).DD_blocks_notin_comms[color], s->DD_blocks_notin_comms[color], s->nr_DD_blocks_notin_comms[color]*sizeof(int), cudaMemcpyHostToDevice) );
  }

  cuda_safe_call( cudaMalloc( (void**) (&( (s->cu_s).block )), s->num_blocks*sizeof(block_struct) ) );
  cuda_safe_call( cudaMemcpy((s->cu_s).block, s->block, s->num_blocks*sizeof(block_struct), cudaMemcpyHostToDevice) );

  // CRITICAL DATA MOVEMENT: copying op.oe_clover_vectorized to the GPU
  //-------------------------------------------------------------------------------------------------------------------------
  int b, h, nr_DD_sites;

  nr_DD_sites = s->num_block_sites;

  schwarz_PRECISION_struct_on_gpu *out = &(s->s_on_gpu_cpubuff);
  schwarz_PRECISION_struct *in = s;

  cu_cmplx_PRECISION *buf_D_oe_cpu, *buf_D_oe_cpu_bare;
  cu_config_PRECISION *buf_D_oe_gpu;

  if(g.csw != 0){
    buf_D_oe_cpu = (cu_cmplx_PRECISION*) malloc( 72 * nr_DD_sites*in->num_blocks * sizeof(cu_cmplx_PRECISION) );
    cuda_safe_call( cudaMalloc( (void**) &(buf_D_oe_gpu), 72 * sizeof(cu_config_PRECISION) * nr_DD_sites*in->num_blocks ) );
  }
  else{
    buf_D_oe_cpu = (cu_cmplx_PRECISION*) malloc( 12 * nr_DD_sites*in->num_blocks * sizeof(cu_cmplx_PRECISION) );
    cuda_safe_call( cudaMalloc( (void**) &(buf_D_oe_gpu), 12 * sizeof(cu_config_PRECISION) * nr_DD_sites*in->num_blocks ) );
  }
  buf_D_oe_cpu_bare = buf_D_oe_cpu;

  PRECISION *op_oe_vect_bare = in->op.oe_clover_vectorized;
  PRECISION *op_oe_vect = op_oe_vect_bare;

  //TODO: put the following 4 lines within the appropriate if statement
  PRECISION M_tmp[144];
  PRECISION *M_tmp1, *M_tmp2;
  M_tmp1 = M_tmp;
  M_tmp2 = M_tmp + 72;

  for(b=0; b < in->num_blocks; b++){

    if(g.csw != 0){
      for(h=0; h<nr_DD_sites; h++){
        //the following snippet of code was taken from function sse_site_clover_invert_float(...) in sse_dirac.c
        for ( k=0; k<12; k+=SIMD_LENGTH_float ) {
          for ( j=0; j<6; j++ ) {
            for ( i=k; i<k+SIMD_LENGTH_float; i++ ) {
              if ( i<6 ) {
                M_tmp1[12*j+i] = *op_oe_vect;
                M_tmp1[12*j+i+6] = *(op_oe_vect+SIMD_LENGTH_float);
              } else {
                M_tmp2[12*j+i-6] = *op_oe_vect;
                M_tmp2[12*j+i] = *(op_oe_vect+SIMD_LENGTH_float);
              }
              op_oe_vect++;
            }
            op_oe_vect += SIMD_LENGTH_float;
          }
        }

        //the following snippet of code was taken from the function sse_cgem_inverse(...) within sse_blas_vectorized.h
        int N=6;
        //cu_cmplx_PRECISION tmpA[2*N*N];
        cu_cmplx_PRECISION *tmpA_1, *tmpA_2;
        tmpA_1 = buf_D_oe_cpu;
        tmpA_2 = buf_D_oe_cpu + N*N;
        for ( j=0; j<N; j++ ) {
          for ( i=0; i<N; i++ ) {
            tmpA_1[i+N*j] = make_cu_cmplx_PRECISION(M_tmp1[2*j*N+i], M_tmp1[(2*j+1)*N+i]);
            tmpA_2[i+N*j] = make_cu_cmplx_PRECISION(M_tmp2[2*j*N+i], M_tmp2[(2*j+1)*N+i]);
            //printf("%f + i%f\n", cu_creal_PRECISION(tmpA_2[i+N*j]), cu_cimag_PRECISION(tmpA_2[i+N*j]));
          }
        }

        buf_D_oe_cpu += 72;
        op_oe_vect_bare += 144;
        op_oe_vect = op_oe_vect_bare;
      }
    }
    else{
      //TODO !!
    }

  }

  //MPI_Barrier(MPI_COMM_WORLD);
  //MPI_Abort(MPI_COMM_WORLD, 911);

  if(g.csw != 0){
    cuda_safe_call( cudaMemcpy(buf_D_oe_gpu, buf_D_oe_cpu_bare, 72*sizeof(cu_cmplx_PRECISION)*nr_DD_sites*in->num_blocks, cudaMemcpyHostToDevice) );
  }
  else{
    cuda_safe_call( cudaMemcpy(buf_D_oe_gpu, buf_D_oe_cpu_bare, 12*sizeof(cu_cmplx_PRECISION)*nr_DD_sites*in->num_blocks, cudaMemcpyHostToDevice) );
  }

  free(buf_D_oe_cpu_bare);

  //cudaMemcpy(&(out->op.oe_clover_vectorized), &buf_D_oe_gpu, sizeof(cu_config_PRECISION*), cudaMemcpyHostToDevice);
  out->op.oe_clover_vectorized = buf_D_oe_gpu;
  //-------------------------------------------------------------------------------------------------------------------------

  (s->s_on_gpu_cpubuff).num_block_even_sites = s->num_block_even_sites;
  (s->s_on_gpu_cpubuff).num_block_odd_sites = s->num_block_odd_sites;

  // After all the allocations and definitions associated to s->s_on_gpu_cpubuff, it's time to move to the GPU
  cuda_safe_call( cudaMemcpy(s->s_on_gpu, &(s->s_on_gpu_cpubuff), 1*sizeof(schwarz_PRECISION_struct_on_gpu), cudaMemcpyHostToDevice) );

  */

#endif
}


void schwarz_PRECISION_boundary_update( schwarz_PRECISION_struct *s, level_struct *l ) {

/*********************************************************************************
* Updates the current level hopping term in "s->op.D" on the process boundaries
* in all negative directions. This is necessary for enabling Schwarz to perform
* local block residual updates on demand.
*********************************************************************************/   

  int i, t, z, y, x, mu, nu, index, *it = s->op.index_table, *dt = s->op.table_dim,
      ls[4], le[4], buf_length[4], link_size;
  vector_PRECISION buf[4] = {NULL,NULL,NULL,NULL}, rbuf[4] = {NULL,NULL,NULL,NULL};
  config_PRECISION D=s->op.D;
  
  for ( mu=0; mu<4; mu++ ) {
    ls[mu] = 0;
    le[mu] = l->local_lattice[mu];
    buf_length[mu] = 0;
  }
  
  if ( l->depth == 0 )
    link_size = 4*9;
  else
    link_size = 4*SQUARE(l->num_lattice_site_var);
  
  // allocate buffers
  for ( mu=0; mu<4; mu++ ) {
    if ( l->global_splitting[mu] > 1 ) {
      buf_length[mu] = link_size;
      for ( nu=0; nu<4; nu++ ) {
        if ( nu != mu )
          buf_length[mu] *= le[nu];
      }
      MALLOC( buf[mu], complex_PRECISION, buf_length[mu] );
      MALLOC( rbuf[mu], complex_PRECISION, buf_length[mu] );
    }
  }
  
  // post recv for desired directions
  for ( mu=0; mu<4; mu++ ) {
    if ( l->global_splitting[mu] > 1 ) {
      MPI_Irecv( rbuf[mu], buf_length[mu], MPI_COMPLEX_PRECISION, l->neighbor_rank[2*mu+1],
                 2*mu+1, g.comm_cart, &(s->op.c.rreqs[2*mu+1]) );
    }
  }
  
  // buffer data for send and send it
  for ( mu=0; mu<4; mu++ ) {
    if ( l->global_splitting[mu] > 1 ) {
      ls[mu] = l->local_lattice[mu]-1;
      i=0;
      for ( t=ls[T]; t<le[T]; t++ )
        for ( z=ls[Z]; z<le[Z]; z++ )
          for ( y=ls[Y]; y<le[Y]; y++ )
            for ( x=ls[X]; x<le[X]; x++ ) {
              index = site_index( t, z, y, x, dt, it );
              vector_PRECISION_copy( buf[mu]+i*link_size, D+index*link_size, 0, link_size, l );
              i++;
            }
      MPI_Isend( buf[mu], buf_length[mu], MPI_COMPLEX_PRECISION, l->neighbor_rank[2*mu],
                 2*mu+1, g.comm_cart, &(s->op.c.sreqs[2*mu+1]) );
      ls[mu] = 0;
    }
  }
  
  // store links in desired ordering after recv
  for ( mu=0; mu<4; mu++ ) {
    if ( l->global_splitting[mu] > 1 ) {
      MPI_Wait( &(s->op.c.rreqs[2*mu+1]), MPI_STATUS_IGNORE );
      ls[mu] = -1;
      le[mu] = 0;
      i=0;
      for ( t=ls[T]; t<le[T]; t++ )
        for ( z=ls[Z]; z<le[Z]; z++ )
          for ( y=ls[Y]; y<le[Y]; y++ )
            for ( x=ls[X]; x<le[X]; x++ ) {
              index = site_mod_index( t, z, y, x, dt, it );
              vector_PRECISION_copy( D+index*link_size, rbuf[mu]+i*link_size, 0, link_size, l );
              i++;
            }
      ls[mu] = 0;
      le[mu] = l->local_lattice[mu];
    }
  }
  
  // free buffers
  for ( mu=0; mu<4; mu++ ) {
    if ( l->global_splitting[mu] > 1 ) {
      MPI_Wait( &(s->op.c.sreqs[2*mu+1]), MPI_STATUS_IGNORE );
      FREE( buf[mu], complex_PRECISION, buf_length[mu] );
      FREE( rbuf[mu], complex_PRECISION, buf_length[mu] );
    }
  }
}

#ifndef OPTIMIZED_NEIGHBOR_COUPLING_PRECISION
void block_PRECISION_boundary_op( vector_PRECISION eta, vector_PRECISION phi, int k,
                                  schwarz_PRECISION_struct *s, level_struct *l ) {
  // k: number of current block
  int i, mu, index, neighbor_index, *bbl = s->block_boundary_length;
  complex_PRECISION buf1[12], *buf2=buf1+6;
  config_PRECISION D_pt, D = s->op.D;
  vector_PRECISION phi_pt, eta_pt;
  
  mu=T;
  // plus mu direction
  for ( i=bbl[2*mu]; i<bbl[2*mu+1]; i+=2 ) {
    index = s->block[k].bt[i];
    neighbor_index = s->block[k].bt[i+1];
    D_pt = D + 36*index + 9*mu;
    phi_pt = phi + 12*neighbor_index;
    eta_pt = eta + 12*index;
    prp_T_PRECISION( buf1, phi_pt );
    mvm_PRECISION( buf2, D_pt, buf1 );
    mvm_PRECISION( buf2+3, D_pt, buf1+3 );
    pbp_su3_T_PRECISION( buf2, eta_pt );
  }
  // minus mu direction
  for ( i=bbl[2*mu+1]; i<bbl[2*mu+2]; i+=2 ) {
    index = s->block[k].bt[i];
    neighbor_index = s->block[k].bt[i+1];
    D_pt = D + 36*neighbor_index + 9*mu;
    phi_pt = phi + 12*neighbor_index;
    eta_pt = eta + 12*index;
    prn_T_PRECISION( buf1, phi_pt );
    mvmh_PRECISION( buf2, D_pt, buf1 );
    mvmh_PRECISION( buf2+3, D_pt, buf1+3 );
    pbn_su3_T_PRECISION( buf2, eta_pt );
  }
  
  mu=Z;
  // plus mu direction
  for ( i=bbl[2*mu]; i<bbl[2*mu+1]; i+=2 ) {
    index = s->block[k].bt[i];
    neighbor_index = s->block[k].bt[i+1];
    D_pt = D + 36*index + 9*mu;
    phi_pt = phi + 12*neighbor_index;
    eta_pt = eta + 12*index;
    prp_Z_PRECISION( buf1, phi_pt );
    mvm_PRECISION( buf2, D_pt, buf1 );
    mvm_PRECISION( buf2+3, D_pt, buf1+3 );
    pbp_su3_Z_PRECISION( buf2, eta_pt );
  }
  // minus mu direction
  for ( i=bbl[2*mu+1]; i<bbl[2*mu+2]; i+=2 ) {
    index = s->block[k].bt[i];
    neighbor_index = s->block[k].bt[i+1];
    D_pt = D + 36*neighbor_index + 9*mu;
    phi_pt = phi + 12*neighbor_index;
    eta_pt = eta + 12*index;
    prn_Z_PRECISION( buf1, phi_pt );
    mvmh_PRECISION( buf2, D_pt, buf1 );
    mvmh_PRECISION( buf2+3, D_pt, buf1+3 );
    pbn_su3_Z_PRECISION( buf2, eta_pt );
  }
  
  mu=Y;
  // plus mu direction
  for ( i=bbl[2*mu]; i<bbl[2*mu+1]; i+=2 ) {
    index = s->block[k].bt[i];
    neighbor_index = s->block[k].bt[i+1];
    D_pt = D + 36*index + 9*mu;
    phi_pt = phi + 12*neighbor_index;
    eta_pt = eta + 12*index;
    prp_Y_PRECISION( buf1, phi_pt );
    mvm_PRECISION( buf2, D_pt, buf1 );
    mvm_PRECISION( buf2+3, D_pt, buf1+3 );
    pbp_su3_Y_PRECISION( buf2, eta_pt );
  }
  // minus mu direction
  for ( i=bbl[2*mu+1]; i<bbl[2*mu+2]; i+=2 ) {
    index = s->block[k].bt[i];
    neighbor_index = s->block[k].bt[i+1];
    D_pt = D + 36*neighbor_index + 9*mu;
    phi_pt = phi + 12*neighbor_index;
    eta_pt = eta + 12*index;
    prn_Y_PRECISION( buf1, phi_pt );
    mvmh_PRECISION( buf2, D_pt, buf1 );
    mvmh_PRECISION( buf2+3, D_pt, buf1+3 );
    pbn_su3_Y_PRECISION( buf2, eta_pt );
  }
  
  mu=X;
  // plus mu direction
  for ( i=bbl[2*mu]; i<bbl[2*mu+1]; i+=2 ) {
    index = s->block[k].bt[i];
    neighbor_index = s->block[k].bt[i+1];
    D_pt = D + 36*index + 9*mu;
    phi_pt = phi + 12*neighbor_index;
    eta_pt = eta + 12*index;
    prp_X_PRECISION( buf1, phi_pt );
    mvm_PRECISION( buf2, D_pt, buf1 );
    mvm_PRECISION( buf2+3, D_pt, buf1+3 );
    pbp_su3_X_PRECISION( buf2, eta_pt );
  }
  // minus mu direction
  for ( i=bbl[2*mu+1]; i<bbl[2*mu+2]; i+=2 ) {
    index = s->block[k].bt[i];
    neighbor_index = s->block[k].bt[i+1];
    D_pt = D + 36*neighbor_index + 9*mu;
    phi_pt = phi + 12*neighbor_index;
    eta_pt = eta + 12*index;
    prn_X_PRECISION( buf1, phi_pt );
    mvmh_PRECISION( buf2, D_pt, buf1 );
    mvmh_PRECISION( buf2+3, D_pt, buf1+3 );
    pbn_su3_X_PRECISION( buf2, eta_pt );
  }  
}
#endif


#ifndef OPTIMIZED_NEIGHBOR_COUPLING_PRECISION
void n_block_PRECISION_boundary_op( vector_PRECISION eta, vector_PRECISION phi, int k,
                                    schwarz_PRECISION_struct *s, level_struct *l ) {
  // k: number of current block
  int i, mu, index, neighbor_index, *bbl = s->block_boundary_length;
  complex_PRECISION buf1[12], *buf2=buf1+6;
  config_PRECISION D_pt, D = s->op.D;
  vector_PRECISION phi_pt, eta_pt;
  
  mu=T;
  // plus mu direction
  for ( i=bbl[2*mu]; i<bbl[2*mu+1]; i+=2 ) {
    index = s->block[k].bt[i];
    neighbor_index = s->block[k].bt[i+1];
    D_pt = D + 36*index + 9*mu;
    phi_pt = phi + 12*neighbor_index;
    eta_pt = eta + 12*index;
    prp_T_PRECISION( buf1, phi_pt );
    nmvm_PRECISION( buf2, D_pt, buf1 );
    nmvm_PRECISION( buf2+3, D_pt, buf1+3 );
    pbp_su3_T_PRECISION( buf2, eta_pt );
  }
  // minus mu direction
  for ( i=bbl[2*mu+1]; i<bbl[2*mu+2]; i+=2 ) {
    index = s->block[k].bt[i];
    neighbor_index = s->block[k].bt[i+1];
    D_pt = D + 36*neighbor_index + 9*mu;
    phi_pt = phi + 12*neighbor_index;
    eta_pt = eta + 12*index;
    prn_T_PRECISION( buf1, phi_pt );
    nmvmh_PRECISION( buf2, D_pt, buf1 );
    nmvmh_PRECISION( buf2+3, D_pt, buf1+3 );
    pbn_su3_T_PRECISION( buf2, eta_pt );
  }
  
  mu=Z;
  // plus mu direction
  for ( i=bbl[2*mu]; i<bbl[2*mu+1]; i+=2 ) {
    index = s->block[k].bt[i];
    neighbor_index = s->block[k].bt[i+1];
    D_pt = D + 36*index + 9*mu;
    phi_pt = phi + 12*neighbor_index;
    eta_pt = eta + 12*index;
    prp_Z_PRECISION( buf1, phi_pt );
    nmvm_PRECISION( buf2, D_pt, buf1 );
    nmvm_PRECISION( buf2+3, D_pt, buf1+3 );
    pbp_su3_Z_PRECISION( buf2, eta_pt );
  }
  // minus mu direction
  for ( i=bbl[2*mu+1]; i<bbl[2*mu+2]; i+=2 ) {
    index = s->block[k].bt[i];
    neighbor_index = s->block[k].bt[i+1];
    D_pt = D + 36*neighbor_index + 9*mu;
    phi_pt = phi + 12*neighbor_index;
    eta_pt = eta + 12*index;
    prn_Z_PRECISION( buf1, phi_pt );
    nmvmh_PRECISION( buf2, D_pt, buf1 );
    nmvmh_PRECISION( buf2+3, D_pt, buf1+3 );
    pbn_su3_Z_PRECISION( buf2, eta_pt );
  }
  
  mu=Y;
  // plus mu direction
  for ( i=bbl[2*mu]; i<bbl[2*mu+1]; i+=2 ) {
    index = s->block[k].bt[i];
    neighbor_index = s->block[k].bt[i+1];
    D_pt = D + 36*index + 9*mu;
    phi_pt = phi + 12*neighbor_index;
    eta_pt = eta + 12*index;
    prp_Y_PRECISION( buf1, phi_pt );
    nmvm_PRECISION( buf2, D_pt, buf1 );
    nmvm_PRECISION( buf2+3, D_pt, buf1+3 );
    pbp_su3_Y_PRECISION( buf2, eta_pt );
  }
  // minus mu direction
  for ( i=bbl[2*mu+1]; i<bbl[2*mu+2]; i+=2 ) {
    index = s->block[k].bt[i];
    neighbor_index = s->block[k].bt[i+1];
    D_pt = D + 36*neighbor_index + 9*mu;
    phi_pt = phi + 12*neighbor_index;
    eta_pt = eta + 12*index;
    prn_Y_PRECISION( buf1, phi_pt );
    nmvmh_PRECISION( buf2, D_pt, buf1 );
    nmvmh_PRECISION( buf2+3, D_pt, buf1+3 );
    pbn_su3_Y_PRECISION( buf2, eta_pt );
  }
  
  mu=X;
  // plus mu direction
  for ( i=bbl[2*mu]; i<bbl[2*mu+1]; i+=2 ) {
    index = s->block[k].bt[i];
    neighbor_index = s->block[k].bt[i+1];
    D_pt = D + 36*index + 9*mu;
    phi_pt = phi + 12*neighbor_index;
    eta_pt = eta + 12*index;
    prp_X_PRECISION( buf1, phi_pt );
    nmvm_PRECISION( buf2, D_pt, buf1 );
    nmvm_PRECISION( buf2+3, D_pt, buf1+3 );
    pbp_su3_X_PRECISION( buf2, eta_pt );
  }
  // minus mu direction
  for ( i=bbl[2*mu+1]; i<bbl[2*mu+2]; i+=2 ) {
    index = s->block[k].bt[i];
    neighbor_index = s->block[k].bt[i+1];
    D_pt = D + 36*neighbor_index + 9*mu;
    phi_pt = phi + 12*neighbor_index;
    eta_pt = eta + 12*index;
    prn_X_PRECISION( buf1, phi_pt );
    nmvmh_PRECISION( buf2, D_pt, buf1 );
    nmvmh_PRECISION( buf2+3, D_pt, buf1+3 );
    pbn_su3_X_PRECISION( buf2, eta_pt );
  }  
}
#endif


#ifndef OPTIMIZED_NEIGHBOR_COUPLING_PRECISION
void coarse_block_PRECISION_boundary_op( vector_PRECISION eta, vector_PRECISION phi,
                                         int k, schwarz_PRECISION_struct *s, level_struct *l ) {
  // k: number of current block
  int *bbl = s->block_boundary_length, n = l->num_lattice_site_var;
  config_PRECISION D = s->op.D;
  int link_size = SQUARE(l->num_lattice_site_var), site_size=4*link_size;
  
  for ( int mu=0; mu<4; mu++ ) {
    // plus mu direction
    for ( int i=bbl[2*mu]; i<bbl[2*mu+1]; i+=2 ) {
      int index = s->block[k].bt[i];
      int neighbor_index = s->block[k].bt[i+1];
      vector_PRECISION phi_pt = phi + n*neighbor_index;
      vector_PRECISION eta_pt = eta + n*index;
      config_PRECISION D_pt = D + site_size*index + link_size*mu;
      coarse_hopp_PRECISION( eta_pt, phi_pt, D_pt, l );
    }
    // minus mu direction
    for ( int i=bbl[2*mu+1]; i<bbl[2*mu+2]; i+=2 ) {
      int index = s->block[k].bt[i];
      int neighbor_index = s->block[k].bt[i+1];
      vector_PRECISION phi_pt = phi + n*neighbor_index;
      vector_PRECISION eta_pt = eta + n*index;
      config_PRECISION D_pt = D + site_size*neighbor_index + link_size*mu;
      coarse_daggered_hopp_PRECISION( eta_pt, phi_pt, D_pt, l );
    }
  }
}
#endif

#ifndef OPTIMIZED_NEIGHBOR_COUPLING_PRECISION
void n_coarse_block_PRECISION_boundary_op( vector_PRECISION eta, vector_PRECISION phi,
                                           int k, schwarz_PRECISION_struct *s, level_struct *l ) {
  // k: number of current block
  int *bbl = s->block_boundary_length, n = l->num_lattice_site_var;
  int link_size = SQUARE(l->num_lattice_site_var), site_size=4*link_size;
  config_PRECISION D = s->op.D;
  
  for ( int mu=0; mu<4; mu++ ) {
    // plus mu direction
    for ( int i=bbl[2*mu]; i<bbl[2*mu+1]; i+=2 ) {
      int index = s->block[k].bt[i];
      int neighbor_index = s->block[k].bt[i+1];
      vector_PRECISION phi_pt = phi + n*neighbor_index;
      vector_PRECISION eta_pt = eta + n*index;
      config_PRECISION D_pt = D + site_size*index + link_size*mu;
      coarse_n_hopp_PRECISION( eta_pt, phi_pt, D_pt, l );
    }
    // minus mu direction
    for ( int i=bbl[2*mu+1]; i<bbl[2*mu+2]; i+=2 ) {
      int index = s->block[k].bt[i];
      int neighbor_index = s->block[k].bt[i+1];
      vector_PRECISION phi_pt = phi + n*neighbor_index;
      vector_PRECISION eta_pt = eta + n*index;
      config_PRECISION D_pt = D + site_size*neighbor_index + link_size*mu;
      coarse_n_daggered_hopp_PRECISION( eta_pt, phi_pt, D_pt, l );
    }
  }
}
#endif

#if !defined(OPTIMIZED_NEIGHBOR_COUPLING_PRECISION) && !defined(OPTIMIZED_SELF_COUPLING_PRECISION)
void schwarz_PRECISION_setup( schwarz_PRECISION_struct *s, operator_double_struct *op_in, level_struct *l ) {

/*********************************************************************************  
* Copies the Dirac operator and the clover term from op_in into the Schwarz 
* struct (this function is depth 0 only).
* - operator_double_struct *op_in: Input operator.                                  
*********************************************************************************/

  int i, index, n = l->num_inner_lattice_sites, *tt = s->op.translation_table;
  config_PRECISION D_out_pt, clover_out_pt;
  config_double D_in_pt = op_in->D, clover_in_pt = op_in->clover;
  s->op.shift = op_in->shift;
  
  for ( i=0; i<n; i++ ) {
    index = tt[i];
    D_out_pt = s->op.D + 36*index;
    FOR36( *D_out_pt = (complex_PRECISION) *D_in_pt; D_out_pt++; D_in_pt++; )
  }
  
  if ( g.csw != 0 ) {
    for ( i=0; i<n; i++ ) {
      index = tt[i];
      clover_out_pt = s->op.clover + 42*index;
      FOR42( *clover_out_pt = (complex_PRECISION) *clover_in_pt; clover_out_pt++; clover_in_pt++; )
    }
  } else {
    for ( i=0; i<n; i++ ) {
      index = tt[i];
      clover_out_pt = s->op.clover + 12*index;
      FOR12( *clover_out_pt = (complex_PRECISION) *clover_in_pt; clover_out_pt++; clover_in_pt++; )
    }
  }
  
  if ( g.odd_even )
    schwarz_PRECISION_oddeven_setup( &(s->op), l );
  
  schwarz_PRECISION_boundary_update( s, l );
}
#endif


// TODO
#ifdef CUDA_OPT
void schwarz_PRECISION_setup_CUDA( schwarz_PRECISION_struct *s, operator_double_struct *op_in, level_struct *l ) {

  int color, comms_ctr, noncomms_ctr, i, j, k;

  // FIXME: call malloc with error check

  // TODO: should I move some of these lines to another schwarz_* function ?

  s->nr_DD_blocks_in_comms = (int*) malloc( s->num_colors*sizeof(int) );
  s->nr_DD_blocks_notin_comms = (int*) malloc( s->num_colors*sizeof(int) );

  s->DD_blocks_in_comms = (int**) malloc( s->num_colors*sizeof(int*) );
  s->DD_blocks_notin_comms = (int**) malloc( s->num_colors*sizeof(int*) );

  printf("s->num_colors = %d\n", s->num_colors);

  for(color=0; color<s->num_colors; color++){
    s->nr_DD_blocks_notin_comms[color] = 0;
    s->nr_DD_blocks_in_comms[color] = 0;
    for(i=0; i<s->num_blocks; i++){
      if ( color == s->block[i].color && s->block[i].no_comm ) {
        s->nr_DD_blocks_notin_comms[color]++;
      }
      else if( color == s->block[i].color && !s->block[i].no_comm ){
        s->nr_DD_blocks_in_comms[color]++;
      }
    }
  }

  for(color=0; color<s->num_colors; color++){
    //printf("s->nr_DD_blocks_in_comms[color] = %d\n", s->nr_DD_blocks_in_comms[color]);
    //printf("s->nr_DD_blocks_notin_comms[color] = %d\n", s->nr_DD_blocks_notin_comms[color]);
    s->DD_blocks_in_comms[color] = (int*) malloc( s->nr_DD_blocks_in_comms[color]*sizeof(int) );
    s->DD_blocks_notin_comms[color] = (int*) malloc( s->nr_DD_blocks_notin_comms[color]*sizeof(int) );
  }

  for(color=0; color<s->num_colors; color++){
    comms_ctr = 0;
    noncomms_ctr = 0;
    for(i=0; i<s->num_blocks; i++){
      if ( color == s->block[i].color && s->block[i].no_comm ) {
        s->DD_blocks_notin_comms[color][noncomms_ctr] = i;
        noncomms_ctr++;
      }
      else if( color == s->block[i].color && !s->block[i].no_comm ){
        s->DD_blocks_in_comms[color][comms_ctr] = i;
        comms_ctr++;
      }
    }
  }

  (s->cu_s).DD_blocks_in_comms = (int**) malloc( s->num_colors*sizeof(int*) );
  (s->cu_s).DD_blocks_notin_comms = (int**) malloc( s->num_colors*sizeof(int*) );

  for(color=0; color<s->num_colors; color++){
    cuda_safe_call( cudaMalloc( (void**) (&( (s->cu_s).DD_blocks_in_comms[color] )), s->nr_DD_blocks_in_comms[color]*sizeof(int) ) );
    cuda_safe_call( cudaMalloc( (void**) (&( (s->cu_s).DD_blocks_notin_comms[color] )), s->nr_DD_blocks_notin_comms[color]*sizeof(int) ) );
  }

  for(color=0; color<s->num_colors; color++){
    cuda_safe_call( cudaMemcpy((s->cu_s).DD_blocks_in_comms[color], s->DD_blocks_in_comms[color], s->nr_DD_blocks_in_comms[color]*sizeof(int), cudaMemcpyHostToDevice) );
    cuda_safe_call( cudaMemcpy((s->cu_s).DD_blocks_notin_comms[color], s->DD_blocks_notin_comms[color], s->nr_DD_blocks_notin_comms[color]*sizeof(int), cudaMemcpyHostToDevice) );
  }

  cuda_safe_call( cudaMalloc( (void**) (&( (s->cu_s).block )), s->num_blocks*sizeof(block_struct) ) );
  cuda_safe_call( cudaMemcpy((s->cu_s).block, s->block, s->num_blocks*sizeof(block_struct), cudaMemcpyHostToDevice) );

  // CRITICAL DATA MOVEMENT: copying op.oe_clover_vectorized to the GPU
  //-------------------------------------------------------------------------------------------------------------------------
  int b, h, nr_DD_sites;

  nr_DD_sites = s->num_block_sites;

  schwarz_PRECISION_struct_on_gpu *out = &(s->s_on_gpu_cpubuff);
  schwarz_PRECISION_struct *in = s;

  cu_cmplx_PRECISION *buf_D_oe_cpu, *buf_D_oe_cpu_bare;
  cu_config_PRECISION *buf_D_oe_gpu;

  if(g.csw != 0){
    buf_D_oe_cpu = (cu_cmplx_PRECISION*) malloc( 72 * nr_DD_sites*in->num_blocks * sizeof(cu_cmplx_PRECISION) );
    cuda_safe_call( cudaMalloc( (void**) &(buf_D_oe_gpu), 72 * sizeof(cu_config_PRECISION) * nr_DD_sites*in->num_blocks ) );
  }
  else{
    buf_D_oe_cpu = (cu_cmplx_PRECISION*) malloc( 12 * nr_DD_sites*in->num_blocks * sizeof(cu_cmplx_PRECISION) );
    cuda_safe_call( cudaMalloc( (void**) &(buf_D_oe_gpu), 12 * sizeof(cu_config_PRECISION) * nr_DD_sites*in->num_blocks ) );
  }
  buf_D_oe_cpu_bare = buf_D_oe_cpu;

  PRECISION *op_oe_vect_bare = in->op.oe_clover_vectorized;
  PRECISION *op_oe_vect = op_oe_vect_bare;

  //TODO: put the following 4 lines within the appropriate if statement
  PRECISION M_tmp[144];
  PRECISION *M_tmp1, *M_tmp2;
  M_tmp1 = M_tmp;
  M_tmp2 = M_tmp + 72;

  for(b=0; b < in->num_blocks; b++){

    if(g.csw != 0){
      for(h=0; h<nr_DD_sites; h++){
        //the following snippet of code was taken from function sse_site_clover_invert_float(...) in sse_dirac.c
        for ( k=0; k<12; k+=SIMD_LENGTH_float ) {
          for ( j=0; j<6; j++ ) {
            for ( i=k; i<k+SIMD_LENGTH_float; i++ ) {
              if ( i<6 ) {
                M_tmp1[12*j+i] = *op_oe_vect;
                M_tmp1[12*j+i+6] = *(op_oe_vect+SIMD_LENGTH_float);
              } else {
                M_tmp2[12*j+i-6] = *op_oe_vect;
                M_tmp2[12*j+i] = *(op_oe_vect+SIMD_LENGTH_float);
              }
              op_oe_vect++;
            }
            op_oe_vect += SIMD_LENGTH_float;
          }
        }

        //the following snippet of code was taken from the function sse_cgem_inverse(...) within sse_blas_vectorized.h
        int N=6;
        //cu_cmplx_PRECISION tmpA[2*N*N];
        cu_cmplx_PRECISION *tmpA_1, *tmpA_2;
        tmpA_1 = buf_D_oe_cpu;
        tmpA_2 = buf_D_oe_cpu + N*N;
        for ( j=0; j<N; j++ ) {
          for ( i=0; i<N; i++ ) {
            tmpA_1[i+N*j] = make_cu_cmplx_PRECISION(M_tmp1[2*j*N+i], M_tmp1[(2*j+1)*N+i]);
            tmpA_2[i+N*j] = make_cu_cmplx_PRECISION(M_tmp2[2*j*N+i], M_tmp2[(2*j+1)*N+i]);
            //printf("%f + i%f\n", cu_creal_PRECISION(tmpA_2[i+N*j]), cu_cimag_PRECISION(tmpA_2[i+N*j]));
          }
        }

        /*
        if( b==100 && h==100 ){
        for( i=0; i<N; i++ ){
          for( j=0; j<N; j++ ){
            if( g.my_rank ==0 ) printf(" %f+i%f |", cu_creal_PRECISION(buf_D_oe_cpu[i+j*N]), cu_cimag_PRECISION(buf_D_oe_cpu[i+j*N]));
          }
          if( g.my_rank==0 ) printf("\n");
        }

        printf("\n");
        printf("\n");

        if( b==100 && h==100 ){
        for( i=0; i<N; i++ ){
          for( j=0; j<N; j++ ){
            if( g.my_rank ==0 ) printf(" %f+i%f |", cu_creal_PRECISION((buf_D_oe_cpu+36)[i+j*N]), cu_cimag_PRECISION((buf_D_oe_cpu+36)[i+j*N]));
          }
          if( g.my_rank==0 ) printf("\n");
        }
        }

        printf("\n");
        printf("\n");
        printf("\n");
        //MPI_Barrier(MPI_COMM_WORLD);
        //MPI_Finalize();
        //exit(0);
        }
        */

        buf_D_oe_cpu += 72;
        op_oe_vect_bare += 144;
        op_oe_vect = op_oe_vect_bare;
      }
    }
    else{
      //TODO !!
    }

  }


  // making use of Doo^-1 (i.e. buf_D_oe_cpu_bare) being Hermitian to store in reduced form
  // IMPORTANT: this matrix is stored in column-form

  cu_cmplx_PRECISION *buf_D_oe_cpu_gpustorg, *buf_D_oe_gpu_gpustorg, *buf_D_oe_cpu_gpustorg_bare;

  if(g.csw != 0){
    buf_D_oe_cpu_gpustorg = (cu_cmplx_PRECISION*) malloc( 42 * nr_DD_sites*in->num_blocks * sizeof(cu_cmplx_PRECISION) );
    cuda_safe_call( cudaMalloc( (void**) &(buf_D_oe_gpu_gpustorg), 42 * sizeof(cu_cmplx_PRECISION) * nr_DD_sites*in->num_blocks ) );
  }
  else{
    // TODO !
    //buf_D_oe_cpu = (cu_cmplx_PRECISION*) malloc( 12 * nr_DD_sites*in->num_blocks * sizeof(cu_cmplx_PRECISION) );
    //cuda_safe_call( cudaMalloc( (void**) &(buf_D_oe_gpu), 12 * sizeof(cu_config_PRECISION) * nr_DD_sites*in->num_blocks ) );
  }

  buf_D_oe_cpu = buf_D_oe_cpu_bare;
  buf_D_oe_cpu_gpustorg_bare = buf_D_oe_cpu_gpustorg;

  for(b=0; b < in->num_blocks; b++){
    for(h=0; h<nr_DD_sites; h++){
      int N=6, k=0;
      for ( j=0; j<N; j++ ) {
        for ( i=j; i<N; i++ ) {

          (buf_D_oe_cpu_gpustorg +    0)[k] = (buf_D_oe_cpu +    0)[i+j*N];
          (buf_D_oe_cpu_gpustorg + 42/2)[k] = (buf_D_oe_cpu + 72/2)[i+j*N];
          k++;

        }
      }


        /*
        int matrx_indx, local_idx;
        if( b==100 && h==100 ){
        for( i=0; i<2; i++ ){
          for( local_idx=0; local_idx<N; local_idx++ ){
            for( j=0; j<N; j++ ){

              if( local_idx>j ){
                matrx_indx = 21 - (5-j+1)*((5-j+1)+1)/2 + (local_idx-j);
              }
              else if( local_idx==j ){
                matrx_indx = 21 - (5-j+1)*((5-j+1)+1)/2;
              }
              else{
                matrx_indx = 21 - (5-local_idx+1)*((5-local_idx+1)+1)/2 + (j-local_idx);
              }

              //if( (threadIdx.x + blockDim.x * blockIdx.x)==0 ){
              //  printf("local_idx=%d, j=%d\n", local_idx, j);
              //}

              if( local_idx>j || local_idx==j ){
                //eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ], cu_cmul_PRECISION( (op_clov_vect_site + i*21)[matrx_indx], phi_site[j + 6*i] ) );
                if( g.my_rank ==0 ) printf(" %f+i%f |", cu_creal_PRECISION((buf_D_oe_cpu_gpustorg+i*21)[matrx_indx]), cu_cimag_PRECISION((buf_D_oe_cpu_gpustorg+i*21)[matrx_indx]));
              }
              else{
                //eta_site[ 6*i + local_idx ] = cu_cadd_PRECISION( eta_site[ 6*i + local_idx ], cu_cmul_PRECISION( cu_conj_PRECISION((op_clov_vect_site + i*21)[matrx_indx]), phi_site[j + 6*i] ) );
                if( g.my_rank ==0 ) printf(" %f+i%f |", cu_creal_PRECISION(cu_conj_PRECISION((buf_D_oe_cpu_gpustorg+i*21)[matrx_indx])), cu_cimag_PRECISION(cu_conj_PRECISION((buf_D_oe_cpu_gpustorg+i*21)[matrx_indx])));
              }

              //if( g.my_rank ==0 ) printf(" %f+i%f |", cu_creal_PRECISION(buf_D_oe_cpu[i*N+j]), cu_cimag_PRECISION(buf_D_oe_cpu[i*N+j]));

            }
            if( g.my_rank==0 ) printf("\n");
          }
          if( g.my_rank==0 ) printf("\n\n");
        }
        }

        //printf("\n");
        //MPI_Barrier(MPI_COMM_WORLD);
        //MPI_Finalize();
        //exit(0);
        */


      buf_D_oe_cpu += 72;
      buf_D_oe_cpu_gpustorg += 42;
    }
  }

  //MPI_Barrier(MPI_COMM_WORLD);
  //MPI_Abort(MPI_COMM_WORLD, 911);

  if(g.csw != 0){
    cuda_safe_call( cudaMemcpy(buf_D_oe_gpu, buf_D_oe_cpu_bare, 72*sizeof(cu_cmplx_PRECISION)*nr_DD_sites*in->num_blocks, cudaMemcpyHostToDevice) );
  }
  else{
    cuda_safe_call( cudaMemcpy(buf_D_oe_gpu, buf_D_oe_cpu_bare, 12*sizeof(cu_cmplx_PRECISION)*nr_DD_sites*in->num_blocks, cudaMemcpyHostToDevice) );
  }

  if(g.csw != 0){
    cuda_safe_call( cudaMemcpy(buf_D_oe_gpu_gpustorg, buf_D_oe_cpu_gpustorg_bare, 42*sizeof(cu_cmplx_PRECISION)*nr_DD_sites*in->num_blocks, cudaMemcpyHostToDevice) );
  }
  else{
    // TODO !
    //cuda_safe_call( cudaMemcpy(buf_D_oe_gpu, buf_D_oe_cpu_bare, 12*sizeof(cu_cmplx_PRECISION)*nr_DD_sites*in->num_blocks, cudaMemcpyHostToDevice) );
  }

  free(buf_D_oe_cpu_bare);
  free(buf_D_oe_cpu_gpustorg_bare);

  //cudaMemcpy(&(out->op.oe_clover_vectorized), &buf_D_oe_gpu, sizeof(cu_config_PRECISION*), cudaMemcpyHostToDevice);
  out->op.oe_clover_vectorized = buf_D_oe_gpu;
  out->op.oe_clover_gpustorg = buf_D_oe_gpu_gpustorg;
  //-------------------------------------------------------------------------------------------------------------------------

  int dir, nr_oe_elems_hopp;
  for( dir=0; dir<4; dir++ ){
    (s->s_on_gpu_cpubuff).dir_length_even[dir] = s->dir_length_even[dir];
    (s->s_on_gpu_cpubuff).dir_length_odd[dir] = s->dir_length_odd[dir];
  }
  for( dir=0; dir<4; dir++ ){
    nr_oe_elems_hopp = s->dir_length_even[dir] + s->dir_length_odd[dir];
    cuda_safe_call( cudaMalloc( (void**)&( (s->s_on_gpu_cpubuff).oe_index[dir] ), nr_oe_elems_hopp*sizeof(int) ) );
    cuda_safe_call( cudaMemcpy( (s->s_on_gpu_cpubuff).oe_index[dir], s->oe_index[dir], nr_oe_elems_hopp*sizeof(int), cudaMemcpyHostToDevice ) );
  }

  cuda_safe_call( cudaMalloc( (void**)&( (s->s_on_gpu_cpubuff).op.neighbor_table ), (l->depth==0?4:5)*l->num_inner_lattice_sites*sizeof(int) ) );
  cuda_safe_call( cudaMemcpy( (s->s_on_gpu_cpubuff).op.neighbor_table, s->op.neighbor_table, (l->depth==0?4:5)*l->num_inner_lattice_sites*sizeof(int), cudaMemcpyHostToDevice ) );

  int type=_SCHWARZ, nls, coupling_site_size, nr_elems_D;
  if ( l->depth == 0 ) {
    coupling_site_size = 4*9;
  } else {
    coupling_site_size = 4*l->num_lattice_site_var*l->num_lattice_site_var;
  }  
  nls = (type==_ORDINARY)?l->num_inner_lattice_sites:2*l->num_lattice_sites-l->num_inner_lattice_sites;
  nr_elems_D = coupling_site_size*nls;
  cuda_safe_call( cudaMalloc( (void**) &( (s->s_on_gpu_cpubuff).op.D ), nr_elems_D*sizeof(cu_config_PRECISION) ) );
  cuda_safe_call( cudaMemcpy((s->s_on_gpu_cpubuff).op.D, (cu_config_PRECISION*)(s->op.D), nr_elems_D*sizeof(cu_config_PRECISION), cudaMemcpyHostToDevice) );

  // TODO: remove the following lines for setting GAMMA-related data, and rather put constants within hopping CUDA kernels
  cu_cmplx_PRECISION *gamma_info_vals_buff = (s->s_on_gpu_cpubuff).gamma_info_vals;
  int *gamma_info_coo_buff = (s->s_on_gpu_cpubuff).gamma_info_coo;
  /* even spins - T direction */
  gamma_info_vals_buff[0] = make_cu_cmplx_PRECISION(creal_PRECISION(GAMMA_T_SPIN0_VAL), cimag_PRECISION(GAMMA_T_SPIN0_VAL));
  gamma_info_vals_buff[1] = make_cu_cmplx_PRECISION(creal_PRECISION(GAMMA_T_SPIN2_VAL), cimag_PRECISION(GAMMA_T_SPIN2_VAL));
  /* odd spins */
  gamma_info_vals_buff[2] = make_cu_cmplx_PRECISION(creal_PRECISION(GAMMA_T_SPIN1_VAL), cimag_PRECISION(GAMMA_T_SPIN1_VAL));
  gamma_info_vals_buff[3] = make_cu_cmplx_PRECISION(creal_PRECISION(GAMMA_T_SPIN3_VAL), cimag_PRECISION(GAMMA_T_SPIN3_VAL));
  /* and for the other directions */
  /* Z */
  gamma_info_vals_buff[4] = make_cu_cmplx_PRECISION(creal_PRECISION(GAMMA_Z_SPIN0_VAL), cimag_PRECISION(GAMMA_Z_SPIN0_VAL));
  gamma_info_vals_buff[5] = make_cu_cmplx_PRECISION(creal_PRECISION(GAMMA_Z_SPIN2_VAL), cimag_PRECISION(GAMMA_Z_SPIN2_VAL));
  gamma_info_vals_buff[6] = make_cu_cmplx_PRECISION(creal_PRECISION(GAMMA_Z_SPIN1_VAL), cimag_PRECISION(GAMMA_Z_SPIN1_VAL));
  gamma_info_vals_buff[7] = make_cu_cmplx_PRECISION(creal_PRECISION(GAMMA_Z_SPIN3_VAL), cimag_PRECISION(GAMMA_Z_SPIN3_VAL));
  /* Y */
  gamma_info_vals_buff[8] = make_cu_cmplx_PRECISION(creal_PRECISION(GAMMA_Y_SPIN0_VAL), cimag_PRECISION(GAMMA_Y_SPIN0_VAL));
  gamma_info_vals_buff[9] = make_cu_cmplx_PRECISION(creal_PRECISION(GAMMA_Y_SPIN2_VAL), cimag_PRECISION(GAMMA_Y_SPIN2_VAL));
  gamma_info_vals_buff[10] = make_cu_cmplx_PRECISION(creal_PRECISION(GAMMA_Y_SPIN1_VAL), cimag_PRECISION(GAMMA_Y_SPIN1_VAL));
  gamma_info_vals_buff[11] = make_cu_cmplx_PRECISION(creal_PRECISION(GAMMA_Y_SPIN3_VAL), cimag_PRECISION(GAMMA_Y_SPIN3_VAL));
  /* X */
  gamma_info_vals_buff[12] = make_cu_cmplx_PRECISION(creal_PRECISION(GAMMA_X_SPIN0_VAL), cimag_PRECISION(GAMMA_X_SPIN0_VAL));
  gamma_info_vals_buff[13] = make_cu_cmplx_PRECISION(creal_PRECISION(GAMMA_X_SPIN2_VAL), cimag_PRECISION(GAMMA_X_SPIN2_VAL));
  gamma_info_vals_buff[14] = make_cu_cmplx_PRECISION(creal_PRECISION(GAMMA_X_SPIN1_VAL), cimag_PRECISION(GAMMA_X_SPIN1_VAL));
  gamma_info_vals_buff[15] = make_cu_cmplx_PRECISION(creal_PRECISION(GAMMA_X_SPIN3_VAL), cimag_PRECISION(GAMMA_X_SPIN3_VAL));
  /* even spins - T direction */
  gamma_info_coo_buff[0] = GAMMA_T_SPIN0_CO;
  gamma_info_coo_buff[1] = GAMMA_T_SPIN2_CO;
  /* odd spins */
  gamma_info_coo_buff[2] = GAMMA_T_SPIN1_CO;
  gamma_info_coo_buff[3] = GAMMA_T_SPIN3_CO;
  /* and for the other directions */
  /* Z */
  gamma_info_coo_buff[4] = GAMMA_Z_SPIN0_CO;
  gamma_info_coo_buff[5] = GAMMA_Z_SPIN2_CO;
  gamma_info_coo_buff[6] = GAMMA_Z_SPIN1_CO;
  gamma_info_coo_buff[7] = GAMMA_Z_SPIN3_CO;
  /* Y */
  gamma_info_coo_buff[8] = GAMMA_Y_SPIN0_CO;
  gamma_info_coo_buff[9] = GAMMA_Y_SPIN2_CO;
  gamma_info_coo_buff[10] = GAMMA_Y_SPIN1_CO;
  gamma_info_coo_buff[11] = GAMMA_Y_SPIN3_CO;
  /* X */
  gamma_info_coo_buff[12] = GAMMA_X_SPIN0_CO;
  gamma_info_coo_buff[13] = GAMMA_X_SPIN2_CO;
  gamma_info_coo_buff[14] = GAMMA_X_SPIN1_CO;
  gamma_info_coo_buff[15] = GAMMA_X_SPIN3_CO;

  (s->s_on_gpu_cpubuff).num_block_even_sites = s->num_block_even_sites;
  (s->s_on_gpu_cpubuff).num_block_odd_sites = s->num_block_odd_sites;
  (s->s_on_gpu_cpubuff).block_vector_size = s->block_vector_size;

  // After all the allocations and definitions associated to s->s_on_gpu_cpubuff, it's time to move to the GPU
  cuda_safe_call( cudaMemcpy(s->s_on_gpu, &(s->s_on_gpu_cpubuff), 1*sizeof(schwarz_PRECISION_struct_on_gpu), cudaMemcpyHostToDevice) );

}
#endif


void additive_schwarz_PRECISION( vector_PRECISION phi, vector_PRECISION D_phi, vector_PRECISION eta, const int cycles, int res, 
                                 schwarz_PRECISION_struct *s, level_struct *l, struct Thread *threading ) {
  
  START_NO_HYPERTHREADS(threading)
  
  int k, mu, i, nb = s->num_blocks;
  vector_PRECISION r = s->buf1, Dphi = s->buf4, latest_iter = s->buf2, x = s->buf3, latest_iter2 = s->buf5, swap = NULL;
  void (*block_op)() = (l->depth==0)?block_d_plus_clover_PRECISION:coarse_block_operator_PRECISION,
       (*boundary_op)() = (l->depth==0)?block_PRECISION_boundary_op:coarse_block_PRECISION_boundary_op,
       (*n_boundary_op)() = (l->depth==0)?n_block_PRECISION_boundary_op:n_coarse_block_PRECISION_boundary_op,
       (*block_solve)() = (l->depth==0&&g.odd_even)?block_solve_oddeven_PRECISION:local_minres_PRECISION;

  int nb_thread_start;
  int nb_thread_end;
  compute_core_start_end_custom(0, nb, &nb_thread_start, &nb_thread_end, l, threading, 1);
  
  SYNC_CORES(threading)
  
  if ( res == _NO_RES ) {
    vector_PRECISION_copy( r, eta, nb_thread_start*s->block_vector_size, nb_thread_end*s->block_vector_size, l );
    vector_PRECISION_define( x, 0, nb_thread_start*s->block_vector_size, nb_thread_end*s->block_vector_size, l );
  } else {
    vector_PRECISION_copy( x, phi, nb_thread_start*s->block_vector_size, nb_thread_end*s->block_vector_size, l );
    vector_PRECISION_copy( latest_iter, phi, nb_thread_start*s->block_vector_size, nb_thread_end*s->block_vector_size, l );
  }
  
  START_MASTER(threading)
  if ( res == _NO_RES ) {
    vector_PRECISION_define( x, 0, l->inner_vector_size, l->schwarz_vector_size, l );
  }
  END_MASTER(threading)
  
  SYNC_CORES(threading)
  
  for ( k=0; k<cycles; k++ ) {
    if ( res == _RES ) {
      START_LOCKED_MASTER(threading)
      for ( mu=0; mu<4; mu++ ) {
        ghost_update_PRECISION( latest_iter, mu, +1, &(s->op.c), l );
        ghost_update_PRECISION( latest_iter, mu, -1, &(s->op.c), l );
      }
      END_LOCKED_MASTER(threading)
    } else {
      SYNC_CORES(threading)
    }
      
    for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
      // for all blocks of current color NOT involved in communication
      if ( s->block[i].no_comm ) {
        // calculate block residual
        if ( res == _RES ) {
          if ( k==0 ) {
            block_op( Dphi, latest_iter, s->block[i].start*l->num_lattice_site_var, s, l, no_threading );
            boundary_op( Dphi, latest_iter, i, s, l, no_threading );
            vector_PRECISION_minus( r, eta, Dphi, s->block[i].start*l->num_lattice_site_var,
                                    s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
          } else {
            n_boundary_op( r, latest_iter, i, s, l );
          }
        }
        // local minres updates x, r and latest iter
        block_solve( x, r, latest_iter2, s->block[i].start*l->num_lattice_site_var, s, l, no_threading );
      }
    }
    
    if ( res == _RES ) {
      START_LOCKED_MASTER(threading)
      for ( mu=0; mu<4; mu++ ) {
        ghost_update_wait_PRECISION( latest_iter, mu, +1, &(s->op.c), l );
        ghost_update_wait_PRECISION( latest_iter, mu, -1, &(s->op.c), l );
      }
      END_LOCKED_MASTER(threading)
    } else {
      SYNC_CORES(threading)
    }
      
    for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
      // for all blocks of current color involved in communication
      if ( !s->block[i].no_comm ) {
        // calculate block residual
        if ( res == _RES ) {
          if ( k==0 ) {
            block_op( Dphi, latest_iter, s->block[i].start*l->num_lattice_site_var, s, l, no_threading );
            boundary_op( Dphi, latest_iter, i, s, l, no_threading );
            vector_PRECISION_minus( r, eta, Dphi, s->block[i].start*l->num_lattice_site_var,
                                    s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
          } else {
            n_boundary_op( r, latest_iter, i, s, l );
          }
        }
        // local minres updates x, r and latest iter
        block_solve( x, r, latest_iter2, s->block[i].start*l->num_lattice_site_var, s, l, no_threading );
      }
    }
    res = _RES;
    swap = latest_iter; latest_iter = latest_iter2; latest_iter2 = swap;
  }

  for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
    if ( l->relax_fac != 1.0 )
      vector_PRECISION_scale( phi, x, l->relax_fac, s->block[i].start*l->num_lattice_site_var, s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
    else
      vector_PRECISION_copy( phi, x, s->block[i].start*l->num_lattice_site_var, s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
  }
  
  // calculate D * phi with help of the almost computed residual
  if ( D_phi != NULL ) {
    START_LOCKED_MASTER(threading)
    for ( mu=0; mu<4; mu++ ) {
      ghost_update_PRECISION( latest_iter, mu, +1, &(s->op.c), l );
      ghost_update_PRECISION( latest_iter, mu, -1, &(s->op.c), l );
    }
    END_LOCKED_MASTER(threading)
    
    for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
      if ( s->block[i].no_comm ) {
        n_boundary_op( r, latest_iter, i, s, l );
        vector_PRECISION_minus( D_phi, eta, r,
            s->block[i].start*l->num_lattice_site_var,
            s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
        if ( l->relax_fac != 1.0 )
          vector_PRECISION_scale( D_phi, D_phi, l->relax_fac, s->block[i].start*l->num_lattice_site_var,
              s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
      }
    }
    
    START_LOCKED_MASTER(threading)
    for ( mu=0; mu<4; mu++ ) {
      ghost_update_wait_PRECISION( latest_iter, mu, +1, &(s->op.c), l );
      ghost_update_wait_PRECISION( latest_iter, mu, -1, &(s->op.c), l );
    }
    END_LOCKED_MASTER(threading)
    
    for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
      if ( !s->block[i].no_comm ) {
        n_boundary_op( r, latest_iter, i, s, l );
        vector_PRECISION_minus( D_phi, eta, r,
            s->block[i].start*l->num_lattice_site_var,
            s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
        if ( l->relax_fac != 1.0 )
          vector_PRECISION_scale( D_phi, D_phi, l->relax_fac, s->block[i].start*l->num_lattice_site_var,
              s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
      }
    }
  }
  SYNC_CORES(threading)
  
#ifdef SCHWARZ_RES
  START_LOCKED_MASTER(threading)
  if ( D_phi == NULL ) {
    for ( mu=0; mu<4; mu++ ) {
      ghost_update_PRECISION( latest_iter, mu, +1, &(s->op.c), l );
      ghost_update_PRECISION( latest_iter, mu, -1, &(s->op.c), l );
    }
    
    for ( i=0; i<nb; i++ ) {
      if ( s->block[i].no_comm ) {
        n_boundary_op( r, latest_iter, i, s, l );
      }
    }
    
    for ( mu=0; mu<4; mu++ ) {
      ghost_update_wait_PRECISION( latest_iter, mu, +1, &(s->op.c), l );
      ghost_update_wait_PRECISION( latest_iter, mu, -1, &(s->op.c), l );
    }
    
    for ( i=0; i<nb; i++ ) {
      if ( !s->block[i].no_comm ) {
        n_boundary_op( r, latest_iter, i, s, l );
      }
    }
  }
  double rnorm = global_norm_PRECISION( r, 0, l->inner_vector_size, l, no_threading );
  char number[3]; sprintf( number, "%2d", 31+l->depth ); printf0("\033[1;%2sm|", number );
  printf0(" ---- depth: %d, c: %d, schwarz iter %2d, norm: %11.6le |", l->depth, s->num_colors, k, rnorm );
  printf0("\033[0m\n"); fflush(0);
  END_LOCKED_MASTER(threading)
#endif

  END_NO_HYPERTHREADS(threading)
}


void red_black_schwarz_PRECISION( vector_PRECISION phi, vector_PRECISION D_phi, vector_PRECISION eta, const int cycles, int res,
                                  schwarz_PRECISION_struct *s, level_struct *l, struct Thread *threading ) {
  
  START_NO_HYPERTHREADS(threading)
  
  int k=0, mu, i, init_res = res, res_comm = res, step;
  vector_PRECISION r = s->buf1;
  vector_PRECISION Dphi = s->buf4;
  vector_PRECISION latest_iter = s->buf2;
  vector_PRECISION x = s->buf3;
  void (*block_op)() = (l->depth==0)?block_d_plus_clover_PRECISION:coarse_block_operator_PRECISION,
       (*boundary_op)() = (l->depth==0)?block_PRECISION_boundary_op:coarse_block_PRECISION_boundary_op,
       (*n_boundary_op)() = (l->depth==0)?n_block_PRECISION_boundary_op:n_coarse_block_PRECISION_boundary_op,
       (*block_solve)() = (l->depth==0&&g.odd_even)?block_solve_oddeven_PRECISION:local_minres_PRECISION;
  void (*communicate[2])() = {ghost_update_wait_PRECISION, ghost_update_PRECISION};
  int commdir[8] = {+1,-1,-1,+1,-1,+1,+1,-1};
#ifdef SCHWARZ_RES
  int nb = s->num_blocks;
#endif
       
  SYNC_CORES(threading)
       
  int block_thread_start[8], block_thread_end[8];
  for ( i=0; i<8; i++ )
     compute_core_start_end_custom(0, s->block_list_length[i], block_thread_start+i, block_thread_end+i, l, threading, 1 );
  int start, end;
  compute_core_start_end_custom(0, l->inner_vector_size, &start, &end, l, threading, l->num_lattice_site_var );
  
  if ( res == _NO_RES ) {
    vector_PRECISION_copy( r, eta, start, end, l );
    vector_PRECISION_define( x, 0, start, end, l );
    START_MASTER(threading)
    vector_PRECISION_define( x, 0, l->inner_vector_size, l->schwarz_vector_size, l );
    END_MASTER(threading)
    SYNC_CORES(threading)
  } else {
    vector_PRECISION_copy( x, phi, start, end, l );
    START_LOCKED_MASTER(threading)
    for ( mu=0; mu<4; mu++ )
      ghost_update_PRECISION( x, mu, +1, &(s->op.c), l );
    for ( mu=0; mu<4; mu++ )
      ghost_update_PRECISION( x, mu, -1, &(s->op.c), l );
    END_LOCKED_MASTER(threading)
  }
  
  // perform the Schwarz iteration, solve the block systems
  for ( k=0; k<cycles; k++ ) {
    for ( step=0; step<8; step++ ) {
      for ( i=block_thread_start[step]; i<block_thread_end[step]; i++ ) {
        int index = s->block_list[step][i];
        START_MASTER(threading)
        PROF_PRECISION_START( _SM3 );
        END_MASTER(threading)
        if ( res == _RES ) {
          if ( k==0 && init_res == _RES ) {
            block_op( Dphi, x, s->block[index].start*l->num_lattice_site_var, s, l, no_threading );
            boundary_op( Dphi, x, index, s, l, no_threading );
            vector_PRECISION_minus( r, eta, Dphi, s->block[index].start*l->num_lattice_site_var,
                                    s->block[index].start*l->num_lattice_site_var+s->block_vector_size, l );
          } else {
            n_boundary_op( r, latest_iter, index, s, l );
          }
        }
        START_MASTER(threading)
        PROF_PRECISION_STOP( _SM3, 1 );
        PROF_PRECISION_START( _SM4 );
        END_MASTER(threading)
        // local minres updates x, r and latest iter
        block_solve( x, r, latest_iter, s->block[index].start*l->num_lattice_site_var, s, l, no_threading );
        START_MASTER(threading)
        PROF_PRECISION_STOP( _SM4, 1 );
        END_MASTER(threading)
      }
      
      if ( res_comm == _RES && !(k==cycles-1 && (step==6||step==7) && D_phi==NULL) ) {
        START_LOCKED_MASTER(threading)
        for ( mu=0; mu<4; mu++ ) {
          communicate[(step%4)/2]( (k==0 && step < 6 && init_res == _RES)?x:latest_iter, mu, commdir[step], &(s->op.c), l );
        }
        END_LOCKED_MASTER(threading)
      } else {
        SYNC_CORES(threading)
      }
      
      if ( k==0 && step == 5 ) res = _RES;
      if ( k==0 && step == 1 ) res_comm = _RES;
    }
  }
  
  // copy phi = x
  if ( l->relax_fac != 1.0 )
    vector_PRECISION_scale( phi, x, l->relax_fac, start, end, l );
  else
    vector_PRECISION_copy( phi, x, start, end, l );
  
  // calculate D * phi from r
  if ( D_phi != NULL ) {
    for ( step=4; step<8; step++ ) {
      for ( i=block_thread_start[step]; i<block_thread_end[step]; i++ ) {
        int index = s->block_list[step][i];
        vector_PRECISION_minus( D_phi, eta, r, s->block[index].start*l->num_lattice_site_var,
                                s->block[index].start*l->num_lattice_site_var+s->block_vector_size, l );
        if ( l->relax_fac != 1.0 ) {
          vector_PRECISION_scale( D_phi, D_phi, l->relax_fac, s->block[index].start*l->num_lattice_site_var,
                                  s->block[index].start*l->num_lattice_site_var+s->block_vector_size, l );
        }
      }
    }
    
    for ( step=0; step<4; step++ ) {
      for ( i=block_thread_start[step]; i<block_thread_end[step]; i++ ) {
        int index = s->block_list[step][i];
        
        START_MASTER(threading)
        PROF_PRECISION_START( _SM3 );
        END_MASTER(threading)
        n_boundary_op( r, latest_iter, index, s, l );
        vector_PRECISION_minus( D_phi, eta, r, s->block[index].start*l->num_lattice_site_var,
                                s->block[index].start*l->num_lattice_site_var+s->block_vector_size, l );
        if ( l->relax_fac != 1.0 ) {
          vector_PRECISION_scale( D_phi, D_phi, l->relax_fac, s->block[index].start*l->num_lattice_site_var,
                                  s->block[index].start*l->num_lattice_site_var+s->block_vector_size, l );
        }
        START_MASTER(threading)
        PROF_PRECISION_STOP( _SM3, 1 );
        END_MASTER(threading)
      }
      if ( step == 0 || step == 1 ) {
        START_LOCKED_MASTER(threading)
        for ( mu=0; mu<4; mu++ )
          communicate[0]( latest_iter, mu, commdir[step], &(s->op.c), l );
        END_LOCKED_MASTER(threading)
      } else {
        SYNC_CORES(threading)
      }
    }
  }
  SYNC_CORES(threading)
  
#ifdef SCHWARZ_RES
  START_LOCKED_MASTER(threading)
  if ( D_phi == NULL ) {
    for ( mu=0; mu<4; mu++ ) {
      ghost_update_PRECISION( latest_iter, mu, +1, &(s->op.c), l );
      ghost_update_PRECISION( latest_iter, mu, -1, &(s->op.c), l );
    }
    
    for ( i=0; i<nb; i++ ) {
      if ( s->block[i].no_comm ) {
        n_boundary_op( r, latest_iter, i, s, l );
      }
    }
    
    for ( mu=0; mu<4; mu++ ) {
      ghost_update_wait_PRECISION( latest_iter, mu, +1, &(s->op.c), l );
      ghost_update_wait_PRECISION( latest_iter, mu, -1, &(s->op.c), l );
    }
    
    for ( i=0; i<nb; i++ ) {
      if ( !s->block[i].no_comm ) {
        n_boundary_op( r, latest_iter, i, s, l );
      }
    }
  }
  double rnorm = global_norm_PRECISION( r, 0, l->inner_vector_size, l, no_threading );
  char number[3]; sprintf( number, "%2d", 31+l->depth ); printf0("\033[1;%2sm|", number );
  printf0(" ---- depth: %d, c: %d, schwarz iter %2d, norm: %11.6le |", l->depth, s->num_colors, k, rnorm );
  printf0("\033[0m\n"); fflush(0);
  END_LOCKED_MASTER(threading)
#endif
  END_NO_HYPERTHREADS(threading)
}


void schwarz_PRECISION( vector_PRECISION phi, vector_PRECISION D_phi, vector_PRECISION eta, const int cycles, int res,
                        schwarz_PRECISION_struct *s, level_struct *l, struct Thread *threading ) {
  
  START_NO_HYPERTHREADS(threading)

  int color, k, mu, i,  nb = s->num_blocks, init_res = res;
  vector_PRECISION r = s->buf1;
  vector_PRECISION Dphi = s->buf4;
  vector_PRECISION latest_iter = s->buf2;
  vector_PRECISION x = s->buf3;
  void (*block_op)() = (l->depth==0)?block_d_plus_clover_PRECISION:coarse_block_operator_PRECISION,
       (*boundary_op)() = (l->depth==0)?block_PRECISION_boundary_op:coarse_block_PRECISION_boundary_op,
       (*n_boundary_op)() = (l->depth==0)?n_block_PRECISION_boundary_op:n_coarse_block_PRECISION_boundary_op,
       (*block_solve)() = (l->depth==0&&g.odd_even)?block_solve_oddeven_PRECISION:local_minres_PRECISION;
  
  SYNC_CORES(threading)
  
  int nb_thread_start;
  int nb_thread_end;
  compute_core_start_end_custom(0, nb, &nb_thread_start, &nb_thread_end, l, threading, 1);
  
  if ( res == _NO_RES ) {
    vector_PRECISION_copy( r, eta, nb_thread_start*s->block_vector_size, nb_thread_end*s->block_vector_size, l );
    vector_PRECISION_define( x, 0, nb_thread_start*s->block_vector_size, nb_thread_end*s->block_vector_size, l );
  } else {
    vector_PRECISION_copy( x, phi, nb_thread_start*s->block_vector_size, nb_thread_end*s->block_vector_size, l );
  }
    
  START_MASTER(threading)
  if ( res == _NO_RES ) {
    vector_PRECISION_define( x, 0, l->inner_vector_size, l->schwarz_vector_size, l );
  }
  END_MASTER(threading)
  
  SYNC_CORES(threading)
  
  for ( k=0; k<cycles; k++ ) {
    
    for ( color=0; color<s->num_colors; color++ ) {
      if ( res == _RES ) {
        START_LOCKED_MASTER(threading)
        for ( mu=0; mu<4; mu++ ) {
          ghost_update_PRECISION( (k==0 && init_res == _RES)?x:latest_iter, mu, +1, &(s->op.c), l );
          ghost_update_PRECISION( (k==0 && init_res == _RES)?x:latest_iter, mu, -1, &(s->op.c), l );
        }
        END_LOCKED_MASTER(threading)
      } else {
        // we need a barrier between black and white blocks
        SYNC_CORES(threading)
      }
        
      for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
        // for all blocks of current color NOT involved in communication
        if ( color == s->block[i].color && s->block[i].no_comm ) {
          // calculate block residual
          START_MASTER(threading)
          PROF_PRECISION_START( _SM1 );
          END_MASTER(threading)
          if ( res == _RES ) {
            if ( k==0 && init_res == _RES ) {
              block_op( Dphi, x, s->block[i].start*l->num_lattice_site_var, s, l, no_threading );
              boundary_op( Dphi, x, i, s, l, no_threading );
              vector_PRECISION_minus( r, eta, Dphi, s->block[i].start*l->num_lattice_site_var,
                                      s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
            } else {
              n_boundary_op( r, latest_iter, i, s, l );
            }
          }
          START_MASTER(threading)
          PROF_PRECISION_STOP( _SM1, 1 );
          // local minres updates x, r and latest iter
          PROF_PRECISION_START( _SM2 );
          END_MASTER(threading)
          block_solve( x, r, latest_iter, s->block[i].start*l->num_lattice_site_var, s, l, no_threading );
          START_MASTER(threading)
          PROF_PRECISION_STOP( _SM2, 1 );
          END_MASTER(threading)
        }
      }
      
      if ( res == _RES ) {
        START_LOCKED_MASTER(threading)
        for ( mu=0; mu<4; mu++ ) {
          ghost_update_wait_PRECISION( (k==0 && init_res == _RES)?x:latest_iter, mu, +1, &(s->op.c), l );
          ghost_update_wait_PRECISION( (k==0 && init_res == _RES)?x:latest_iter, mu, -1, &(s->op.c), l );
        }
        END_LOCKED_MASTER(threading)
      } else {
        // we need a barrier between black and white blocks
        SYNC_CORES(threading)
      }
        
      for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
        // for all blocks of current color involved in communication
        if ( color == s->block[i].color && !s->block[i].no_comm ) {
          // calculate block residual
          START_MASTER(threading)
          PROF_PRECISION_START( _SM3 );
          END_MASTER(threading)
          if ( res == _RES ) {
            if ( k==0 && init_res == _RES ) {
              block_op( Dphi, x, s->block[i].start*l->num_lattice_site_var, s, l, no_threading );
              boundary_op( Dphi, x, i, s, l, no_threading );
              vector_PRECISION_minus( r, eta, Dphi, s->block[i].start*l->num_lattice_site_var,
                                      s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
            } else {
              n_boundary_op( r, latest_iter, i, s, l );
            }
          }
          START_MASTER(threading)
          PROF_PRECISION_STOP( _SM3, 1 );
          // local minres updates x, r and latest iter
          PROF_PRECISION_START( _SM4 );
          END_MASTER(threading)
          block_solve( x, r, latest_iter, s->block[i].start*l->num_lattice_site_var, s, l, no_threading );
          START_MASTER(threading)
          PROF_PRECISION_STOP( _SM4, 1 );
          END_MASTER(threading)
        }
      }
      res = _RES;
    }
  }

  for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
    if ( l->relax_fac != 1.0 )
      vector_PRECISION_scale( phi, x, l->relax_fac, s->block[i].start*l->num_lattice_site_var, s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
    else
      vector_PRECISION_copy( phi, x, s->block[i].start*l->num_lattice_site_var, s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
  }
  
  // calculate D * phi with help of the almost computed residual
  // via updating the residual from odd to even
  if ( D_phi != NULL ) {
    START_LOCKED_MASTER(threading)
    for ( mu=0; mu<4; mu++ ) {
      ghost_update_PRECISION( latest_iter, mu, +1, &(s->op.c), l );
      ghost_update_PRECISION( latest_iter, mu, -1, &(s->op.c), l );
    }
    END_LOCKED_MASTER(threading)
    
    for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
      if ( 0 == s->block[i].color && s->block[i].no_comm ) {
        n_boundary_op( r, latest_iter, i, s, l );
        vector_PRECISION_minus( D_phi, eta, r, s->block[i].start*l->num_lattice_site_var,
                                s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
        if ( l->relax_fac != 1.0 ) {
          vector_PRECISION_scale( D_phi, D_phi, l->relax_fac, s->block[i].start*l->num_lattice_site_var,
                                  s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
        }
      }
      if ( 1 == s->block[i].color ) {
        vector_PRECISION_minus( D_phi, eta, r, s->block[i].start*l->num_lattice_site_var,
                                s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
        if ( l->relax_fac != 1.0 ) {
          vector_PRECISION_scale( D_phi, D_phi, l->relax_fac, s->block[i].start*l->num_lattice_site_var,
                                  s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
        }
      }
    }
    
    START_LOCKED_MASTER(threading)
    for ( mu=0; mu<4; mu++ ) {
      ghost_update_wait_PRECISION( latest_iter, mu, +1, &(s->op.c), l );
      ghost_update_wait_PRECISION( latest_iter, mu, -1, &(s->op.c), l );
    }
    END_LOCKED_MASTER(threading)
    
    for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
      if ( 0 == s->block[i].color && !s->block[i].no_comm ) {
        n_boundary_op( r, latest_iter, i, s, l );
        vector_PRECISION_minus( D_phi, eta, r, s->block[i].start*l->num_lattice_site_var,
                                s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
        if ( l->relax_fac != 1.0 ) {
          vector_PRECISION_scale( D_phi, D_phi, l->relax_fac, s->block[i].start*l->num_lattice_site_var,
                                  s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
        }
      }
    }    
  }
  SYNC_CORES(threading)
  
#ifdef SCHWARZ_RES
  START_LOCKED_MASTER(threading)
  if ( D_phi == NULL ) {
    for ( mu=0; mu<4; mu++ ) {
      ghost_update_PRECISION( latest_iter, mu, +1, &(s->op.c), l );
      ghost_update_PRECISION( latest_iter, mu, -1, &(s->op.c), l );
    }
    
    for ( i=0; i<nb; i++ ) {
      if ( s->block[i].no_comm ) {
        n_boundary_op( r, latest_iter, i, s, l );
      }
    }
    
    for ( mu=0; mu<4; mu++ ) {
      ghost_update_wait_PRECISION( latest_iter, mu, +1, &(s->op.c), l );
      ghost_update_wait_PRECISION( latest_iter, mu, -1, &(s->op.c), l );
    }
    
    for ( i=0; i<nb; i++ ) {
      if ( !s->block[i].no_comm ) {
        n_boundary_op( r, latest_iter, i, s, l );
      }
    }
  }
  double rnorm = global_norm_PRECISION( r, 0, l->inner_vector_size, l, no_threading );
  char number[3]; sprintf( number, "%2d", 31+l->depth ); printf0("\033[1;%2sm|", number );
  printf0(" ---- depth: %d, c: %d, schwarz iter %2d, norm: %11.6le |", l->depth, s->num_colors, k, rnorm );
  printf0("\033[0m\n"); fflush(0);
  END_LOCKED_MASTER(threading)
#endif

  END_NO_HYPERTHREADS(threading)
}


// TODO: the following function does the same as schwarz_PRECISION(...), so we can merge them (can we?). For now,
//       work on it as a separate function, and later on unify them

#ifdef CUDA_OPT
void schwarz_PRECISION_CUDA( vector_PRECISION phi, vector_PRECISION D_phi, vector_PRECISION eta, const int cycles, int res,
                             schwarz_PRECISION_struct *s, level_struct *l, struct Thread *threading ) {

  START_NO_HYPERTHREADS(threading)

  int color, k, mu, i,  nb = s->num_blocks, init_res = res, j;
  vector_PRECISION r = s->buf1;
  vector_PRECISION Dphi = s->buf4;
  vector_PRECISION latest_iter = s->buf2;
  vector_PRECISION x = s->buf3;
  void (*block_op)() = (l->depth==0)?block_d_plus_clover_PRECISION:coarse_block_operator_PRECISION,
       (*boundary_op)() = (l->depth==0)?block_PRECISION_boundary_op:coarse_block_PRECISION_boundary_op,
       (*n_boundary_op)() = (l->depth==0)?n_block_PRECISION_boundary_op:n_coarse_block_PRECISION_boundary_op;

  cuda_vector_PRECISION r_dev = (s->cu_s).buf1,
                        x_dev = (s->cu_s).buf3,
                        latest_iter_dev = (s->cu_s).buf2;

  SYNC_CORES(threading)

  int nb_thread_start;
  int nb_thread_end;
  compute_core_start_end_custom(0, nb, &nb_thread_start, &nb_thread_end, l, threading, 1);

  if ( res == _NO_RES ) {
    vector_PRECISION_copy( r, eta, nb_thread_start*s->block_vector_size, nb_thread_end*s->block_vector_size, l );
    vector_PRECISION_define( x, 0, nb_thread_start*s->block_vector_size, nb_thread_end*s->block_vector_size, l );
  } else {
    vector_PRECISION_copy( x, phi, nb_thread_start*s->block_vector_size, nb_thread_end*s->block_vector_size, l );
  }
    
  START_MASTER(threading)
  if ( res == _NO_RES ) {
    vector_PRECISION_define( x, 0, l->inner_vector_size, l->schwarz_vector_size, l );
  }
  END_MASTER(threading)
  
  SYNC_CORES(threading)

  // Creation of CUDA Events, for time measurement and GPU sync
  cudaEvent_t start_event_copy, stop_event_copy, start_event_comp, stop_event_comp;
  cuda_safe_call( cudaEventCreate(&start_event_copy) );
  cuda_safe_call( cudaEventCreate(&stop_event_copy) );
  cuda_safe_call( cudaEventCreate(&start_event_comp) );
  cuda_safe_call( cudaEventCreate(&stop_event_comp) );

  // TODO: generalization to more than one streams pending
  cudaStream_t *streams_schwarz = s->streams;

  // TODO: this will be the RIGHT way of moving data from CPU to GPU. No back-and-forth data movements for block_solve
  //       will be needed. Besides this type of transfers, ghost_update_PRECISION(...) will contain more exchanges
  //if( l->depth==0&&g.odd_even ){
  //  cuda_vector_PRECISION_copy((void*)x_dev, (void*)x, 0, s->num_blocks*s->block_vector_size, l, _H2D, _CUDA_ASYNC, 0, streams_schwarz );
  //  cuda_vector_PRECISION_copy((void*)r_dev, (void*)r, 0, s->num_blocks*s->block_vector_size, l, _H2D, _CUDA_ASYNC, 0, streams_schwarz );
  //  cuda_vector_PRECISION_copy((void*)latest_iter_dev, (void*)latest_iter, 0, s->num_blocks*s->block_vector_size, l, _H2D, _CUDA_ASYNC, 0, streams_schwarz );
  //}

  // TODO: temporary variable, eliminate !
  int do_block_solve_at_cpu;

  // TODO: remove these allocations... they are just for comparing GPU vs CPU results !
  int vs = (l->depth==0)?l->inner_vector_size:l->vector_size;
  vector_PRECISION r_buff=NULL, x_buff=NULL, latest_iter_buff=NULL;
  MALLOC( r_buff, complex_PRECISION, vs+2*l->schwarz_vector_size );
  latest_iter_buff = r_buff + vs;
  x_buff = latest_iter_buff + l->schwarz_vector_size;
  if ( res == _NO_RES ) {
    vector_PRECISION_copy( r_buff, eta, nb_thread_start*s->block_vector_size, nb_thread_end*s->block_vector_size, l );
    vector_PRECISION_define( x_buff, 0, nb_thread_start*s->block_vector_size, nb_thread_end*s->block_vector_size, l );
  } else {
    vector_PRECISION_copy( x_buff, phi, nb_thread_start*s->block_vector_size, nb_thread_end*s->block_vector_size, l );
  }
  START_MASTER(threading)
  if ( res == _NO_RES ) {
    vector_PRECISION_define( x_buff, 0, l->inner_vector_size, l->schwarz_vector_size, l );
  }
  END_MASTER(threading)
  SYNC_CORES(threading)

  // TODO: remove !
  printf("\x1B[0m\n");

  for ( k=0; k<cycles; k++ ) {

    for ( color=0; color<s->num_colors; color++ ) {

      // Start whole iteration
      //cuda_safe_call( cudaEventRecord(start_event, 0) );

      if ( res == _RES ) {
        START_LOCKED_MASTER(threading)
        for ( mu=0; mu<4; mu++ ) {
          ghost_update_PRECISION( (k==0 && init_res == _RES)?x:latest_iter, mu, +1, &(s->op.c), l );
          ghost_update_PRECISION( (k==0 && init_res == _RES)?x:latest_iter, mu, -1, &(s->op.c), l );
        }
        END_LOCKED_MASTER(threading)
      } else {
        // we need a barrier between black and white blocks
        SYNC_CORES(threading)
      }

      // Boundary operations
      for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
        // for all blocks of current color NOT involved in communication
        if ( color == s->block[i].color && s->block[i].no_comm ) {
          // calculate block residual
          // TODO: move these time measurement lines to the appropriate place !
          //START_MASTER(threading)
          //PROF_PRECISION_START( _SM1 );
          //END_MASTER(threading)
          if ( res == _RES ) {
            if ( k==0 && init_res == _RES ) {
              block_op( Dphi, x, s->block[i].start*l->num_lattice_site_var, s, l, no_threading );
              boundary_op( Dphi, x, i, s, l, no_threading );
              vector_PRECISION_minus( r, eta, Dphi, s->block[i].start*l->num_lattice_site_var,
                                      s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
            } else {
              n_boundary_op( r, latest_iter, i, s, l );
            }
          }
          // TODO: move these time measurement lines to the appropriate place !
          //START_MASTER(threading)
          //PROF_PRECISION_STOP( _SM1, 1 );
        }
      }

      // local minres updates x, r and latest iter
      // TODO: move these time measurement lines to the appropriate place !
      //PROF_PRECISION_START( _SM2 );
      //END_MASTER(threading)

      // TODO: remove ! (this line helps restricting to proc=0)
      if( g.my_rank==0 ){

      // Tmp aux vars for time measurements
      struct timeval start, end;
      long start_us, end_us;
      float time_x_copy, time_x_comp;

      gettimeofday(&start, NULL);
      // Copy to the GPU information needed for block_solve
      cuda_safe_call( cudaEventRecord(start_event_copy, streams_schwarz[0]) );
      // Copy blocks information from CPU to GPU
      for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
        // for all blocks of current color NOT involved in communication
        if ( color == s->block[i].color && s->block[i].no_comm ) {
          cuda_vector_PRECISION_copy((void*)x_dev, (void*)x, s->block[i].start*l->num_lattice_site_var, s->block_vector_size, l, _H2D, _CUDA_SYNC, 0, streams_schwarz );
          cuda_vector_PRECISION_copy((void*)r_dev, (void*)r, s->block[i].start*l->num_lattice_site_var, s->block_vector_size, l, _H2D, _CUDA_SYNC, 0, streams_schwarz );
          cuda_vector_PRECISION_copy((void*)latest_iter_dev, (void*)latest_iter, s->block[i].start*l->num_lattice_site_var, s->block_vector_size, l, _H2D, _CUDA_SYNC, 0, streams_schwarz );
        }
      }
      cuda_safe_call( cudaEventRecord(stop_event_copy, streams_schwarz[0]) );
      cuda_safe_call( cudaEventSynchronize(stop_event_copy) );
      gettimeofday(&end, NULL);
      start_us = start.tv_sec * (int)1e6 + start.tv_usec;
      end_us = end.tv_sec * (int)1e6 + end.tv_usec;
      printf("\nTime (in us) for CPU-to-GPU copy (according to gettimeofday, contains CUDA Events records): %ld\n",
            (end_us-start_us));
      cuda_safe_call( cudaEventElapsedTime(&time_x_copy, start_event_copy, stop_event_copy) );
      printf("Time (in us) for CPU-to-GPU copy (according to CUDA Events):  %f\n", 1000*time_x_copy);

      // TODO: remove eventually...
      cuda_safe_call( cudaDeviceSynchronize() );

      gettimeofday(&start, NULL);
      cuda_safe_call( cudaEventRecord(start_event_comp, streams_schwarz[0]) );
      // FIXME: no need for i anymore !
      if( l->depth==0&&g.odd_even ){

        i=0;
        do_block_solve_at_cpu=0;
        //int nr_DD_blocks_to_compute = (nb_thread_end-nb_thread_start);
        cuda_block_solve_oddeven_PRECISION( (cuda_vector_PRECISION)x_dev, (cuda_vector_PRECISION)r_dev, (cuda_vector_PRECISION)latest_iter_dev,
                                            0, s->nr_DD_blocks_notin_comms[color], s, l, no_threading, 0, streams_schwarz, do_block_solve_at_cpu, color,
                                            (s->cu_s).DD_blocks_notin_comms[color], s->DD_blocks_notin_comms[color] );
      }
      else{
        // block_solve on GPU
        for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
          // for all blocks of current color NOT involved in communication
          if ( color == s->block[i].color && s->block[i].no_comm ) {
            local_minres_PRECISION( x, r, latest_iter, s->block[i].start*l->num_lattice_site_var, s, l, no_threading );
          }
        }
      }
      cuda_safe_call( cudaEventRecord(stop_event_comp, streams_schwarz[0]) );
      cuda_safe_call( cudaEventSynchronize(stop_event_comp) );
      gettimeofday(&end, NULL);
      start_us = start.tv_sec * (int)1e6 + start.tv_usec;
      end_us = end.tv_sec * (int)1e6 + end.tv_usec;
      printf("\nTime (in us) for block solve @ GPU (according to gettimeofday, contains CUDA Events records): %ld\n",
             (end_us-start_us));
      cuda_safe_call( cudaEventElapsedTime(&time_x_comp, start_event_comp, stop_event_comp) );
      printf("Time (in us) for block solve @ GPU (according to CUDA Events):  %f\n", 1000*time_x_comp);

      // TODO: remove eventually...
      cuda_safe_call( cudaDeviceSynchronize() );

      gettimeofday(&start, NULL);
      // Copy to the GPU information needed for block_solve
      cuda_safe_call( cudaEventRecord(start_event_copy, streams_schwarz[0]) );
      // Copy blocks information from CPU to GPU
      for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
        // for all blocks of current color NOT involved in communication
        if ( color == s->block[i].color && s->block[i].no_comm ) {
          cuda_vector_PRECISION_copy((void*)x, (void*)x_dev, s->block[i].start*l->num_lattice_site_var, s->block_vector_size, l, _D2H, _CUDA_SYNC, 0, streams_schwarz );
          cuda_vector_PRECISION_copy((void*)r, (void*)r_dev, s->block[i].start*l->num_lattice_site_var, s->block_vector_size, l, _D2H, _CUDA_SYNC, 0, streams_schwarz );
          cuda_vector_PRECISION_copy((void*)latest_iter, (void*)latest_iter_dev, s->block[i].start*l->num_lattice_site_var, s->block_vector_size, l, _D2H, _CUDA_SYNC, 0, streams_schwarz );
        }
      }
      cuda_safe_call( cudaEventRecord(stop_event_copy, streams_schwarz[0]) );
      cuda_safe_call( cudaEventSynchronize(stop_event_copy) );
      gettimeofday(&end, NULL);
      start_us = start.tv_sec * (int)1e6 + start.tv_usec;
      end_us = end.tv_sec * (int)1e6 + end.tv_usec;
      printf("\nTime (in us) for GPU-to-CPU copy (according to gettimeofday, contains CUDA Events records): %ld\n",
            (end_us-start_us));
      cuda_safe_call( cudaEventElapsedTime(&time_x_copy, start_event_copy, stop_event_copy) );
      printf("Time (in us) for GPU-to-CPU copy (according to CUDA Events):  %f\n", 1000*time_x_copy);

      // TODO: remove eventually...
      cuda_safe_call( cudaDeviceSynchronize() );

      gettimeofday(&start, NULL);
      // block_solve on CPU
      for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
        // for all blocks of current color NOT involved in communication
        if ( color == s->block[i].color && s->block[i].no_comm ) {
          do_block_solve_at_cpu=1;
          if( l->depth==0&&g.odd_even ){
            cuda_block_solve_oddeven_PRECISION( (cuda_vector_PRECISION)x_buff, (cuda_vector_PRECISION)r_buff, (cuda_vector_PRECISION)latest_iter_buff,
                                                s->block[i].start*l->num_lattice_site_var, 1, s, l, no_threading, 0, streams_schwarz, do_block_solve_at_cpu, color,
                                                s->DD_blocks_notin_comms[color], s->DD_blocks_notin_comms[color] );
          } else {
            local_minres_PRECISION( x_buff, r_buff, latest_iter_buff, s->block[i].start*l->num_lattice_site_var, s, l, no_threading );
          }

        }
      }
      gettimeofday(&end, NULL);
      start_us = start.tv_sec * (int)1e6 + start.tv_usec;
      end_us = end.tv_sec * (int)1e6 + end.tv_usec;
      printf("\nIMPORTANT TO NOTE: the time displayed here for block_solve might be lower than usual, due to pre-cached data...\n");
      printf("Time (in us) for block solve @ CPU (according to gettimeofday): %ld\n",
             (end_us-start_us));


      int comp_bool1, comp_bool2;
      float comp_tol = 1.0e-4;

      //printf("\n\n");
      for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
        // for all blocks of current color NOT involved in communication
        if ( color == s->block[i].color && s->block[i].no_comm ) {

          for(j=0; j<s->block_vector_size; j++){

              comp_bool1 = fabs( creal_PRECISION(latest_iter[s->block[i].start*l->num_lattice_site_var+j]) - creal_PRECISION(latest_iter_buff[s->block[i].start*l->num_lattice_site_var+j]) ) < comp_tol;
              comp_bool2 = fabs( cimag_PRECISION(latest_iter[s->block[i].start*l->num_lattice_site_var+j]) - cimag_PRECISION(latest_iter_buff[s->block[i].start*l->num_lattice_site_var+j]) ) < comp_tol;

              if( !comp_bool1 || !comp_bool2 ){
                printf("i=%d, %f + i%f , %f + i%f\n", i, creal_PRECISION(latest_iter[s->block[i].start*l->num_lattice_site_var+j]), cimag_PRECISION(latest_iter[s->block[i].start*l->num_lattice_site_var+j]),
                                                        creal_PRECISION(latest_iter_buff[s->block[i].start*l->num_lattice_site_var+j]), cimag_PRECISION(latest_iter_buff[s->block[i].start*l->num_lattice_site_var+j]));
              }
          }

        }
      }

      //printf("\n\n");
      for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
        // for all blocks of current color NOT involved in communication
        if ( color == s->block[i].color && s->block[i].no_comm ) {

          for(j=0; j<s->block_vector_size; j++){

              comp_bool1 = fabs( creal_PRECISION(x[s->block[i].start*l->num_lattice_site_var+j]) - creal_PRECISION(x_buff[s->block[i].start*l->num_lattice_site_var+j]) ) < comp_tol;
              comp_bool2 = fabs( cimag_PRECISION(x[s->block[i].start*l->num_lattice_site_var+j]) - cimag_PRECISION(x_buff[s->block[i].start*l->num_lattice_site_var+j]) ) < comp_tol;

              if( !comp_bool1 || !comp_bool2 ){
                printf("i=%d, %f + i%f , %f + i%f\n", i, creal_PRECISION(x[s->block[i].start*l->num_lattice_site_var+j]), cimag_PRECISION(x[s->block[i].start*l->num_lattice_site_var+j]),
                                                        creal_PRECISION(x_buff[s->block[i].start*l->num_lattice_site_var+j]), cimag_PRECISION(x_buff[s->block[i].start*l->num_lattice_site_var+j]));
              }
          }

        }
      }

      //printf("\n\n");
      for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
        // for all blocks of current color NOT involved in communication
        if ( color == s->block[i].color && s->block[i].no_comm ) {

          for(j=0; j<s->block_vector_size; j++){

              comp_bool1 = fabs( creal_PRECISION(r[s->block[i].start*l->num_lattice_site_var+j]) - creal_PRECISION(r_buff[s->block[i].start*l->num_lattice_site_var+j]) ) < comp_tol;
              comp_bool2 = fabs( cimag_PRECISION(r[s->block[i].start*l->num_lattice_site_var+j]) - cimag_PRECISION(r_buff[s->block[i].start*l->num_lattice_site_var+j]) ) < comp_tol;

              if( !comp_bool1 || !comp_bool2 ){
                printf("i=%d, %f + i%f , %f + i%f\n", i, creal_PRECISION(r[s->block[i].start*l->num_lattice_site_var+j]), cimag_PRECISION(r[s->block[i].start*l->num_lattice_site_var+j]),
                                                        creal_PRECISION(r_buff[s->block[i].start*l->num_lattice_site_var+j]), cimag_PRECISION(r_buff[s->block[i].start*l->num_lattice_site_var+j]));
              }
          }

        }
      }

      // TODO: remove ! (this line helps restricting to proc=0)
      }

      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Finalize();
      exit(0);

      // TODO: move these time measurement lines to the appropriate place !
      //START_MASTER(threading)
      //PROF_PRECISION_STOP( _SM2, 1 );
      //END_MASTER(threading)

      // local minres updates x, r and latest iter
      // TODO: move these time measurement lines to the appropriate place !
      //PROF_PRECISION_START( _SM4 );
      //END_MASTER(threading)
      //START_MASTER(threading)
      //PROF_PRECISION_STOP( _SM4, 1 );
      //END_MASTER(threading)
      //}

      if ( res == _RES ) {
        START_LOCKED_MASTER(threading)
        for ( mu=0; mu<4; mu++ ) {
          ghost_update_wait_PRECISION( (k==0 && init_res == _RES)?x:latest_iter, mu, +1, &(s->op.c), l );
          ghost_update_wait_PRECISION( (k==0 && init_res == _RES)?x:latest_iter, mu, -1, &(s->op.c), l );
        }
        END_LOCKED_MASTER(threading)
      } else {
        // we need a barrier between black and white blocks
        SYNC_CORES(threading)
      }

      for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
        // for all blocks of current color involved in communication
        if ( color == s->block[i].color && !s->block[i].no_comm ) {
          // calculate block residual
          // TODO: move these time measurement lines to the appropriate place !
          //START_MASTER(threading)
          //PROF_PRECISION_START( _SM3 );
          //END_MASTER(threading)
          if ( res == _RES ) {
            if ( k==0 && init_res == _RES ) {
              block_op( Dphi, x, s->block[i].start*l->num_lattice_site_var, s, l, no_threading );
              boundary_op( Dphi, x, i, s, l, no_threading );
              vector_PRECISION_minus( r, eta, Dphi, s->block[i].start*l->num_lattice_site_var,
                                      s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
            } else {
              n_boundary_op( r, latest_iter, i, s, l );
            }
          }
          // TODO: move these time measurement lines to the appropriate place !
          //START_MASTER(threading)
          //PROF_PRECISION_STOP( _SM3, 1 );
        }
      }

      // TODO: adjust the code from above (NO COMMS section) here. Mainly, the main adjustment
      //       will be to switch s->block[i].no_comm to !s->block[i].no_comm

      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD, 911);

      /*
      // TODO: remove ! (this line helps restricting to proc=0)
      if( g.my_rank==0 ){

      // Tmp aux vars for time measurements
      struct timeval start, end;
      long start_us, end_us;
      float time_x_copy, time_x_comp;

      gettimeofday(&start, NULL);
      // Copy to the GPU information needed for block_solve
      cuda_safe_call( cudaEventRecord(start_event_copy, streams_schwarz[0]) );
      // Copy blocks information from CPU to GPU
      for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
        // for all blocks of current color NOT involved in communication
        if ( color == s->block[i].color && !s->block[i].no_comm ) {
          cuda_vector_PRECISION_copy((void*)x_dev, (void*)x, s->block[i].start*l->num_lattice_site_var, s->block_vector_size, l, _H2D, _CUDA_SYNC, 0, streams_schwarz );
          cuda_vector_PRECISION_copy((void*)r_dev, (void*)r, s->block[i].start*l->num_lattice_site_var, s->block_vector_size, l, _H2D, _CUDA_SYNC, 0, streams_schwarz );
          cuda_vector_PRECISION_copy((void*)latest_iter_dev, (void*)latest_iter, s->block[i].start*l->num_lattice_site_var, s->block_vector_size, l, _H2D, _CUDA_SYNC, 0, streams_schwarz );
        }
      }
      cuda_safe_call( cudaEventRecord(stop_event_copy, streams_schwarz[0]) );
      cuda_safe_call( cudaEventSynchronize(stop_event_copy) );
      gettimeofday(&end, NULL);
      start_us = start.tv_sec * (int)1e6 + start.tv_usec;
      end_us = end.tv_sec * (int)1e6 + end.tv_usec;
      printf("\nTime (in us) for GPU copy (according to gettimeofday, contains CUDA Events records): %ld\n",
            (end_us-start_us));
      cuda_safe_call( cudaEventElapsedTime(&time_x_copy, start_event_copy, stop_event_copy) );
      printf("Time (in us) for GPU copy (according to CUDA Events):  %f\n", 1000*time_x_copy);


      // TODO: remove eventually...
      cuda_safe_call( cudaDeviceSynchronize() );


      gettimeofday(&start, NULL);
      cuda_safe_call( cudaEventRecord(start_event_comp, streams_schwarz[0]) );
      // block_solve on GPU
      for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
        // for all blocks of current color NOT involved in communication
        if ( color == s->block[i].color && !s->block[i].no_comm ) {
          do_block_solve_at_cpu=0;
          cuda_block_solve_oddeven_PRECISION( (cuda_vector_PRECISION)x, (cuda_vector_PRECISION)r, (cuda_vector_PRECISION)latest_iter,
                                              s->block[i].start*l->num_lattice_site_var, 1, s, l, no_threading, 0, streams_schwarz, do_block_solve_at_cpu, color,
                                              s->DD_blocks_in_comms[color] );
        }
      }
      cuda_safe_call( cudaEventRecord(stop_event_comp, streams_schwarz[0]) );
      cuda_safe_call( cudaEventSynchronize(stop_event_comp) );
      gettimeofday(&end, NULL);
      start_us = start.tv_sec * (int)1e6 + start.tv_usec;
      end_us = end.tv_sec * (int)1e6 + end.tv_usec;
      printf("Time (in us) for block solve @ GPU (according to gettimeofday, contains CUDA Events records): %ld\n",
             (end_us-start_us));
      cuda_safe_call( cudaEventElapsedTime(&time_x_comp, start_event_comp, stop_event_comp) );
      printf("Time (in us) for block solve @ GPU (according to CUDA Events):  %f\n", 1000*time_x_comp);


      gettimeofday(&start, NULL);
      // block_solve on CPU
      for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
        // for all blocks of current color NOT involved in communication
        if ( color == s->block[i].color && !s->block[i].no_comm ) {
          do_block_solve_at_cpu=1;
          if( l->depth==0&&g.odd_even ){
            cuda_block_solve_oddeven_PRECISION( (cuda_vector_PRECISION)x, (cuda_vector_PRECISION)r, (cuda_vector_PRECISION)latest_iter,
                                                s->block[i].start*l->num_lattice_site_var, 1, s, l, no_threading, 0, streams_schwarz, do_block_solve_at_cpu, color,
                                                s->DD_blocks_in_comms[color] );
          } else {
            local_minres_PRECISION( x, r, latest_iter, s->block[i].start*l->num_lattice_site_var, s, l, no_threading );
          }

        }
      }
      gettimeofday(&end, NULL);
      start_us = start.tv_sec * (int)1e6 + start.tv_usec;
      end_us = end.tv_sec * (int)1e6 + end.tv_usec;
      printf("Time (in us) for block solve @ CPU (according to gettimeofday): %ld\n",
             (end_us-start_us));


      // TODO: remove ! (this line helps restricting to proc=0)
      }
      */

      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Finalize();
      exit(0);

      // Retrieve back from GPU to CPU
      // TODO: change this to, correspondigly as above, copy only the portions of x and r needed by the block_solve
      //cuda_vector_PRECISION_copy((void*)x, (void*)x_dev, s->block[i].start*l->num_lattice_site_var, s->block_vector_size, l, _D2H, _CUDA_ASYNC, 0, streams_schwarz );
      //cuda_vector_PRECISION_copy((void*)r, (void*)r_dev, s->block[i].start*l->num_lattice_site_var, s->block_vector_size, l, _D2H, _CUDA_ASYNC, 0, streams_schwarz );
      //cuda_vector_PRECISION_copy((void*)latest_iter, (void*)latest_iter_dev, s->block[i].start*l->num_lattice_site_var, s->block_vector_size, l, _D2H, _CUDA_ASYNC, 0, streams_schwarz );

      // TODO: move these time measurement lines to the appropriate place !
      //START_MASTER(threading)
      //PROF_PRECISION_STOP( _SM2, 1 );
      //END_MASTER(threading)

      // local minres updates x, r and latest iter
      // TODO: move these time measurement lines to the appropriate place !
      //PROF_PRECISION_START( _SM4 );
      //END_MASTER(threading)
      //START_MASTER(threading)
      //PROF_PRECISION_STOP( _SM4, 1 );
      //END_MASTER(threading)
      //}

      res = _RES;

      // End whole iteration's tracking
      //cuda_safe_call( cudaEventRecord(stop_event, 0) );

      // Sync whole SAP
      //cuda_safe_call( cudaEventSynchronize(stop_event) );

    }
  }

  //if( l->depth==0&&g.odd_even ){
  //  cuda_vector_PRECISION_copy((void*)x, (void*)x_dev, 0, s->num_blocks*s->block_vector_size, l, _D2H, _CUDA_ASYNC, 0, streams_schwarz );
  //  cuda_vector_PRECISION_copy((void*)r, (void*)r_dev, 0, s->num_blocks*s->block_vector_size, l, _D2H, _CUDA_ASYNC, 0, streams_schwarz );
  //  cuda_vector_PRECISION_copy((void*)latest_iter, (void*)latest_iter_dev, 0, s->num_blocks*s->block_vector_size, l, _D2H, _CUDA_ASYNC, 0, streams_schwarz );
  //}

  /*
  // Saving vector to output file, for correctness comparisons
  int total_nr_vectentries = s->num_blocks * s->block_vector_size;
  if(g.my_rank == 0){
    printf("%d\n", total_nr_vectentries);
    printf("Saving vector into file and exiting...\n");
    field_saver( eta, total_nr_vectentries, "PRECISION", "/home/ramirez/DDalphaAMG/output_xrl_orig.txt" );
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Abort(MPI_COMM_WORLD, 911);
  */

  for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
    if ( l->relax_fac != 1.0 )
      vector_PRECISION_scale( phi, x, l->relax_fac, s->block[i].start*l->num_lattice_site_var, s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
    else
      vector_PRECISION_copy( phi, x, s->block[i].start*l->num_lattice_site_var, s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
  }
  
  // calculate D * phi with help of the almost computed residual
  // via updating the residual from odd to even
  if ( D_phi != NULL ) {
    START_LOCKED_MASTER(threading)
    for ( mu=0; mu<4; mu++ ) {
      ghost_update_PRECISION( latest_iter, mu, +1, &(s->op.c), l );
      ghost_update_PRECISION( latest_iter, mu, -1, &(s->op.c), l );
    }
    END_LOCKED_MASTER(threading)
    
    for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
      if ( 0 == s->block[i].color && s->block[i].no_comm ) {
        n_boundary_op( r, latest_iter, i, s, l );
        vector_PRECISION_minus( D_phi, eta, r, s->block[i].start*l->num_lattice_site_var,
                                s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
        if ( l->relax_fac != 1.0 ) {
          vector_PRECISION_scale( D_phi, D_phi, l->relax_fac, s->block[i].start*l->num_lattice_site_var,
                                  s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
        }
      }
      if ( 1 == s->block[i].color ) {
        vector_PRECISION_minus( D_phi, eta, r, s->block[i].start*l->num_lattice_site_var,
                                s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
        if ( l->relax_fac != 1.0 ) {
          vector_PRECISION_scale( D_phi, D_phi, l->relax_fac, s->block[i].start*l->num_lattice_site_var,
                                  s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
        }
      }
    }
    
    START_LOCKED_MASTER(threading)
    for ( mu=0; mu<4; mu++ ) {
      ghost_update_wait_PRECISION( latest_iter, mu, +1, &(s->op.c), l );
      ghost_update_wait_PRECISION( latest_iter, mu, -1, &(s->op.c), l );
    }
    END_LOCKED_MASTER(threading)
    
    for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
      if ( 0 == s->block[i].color && !s->block[i].no_comm ) {
        n_boundary_op( r, latest_iter, i, s, l );
        vector_PRECISION_minus( D_phi, eta, r, s->block[i].start*l->num_lattice_site_var,
                                s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
        if ( l->relax_fac != 1.0 ) {
          vector_PRECISION_scale( D_phi, D_phi, l->relax_fac, s->block[i].start*l->num_lattice_site_var,
                                  s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
        }
      }
    }
  }
  SYNC_CORES(threading)
  
#ifdef SCHWARZ_RES
  START_LOCKED_MASTER(threading)
  if ( D_phi == NULL ) {
    for ( mu=0; mu<4; mu++ ) {
      ghost_update_PRECISION( latest_iter, mu, +1, &(s->op.c), l );
      ghost_update_PRECISION( latest_iter, mu, -1, &(s->op.c), l );
    }

    for ( i=0; i<nb; i++ ) {
      if ( s->block[i].no_comm ) {
        n_boundary_op( r, latest_iter, i, s, l );
      }
    }
    
    for ( mu=0; mu<4; mu++ ) {
      ghost_update_wait_PRECISION( latest_iter, mu, +1, &(s->op.c), l );
      ghost_update_wait_PRECISION( latest_iter, mu, -1, &(s->op.c), l );
    }
    
    for ( i=0; i<nb; i++ ) {
      if ( !s->block[i].no_comm ) {
        n_boundary_op( r, latest_iter, i, s, l );
      }
    }
  }
  double rnorm = global_norm_PRECISION( r, 0, l->inner_vector_size, l, no_threading );
  char number[3]; sprintf( number, "%2d", 31+l->depth ); printf0("\033[1;%2sm|", number );
  printf0(" ---- depth: %d, c: %d, schwarz iter %2d, norm: %11.6le |", l->depth, s->num_colors, k, rnorm );
  printf0("\033[0m\n"); fflush(0);
  END_LOCKED_MASTER(threading)
#endif

  END_NO_HYPERTHREADS(threading)
}
#endif


void sixteen_color_schwarz_PRECISION( vector_PRECISION phi, vector_PRECISION D_phi, vector_PRECISION eta, const int cycles, int res, 
                                      schwarz_PRECISION_struct *s, level_struct *l, struct Thread *threading ) {
  
  START_NO_HYPERTHREADS(threading)
  ASSERT( D_phi == NULL );
  
  if ( s->num_colors == 2 ) schwarz_PRECISION( phi, D_phi, eta, cycles, res, s, l, no_threading );
  else {
    int color, k, mu, i, nb = s->num_blocks;
    vector_PRECISION r = s->buf1, Dphi = s->buf4, latest_iter = s->buf2, x = s->buf3;
    void (*block_op)() = (l->depth==0)?block_d_plus_clover_PRECISION:coarse_block_operator_PRECISION,
        (*boundary_op)() = (l->depth==0)?block_PRECISION_boundary_op:coarse_block_PRECISION_boundary_op,
        (*n_boundary_op)() = (l->depth==0)?n_block_PRECISION_boundary_op:n_coarse_block_PRECISION_boundary_op,
        (*block_solve)() = (l->depth==0&&g.odd_even)?block_solve_oddeven_PRECISION:local_minres_PRECISION;
        
    int color_to_comm[16][2] = { {T,-1}, {X,+1}, {Y,+1}, {X,-1}, {Z,+1}, {Y,-1}, {X,+1}, {Y,+1},
                                {T,+1}, {X,-1}, {Y,-1}, {X,+1}, {Z,-1}, {Y,+1}, {X,-1}, {Y,-1}  };

    int nb_thread_start;
    int nb_thread_end;
    compute_core_start_end_custom(0, nb, &nb_thread_start, &nb_thread_end, l, threading, 1);
    
    SYNC_CORES(threading)
    
    if ( res == _NO_RES ) {
      vector_PRECISION_copy( r, eta, nb_thread_start*s->block_vector_size, nb_thread_end*s->block_vector_size, l );
      vector_PRECISION_define( x, 0, nb_thread_start*s->block_vector_size, nb_thread_end*s->block_vector_size, l );
    } else {
      vector_PRECISION_copy( x, phi, nb_thread_start*s->block_vector_size, nb_thread_end*s->block_vector_size, l );
    }
    
    START_MASTER(threading)
    if ( res == _NO_RES ) {
      vector_PRECISION_define( x, 0, l->inner_vector_size, l->schwarz_vector_size, l );
    }
    END_MASTER(threading)
    
    SYNC_CORES(threading)
    
    for ( k=0; k<cycles; k++ ) {
      for ( color=0; color<s->num_colors; color++ ) {
        
        // comm start
        if ( res == _RES ) {
          START_LOCKED_MASTER(threading)
          ghost_update_PRECISION( k==0?x:latest_iter, color_to_comm[color][0], color_to_comm[color][1], &(s->op.c), l );
          if ( color == 0 && (k==0 || k==1) ) {
            for ( mu=1; mu<4; mu++ ) {
              ghost_update_PRECISION( k==0?x:latest_iter, mu, -1, &(s->op.c), l );
            }
          }
          END_LOCKED_MASTER(threading)
        } else {
          SYNC_CORES(threading)
        }
        
        // blocks which have their latest neighbor information available
        for ( i=nb_thread_start; i<nb_thread_end; i++ ) {          
          if (  color == s->block[i].color && s->block[i].no_comm ) {
            if ( res == _RES ) {
              if ( k==0 ) {
                block_op( Dphi, x, s->block[i].start*l->num_lattice_site_var, s, l, no_threading );
                boundary_op( Dphi, x, i, s, l, no_threading );
                vector_PRECISION_minus( r, eta, Dphi, s->block[i].start*l->num_lattice_site_var,
                                        s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
              } else {
                n_boundary_op( r, latest_iter, i, s, l );
              }
            }
            // local minres updates x, r and latest iter
            block_solve( x, r, latest_iter, s->block[i].start*l->num_lattice_site_var, s, l, no_threading );
          }
        }
        
        // comm wait
        if ( res == _RES ) {
          START_LOCKED_MASTER(threading)
          ghost_update_wait_PRECISION( k==0?x:latest_iter, color_to_comm[color][0], color_to_comm[color][1], &(s->op.c), l );
          if ( color == 0 && (k==0 || k==1) ) {
            for ( mu=1; mu<4; mu++ ) {
              ghost_update_wait_PRECISION( k==0?x:latest_iter, mu, -1, &(s->op.c), l );
            }
          }
          END_LOCKED_MASTER(threading)
        } else {
          SYNC_CORES(threading)
        }
        
        // blocks which require certain ghost cell updates
        for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
          if ( color == s->block[i].color && !s->block[i].no_comm ) {
            if ( res == _RES ) {
              if ( k==0 ) {
                block_op( Dphi, x, s->block[i].start*l->num_lattice_site_var, s, l, no_threading );
                boundary_op( Dphi, x, i, s, l, no_threading );
                vector_PRECISION_minus( r, eta, Dphi, s->block[i].start*l->num_lattice_site_var,
                                        s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
              } else {
                n_boundary_op( r, latest_iter, i, s, l );
              }
            }
            // local minres updates x, r and latest iter
            block_solve( x, r, latest_iter, s->block[i].start*l->num_lattice_site_var, s, l, no_threading );
          }
        }
             
        res = _RES;
      }
    }

    SYNC_CORES(threading)
    
    if ( l->relax_fac != 1.0 )
      vector_PRECISION_scale( phi, x, l->relax_fac, nb_thread_start*s->block_vector_size, nb_thread_end*s->block_vector_size, l );
    else
      vector_PRECISION_copy( phi, x, nb_thread_start*s->block_vector_size, nb_thread_end*s->block_vector_size, l );
    
    SYNC_CORES(threading)
    
#ifdef SCHWARZ_RES
    START_LOCKED_MASTER(threading)
    vector_PRECISION true_r = NULL;
    PUBLIC_MALLOC( true_r, complex_PRECISION, l->vector_size );
    vector_PRECISION_define( true_r, 0, 0, l->inner_vector_size, l );
    if ( D_phi == NULL ) {
      for ( mu=0; mu<4; mu++ ) {
        ghost_update_PRECISION( x, mu, +1, &(s->op.c), l );
        ghost_update_PRECISION( x, mu, -1, &(s->op.c), l );
      }
      for ( i=0; i<nb; i++ ) {
        block_op( true_r, x, s->block[i].start*l->num_lattice_site_var, s, l, no_threading );
      }
      for ( mu=0; mu<4; mu++ ) {
        ghost_update_wait_PRECISION( x, mu, +1, &(s->op.c), l );
        ghost_update_wait_PRECISION( x, mu, -1, &(s->op.c), l );
      }
      for ( i=0; i<nb; i++ ) {
        boundary_op( true_r, x, i, s, l );
      }
    }
    vector_PRECISION_saxpy( true_r, eta, true_r, -1, 0, l->inner_vector_size, l );
    double rnorm = global_norm_PRECISION( true_r, 0, l->inner_vector_size, l, no_threading )
                 / global_norm_PRECISION( eta, 0, l->inner_vector_size, l, no_threading );
    char number[3]; sprintf( number, "%2d", 31+l->depth ); printf0("\033[1;%2sm|", number );
    printf0(" ---- depth: %d, c: %d, schwarz iter %2d, norm: %11.6le |", l->depth, s->num_colors, k, rnorm );
    printf0("\033[0m\n"); fflush(0);
    PUBLIC_FREE( true_r, complex_PRECISION, l->vector_size );
    END_LOCKED_MASTER(threading)
#endif
  }
  
  END_NO_HYPERTHREADS(threading)
}


void trans_PRECISION( vector_PRECISION out, vector_double in, int *tt, level_struct *l, struct Thread *threading ) {
  
  int i, index;
  vector_PRECISION out_pt = out; vector_double in_pt = in;
  int start = threading->start_site[l->depth];
  int end   = threading->end_site[l->depth];

  // this function seems to do some data reordering, barriers ensure that everything is in sync
  SYNC_CORES(threading)
  START_NO_HYPERTHREADS(threading)
  for ( i=start; i<end; i++ ) {
    index = tt[i];
    out_pt = out + 12*index;
    in_pt  = in + 12*i;
    FOR12( *out_pt = (complex_PRECISION) *in_pt; out_pt++; in_pt++; )
  }
  END_NO_HYPERTHREADS(threading)
  SYNC_CORES(threading)
}


void trans_back_PRECISION( vector_double out, vector_PRECISION in, int *tt, level_struct *l, struct Thread *threading ) {
  
  int i, index;
  vector_double out_pt = out; vector_PRECISION in_pt = in;
  int start = threading->start_site[l->depth];
  int end   = threading->end_site[l->depth];

  // this function seems to do some data reordering, barriers ensure that everything is in sync
  SYNC_CORES(threading)
  START_NO_HYPERTHREADS(threading)
  for ( i=start; i<end; i++ ) {
    index = tt[i];
    in_pt = in + 12*index;
    out_pt = out + 12*i;
    FOR12( *out_pt = (complex_double) *in_pt; out_pt++; in_pt++; )
  }
  END_NO_HYPERTHREADS(threading)
  SYNC_CORES(threading)
}


void schwarz_PRECISION_def( schwarz_PRECISION_struct *s, operator_double_struct *op, level_struct *l ) {

  schwarz_PRECISION_alloc( s, l );
#ifdef CUDA_OPT
  if( l->depth==0 && g.odd_even ){
    schwarz_PRECISION_alloc_CUDA( s, l );
  }
#endif
  schwarz_layout_PRECISION_define( s, l );
  schwarz_PRECISION_setup( s, op, l );
#ifdef CUDA_OPT
  //MPI_Barrier(MPI_COMM_WORLD);
  //printf("before...\n");
  if( l->depth==0 && g.odd_even ){
    schwarz_PRECISION_setup_CUDA( s, op, l );
  }
  //MPI_Barrier(MPI_COMM_WORLD);
  //printf("after...\n");
  //MPI_Barrier(MPI_COMM_WORLD);
  //MPI_Abort(MPI_COMM_WORLD, 911);
#endif
}


// TODO
#ifdef CUDA_OPT
void schwarz_PRECISION_def_CUDA( schwarz_PRECISION_struct *s, operator_double_struct *op, level_struct *l ) { }
#endif


void schwarz_PRECISION_mvm_testfun( schwarz_PRECISION_struct *s, level_struct *l, struct Thread *threading ) {
  
  START_UNTHREADED_FUNCTION(threading)

  int mu, i, nb = s->num_blocks;
  void (*block_op)() = (l->depth==0)?block_d_plus_clover_PRECISION:coarse_block_operator_PRECISION;
  void (*boundary_op)() = (l->depth==0)?block_PRECISION_boundary_op:coarse_block_PRECISION_boundary_op;
  void (*op)() = (l->depth==0)?d_plus_clover_PRECISION:apply_coarse_operator_PRECISION;
  
  vector_PRECISION v1 = NULL, v2 = NULL, v3 = NULL;
  PRECISION norm;
  
  MALLOC( v1, complex_PRECISION, l->schwarz_vector_size );
  MALLOC( v2, complex_PRECISION, l->vector_size );
  MALLOC( v3, complex_PRECISION, l->vector_size );
  
  vector_PRECISION_define_random( v1, 0, l->inner_vector_size, l );
  
  op( v3, v1, &(s->op), l, no_threading );
  
  for ( mu=0; mu<4; mu++ ) {
    ghost_update_PRECISION( v1, mu, +1, &(s->op.c), l );
    ghost_update_PRECISION( v1, mu, -1, &(s->op.c), l );
  }
  
  for ( mu=0; mu<4; mu++ ) {
    ghost_update_wait_PRECISION( v1, mu, +1, &(s->op.c), l );
    ghost_update_wait_PRECISION( v1, mu, -1, &(s->op.c), l );
  }
  
  for ( i=0; i<nb; i++ ) {
    block_op( v2, v1, s->block[i].start*l->num_lattice_site_var, s, l, no_threading );
    boundary_op( v2, v1, i, s, l, no_threading );
  }
  
  vector_PRECISION_minus( v3, v3, v2, 0, l->inner_vector_size, l );  
  norm = global_norm_PRECISION( v3, 0, l->inner_vector_size, l, no_threading )/global_norm_PRECISION( v2, 0, l->inner_vector_size, l, no_threading );
  
  printf0("depth: %d, correctness of local residual vector: %le\n", l->depth, norm );
  
  FREE( v1, complex_PRECISION, l->schwarz_vector_size );
  FREE( v2, complex_PRECISION, l->vector_size );
  FREE( v3, complex_PRECISION, l->vector_size );

  END_UNTHREADED_FUNCTION(threading)
}

