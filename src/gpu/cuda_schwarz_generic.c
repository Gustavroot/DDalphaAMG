#include "main.h"

#ifdef CUDA_OPT
#include "cuda_ghost_PRECISION.h"

void smoother_PRECISION_def_CUDA( level_struct *l ) {
  if ( g.method >= 0 )
    schwarz_PRECISION_def_CUDA( &(l->s_PRECISION), &(g.op_double), l );
}


void smoother_PRECISION_free_CUDA( level_struct *l ) {
  
  if( l->depth==0 && g.odd_even ){
    schwarz_PRECISION_free_CUDA( &(l->s_PRECISION), l );
  }
}


void schwarz_PRECISION_init_CUDA( schwarz_PRECISION_struct *s, level_struct *l ) {

  int i;

  s->streams = NULL;

  (s->cu_s).buf1 = NULL;
  (s->cu_s).buf2 = NULL;
  (s->cu_s).buf3 = NULL;
  (s->cu_s).buf4 = NULL;
  (s->cu_s).buf5 = NULL;
  (s->cu_s).buf6 = NULL;

  for( i=0; i<4; i++ ){
    (s->s_on_gpu_cpubuff).oe_buf[i] = NULL;
  }

  (s->cu_s).local_minres_buffer[0] = NULL;
  (s->cu_s).local_minres_buffer[1] = NULL;
  (s->cu_s).local_minres_buffer[2] = NULL;

  s->s_on_gpu = NULL;

  s->nr_DD_blocks_in_comms = NULL;
  s->nr_DD_blocks_notin_comms = NULL;
  s->nr_DD_blocks = NULL;
  s->DD_blocks_in_comms = NULL;
  s->DD_blocks_notin_comms = NULL;
  s->DD_blocks = NULL;

  s->nr_DD_blocks_in_comms = NULL;
  s->nr_DD_blocks_notin_comms = NULL;
  s->nr_DD_blocks = NULL;
  s->DD_blocks_in_comms = NULL;
  s->DD_blocks_notin_comms = NULL;
  s->DD_blocks = NULL;
  (s->cu_s).DD_blocks_in_comms = NULL;
  (s->cu_s).DD_blocks_notin_comms = NULL;
  (s->cu_s).DD_blocks = NULL;

  (s->cu_s).block = NULL;

  (s->s_on_gpu_cpubuff).op.oe_clover_vectorized = NULL;
  (s->s_on_gpu_cpubuff).op.oe_clover_gpustorg = NULL;

  (s->s_on_gpu_cpubuff).op.clover_gpustorg = NULL;

  for( i=0; i<4; i++ ){
    (s->s_on_gpu_cpubuff).oe_index[i] = NULL;
  }

  for( i=0; i<4; i++ ){
    (s->s_on_gpu_cpubuff).index[i] = NULL;
  }

  (s->s_on_gpu_cpubuff).op.neighbor_table = NULL;

  (s->s_on_gpu_cpubuff).op.D = NULL;

  for(i=0; i<16; i++){
    (s->s_on_gpu_cpubuff).op.Dgpu[i] = NULL;
  }

  // Space to store values from dot products within block solves in SAP
  (s->s_on_gpu_cpubuff).alphas = NULL;

  int mu_dir, inv_mu_dir, mu, dir;

  for( mu=0; mu<4; mu++ ){
    //-1
    dir = -1;
    mu_dir = 2*mu-MIN(dir,0);
    inv_mu_dir = 2*mu+1+MIN(dir,0);
    (s->op.c).boundary_table_gpu[inv_mu_dir] = NULL;
    (s->op.c).buffer_gpu[mu_dir] = NULL;
    //+1
    dir = 1;
    mu_dir = 2*mu-MIN(dir,0);
    inv_mu_dir = 2*mu+1+MIN(dir,0);
    (s->op.c).boundary_table_gpu[inv_mu_dir] = NULL;
    (s->op.c).buffer_gpu[mu_dir] = NULL;
  }

}


void schwarz_PRECISION_alloc_CUDA( schwarz_PRECISION_struct *s, level_struct *l ) {

  int vs, i;

  // The following line has to go here and not in init(), due to previous
  // allocation of s->block
  s->block[0].bt_on_gpu = NULL;

  // -------------------------- s-related

  MALLOC( s->streams, cudaStream_t, g.nr_threads );
  for( i=0; i<g.nr_threads; i++ ){
    cuda_safe_call( cudaStreamCreate( &(s->streams[i]) ) );
  }

  // GPU-version of schwarz_PRECISION_struct
  CUDA_MALLOC( s->s_on_gpu, schwarz_PRECISION_struct_on_gpu, 1 );

  // -------------------------- (s->cu_s)-related

  // Buffers for the computations in SAP
  // TODO: double-check the sizes of all these 6 buffers, in particular
  //	   of buf5
  vs = (l->depth==0)?l->inner_vector_size:l->vector_size;
  CUDA_MALLOC( (s->cu_s).buf1, cu_cmplx_PRECISION, vs );
  CUDA_MALLOC( (s->cu_s).buf2, cu_cmplx_PRECISION, l->schwarz_vector_size );
  CUDA_MALLOC( (s->cu_s).buf3, cu_cmplx_PRECISION, l->schwarz_vector_size );
  CUDA_MALLOC( (s->cu_s).buf4, cu_cmplx_PRECISION, l->schwarz_vector_size );
  CUDA_MALLOC( (s->cu_s).buf5, cu_cmplx_PRECISION, vs );
  CUDA_MALLOC( (s->cu_s).buf6, cu_cmplx_PRECISION, l->schwarz_vector_size );

#ifndef EXTERNAL_DD
  if( l->depth==0 )
#endif
  {
    // these buffers are introduced to make local_minres_PRECISION thread-safe
    CUDA_MALLOC( (s->cu_s).local_minres_buffer[0], cu_cmplx_PRECISION, l->schwarz_vector_size );
    CUDA_MALLOC( (s->cu_s).local_minres_buffer[1], cu_cmplx_PRECISION, l->schwarz_vector_size );
    CUDA_MALLOC( (s->cu_s).local_minres_buffer[2], cu_cmplx_PRECISION, l->schwarz_vector_size );
  }

  // -------------------------- (s->s_on_gpu_cpubuff)-related

  // TODO: is this pre-processor line necessary ?
#ifndef EXTERNAL_DD
  if ( l->depth == 0 ) {
    // TODO: is l->inner_vector_size the ACTUAL size in this allocation ?
    for( i=0; i<4; i++ ){
      CUDA_MALLOC( (s->s_on_gpu_cpubuff).oe_buf[i], cu_cmplx_PRECISION, l->inner_vector_size );
    }
  }
#endif

  // some allocations necessary to set-up the DD block indices in a
  // friendly way for GPU-computations
  MALLOC( s->nr_DD_blocks_in_comms, int, s->num_colors );
  MALLOC( s->nr_DD_blocks_notin_comms, int, s->num_colors );
  MALLOC( s->nr_DD_blocks, int, s->num_colors );
  MALLOC( s->DD_blocks_in_comms, int*, s->num_colors );
  MALLOC( s->DD_blocks_notin_comms, int*, s->num_colors );
  MALLOC( s->DD_blocks, int*, s->num_colors );

  CUDA_MALLOC( s->block[0].bt_on_gpu, int, s->block_boundary_length[8]*s->num_blocks );
  CUDA_MALLOC( (s->cu_s).block, block_struct, s->num_blocks );

  if( g.csw != 0.0 ){
    CUDA_MALLOC( (s->s_on_gpu_cpubuff).op.oe_clover_gpustorg, cu_cmplx_PRECISION, 42*s->num_block_sites*s->num_blocks );
    CUDA_MALLOC( (s->s_on_gpu_cpubuff).op.oe_clover_vectorized, cu_config_PRECISION, 72*s->num_block_sites*s->num_blocks );
  }
  else{
    CUDA_MALLOC( (s->s_on_gpu_cpubuff).op.oe_clover_gpustorg, cu_cmplx_PRECISION, 12*s->num_block_sites*s->num_blocks );
    CUDA_MALLOC( (s->s_on_gpu_cpubuff).op.oe_clover_vectorized, cu_config_PRECISION, 12*s->num_block_sites*s->num_blocks );
  }

  if( g.csw != 0.0 ){
    CUDA_MALLOC( (s->s_on_gpu_cpubuff).op.clover_gpustorg, cu_cmplx_PRECISION, 42*l->num_inner_lattice_sites );
  }
  else{
    CUDA_MALLOC( (s->s_on_gpu_cpubuff).op.clover_gpustorg, cu_cmplx_PRECISION, 12*l->num_inner_lattice_sites );
  }

  // Space to store values from dot products within block solves in SAP
  CUDA_MALLOC( (s->s_on_gpu_cpubuff).alphas, cu_cmplx_PRECISION, s->num_blocks );

  // allocations of: (s->op.c).boundary_table_gpu[inv_mu_dir], (s->op.c).buffer_gpu[mu_dir]

  int mu_dir, inv_mu_dir, mu, boundary_size, dir;

  for( mu=0; mu<4; mu++ ){
    //-1
    dir = -1;
    mu_dir = 2*mu-MIN(dir,0);
    inv_mu_dir = 2*mu+1+MIN(dir,0);
    boundary_size = (s->op.c).num_boundary_sites[mu_dir];
    CUDA_MALLOC( (s->op.c).boundary_table_gpu[inv_mu_dir], int, boundary_size );
    boundary_size *= l->num_lattice_site_var;
    CUDA_MALLOC( (s->op.c).buffer_gpu[mu_dir], cu_cmplx_PRECISION, boundary_size );
    //+1
    dir = 1;
    mu_dir = 2*mu-MIN(dir,0);
    inv_mu_dir = 2*mu+1+MIN(dir,0);
    boundary_size = (s->op.c).num_boundary_sites[mu_dir];
    CUDA_MALLOC( (s->op.c).boundary_table_gpu[inv_mu_dir], int, boundary_size );
    boundary_size *= l->num_lattice_site_var;
    CUDA_MALLOC( (s->op.c).buffer_gpu[mu_dir], cu_cmplx_PRECISION, boundary_size );
  }

}


void schwarz_PRECISION_free_CUDA( schwarz_PRECISION_struct *s, level_struct *l ) {

  int i, color;

  FREE( s->streams, cudaStream_t, g.nr_threads );

  CUDA_FREE( s->s_on_gpu, schwarz_PRECISION_struct_on_gpu, 1 );

  int vs = (l->depth==0)?l->inner_vector_size:l->vector_size;
  CUDA_FREE( (s->cu_s).buf1, cu_cmplx_PRECISION, vs );
  CUDA_FREE( (s->cu_s).buf2, cu_cmplx_PRECISION, l->schwarz_vector_size );
  CUDA_FREE( (s->cu_s).buf3, cu_cmplx_PRECISION, l->schwarz_vector_size );
  CUDA_FREE( (s->cu_s).buf4, cu_cmplx_PRECISION, l->schwarz_vector_size );
  CUDA_FREE( (s->cu_s).buf5, cu_cmplx_PRECISION, vs );
  CUDA_FREE( (s->cu_s).buf6, cu_cmplx_PRECISION, l->schwarz_vector_size );

#ifndef EXTERNAL_DD
  if ( l->depth == 0 ) {
    // TODO: is l->inner_vector_size the ACTUAL size in this allocation ?
    for( i=0; i<4; i++ ){
      CUDA_FREE( (s->s_on_gpu_cpubuff).oe_buf[i], cu_cmplx_PRECISION, l->inner_vector_size );
    }
  }
#endif

#ifndef EXTERNAL_DD
  if( l->depth==0 )
#endif
  {
    // these buffers are introduced to make local_minres_PRECISION thread-safe
    CUDA_FREE( (s->cu_s).local_minres_buffer[0], cu_cmplx_PRECISION, l->schwarz_vector_size );
    CUDA_FREE( (s->cu_s).local_minres_buffer[1], cu_cmplx_PRECISION, l->schwarz_vector_size );
    CUDA_FREE( (s->cu_s).local_minres_buffer[2], cu_cmplx_PRECISION, l->schwarz_vector_size );
  }

  for(color=0; color<s->num_colors; color++){
    FREE( s->DD_blocks_in_comms[color], int, s->nr_DD_blocks_in_comms[color] );
    FREE( s->DD_blocks_notin_comms[color], int, s->nr_DD_blocks_notin_comms[color] );
    FREE( s->DD_blocks[color], int, s->nr_DD_blocks[color] );
  }

  for(color=0; color<s->num_colors; color++){
    CUDA_FREE( (s->cu_s).DD_blocks_in_comms[color], int, s->nr_DD_blocks_in_comms[color] );
    // FIXME: the following line should be switched to a CUDA_FREE(...)
    cuda_safe_call( cudaFree( (s->cu_s).DD_blocks_notin_comms[color] ) );
    CUDA_FREE( (s->cu_s).DD_blocks[color], int, s->nr_DD_blocks[color] );
  }
  
  if (s->num_colors > 0) {
    FREE( s->nr_DD_blocks_in_comms, int, s->num_colors );
    FREE( s->nr_DD_blocks_notin_comms, int, s->num_colors );
    FREE( s->nr_DD_blocks, int, s->num_colors );
    FREE( s->DD_blocks_in_comms, int*, s->num_colors );
    FREE( s->DD_blocks_notin_comms, int*, s->num_colors );
    FREE( s->DD_blocks, int*, s->num_colors );
    FREE( (s->cu_s).DD_blocks_in_comms, int*, s->num_colors );
    FREE( (s->cu_s).DD_blocks_notin_comms, int*, s->num_colors );
    FREE( (s->cu_s).DD_blocks, int*, s->num_colors );
  }

  // FIXME: enable and fix the following line --> ??
  //cuda_safe_call( cudaFree( s->block[0].bt_on_gpu ) );

  CUDA_FREE( s->block[0].bt_on_gpu, int, s->block_boundary_length[8]*s->num_blocks );
  CUDA_FREE( (s->cu_s).block, block_struct, s->num_blocks );

  if( g.csw != 0.0 ){
    CUDA_FREE( (s->s_on_gpu_cpubuff).op.oe_clover_gpustorg, cu_cmplx_PRECISION, 42*s->num_block_sites*s->num_blocks );
    CUDA_FREE( (s->s_on_gpu_cpubuff).op.oe_clover_vectorized, cu_config_PRECISION, 72*s->num_block_sites*s->num_blocks );
  }
  else{
    CUDA_FREE( (s->s_on_gpu_cpubuff).op.oe_clover_gpustorg, cu_cmplx_PRECISION, 12*s->num_block_sites*s->num_blocks );
    CUDA_FREE( (s->s_on_gpu_cpubuff).op.oe_clover_vectorized, cu_config_PRECISION, 12*s->num_block_sites*s->num_blocks );
  }

  if( g.csw != 0.0 ){
    CUDA_FREE( (s->s_on_gpu_cpubuff).op.clover_gpustorg, cu_cmplx_PRECISION, 42*l->num_inner_lattice_sites );
  }
  else{
    CUDA_FREE( (s->s_on_gpu_cpubuff).op.clover_gpustorg, cu_cmplx_PRECISION, 12*l->num_inner_lattice_sites );
  }

  // Space to store values from dot products within block solves in SAP
  CUDA_FREE( (s->s_on_gpu_cpubuff).alphas, cu_cmplx_PRECISION, s->num_blocks );

  for( i=0; i<4; i++ ){
    int nr_oe_elems_hopp = s->dir_length_even[i] + s->dir_length_odd[i];
    CUDA_FREE( (s->s_on_gpu_cpubuff).oe_index[i], int, nr_oe_elems_hopp );
  }

  for( i=0; i<4; i++ ){
    int nr_elems_block_d_plus = s->dir_length[i];
    CUDA_FREE( (s->s_on_gpu_cpubuff).index[i], int, nr_elems_block_d_plus );
  }

  CUDA_FREE( (s->s_on_gpu_cpubuff).op.neighbor_table, int, (l->depth==0?4:5)*l->num_inner_lattice_sites );

  // freeing s->op.D (and its optimal form) from the GPUs :

  int type=_SCHWARZ;

  int coupling_site_size, nls, nr_elems_D;

  if ( l->depth == 0 ) {
    coupling_site_size = 4*9;
  } else {
    coupling_site_size = 4*l->num_lattice_site_var*l->num_lattice_site_var;
  }

  nls = (type==_ORDINARY)?l->num_inner_lattice_sites:2*l->num_lattice_sites-l->num_inner_lattice_sites;
  nr_elems_D = coupling_site_size*nls;

  CUDA_FREE( (s->s_on_gpu_cpubuff).op.D, cu_config_PRECISION, nr_elems_D );

  // 0, 1, 2, 3 ----> plus-even, plus-odd, minus-even, minus-odd

  for( i=0; i<16; i++ ){
    CUDA_FREE( (s->s_on_gpu_cpubuff).op.Dgpu[i], cu_cmplx_PRECISION, ((s->s_on_gpu_cpubuff).op.nr_elems_Dgpu)[i]*9 );
  }

  int mu_dir, inv_mu_dir, mu, boundary_size, dir;

  for( mu=0; mu<4; mu++ ){
    //-1
    dir = -1;
    mu_dir = 2*mu-MIN(dir,0);
    inv_mu_dir = 2*mu+1+MIN(dir,0);
    boundary_size = (s->op.c).num_boundary_sites[mu_dir];
    CUDA_FREE( (s->op.c).boundary_table_gpu[inv_mu_dir], int, boundary_size );
    boundary_size *= l->num_lattice_site_var;
    CUDA_FREE( (s->op.c).buffer_gpu[mu_dir], cu_cmplx_PRECISION, boundary_size );
    //+1
    dir = 1;
    mu_dir = 2*mu-MIN(dir,0);
    inv_mu_dir = 2*mu+1+MIN(dir,0);
    boundary_size = (s->op.c).num_boundary_sites[mu_dir];
    CUDA_FREE( (s->op.c).boundary_table_gpu[inv_mu_dir], int, boundary_size );
    boundary_size *= l->num_lattice_site_var;
    CUDA_FREE( (s->op.c).buffer_gpu[mu_dir], cu_cmplx_PRECISION, boundary_size );
  }
}


void schwarz_PRECISION_setup_CUDA( schwarz_PRECISION_struct *s, operator_double_struct *op_in, level_struct *l ) {

  int color, comms_ctr, noncomms_ctr, color_ctr, i, j, k;

  // ----------------------------------------------------------------------------------------------------------
  // code necessary to set-up the DD block indices in a
  // friendly way for GPU-computations
  for(color=0; color<s->num_colors; color++){
    s->nr_DD_blocks_notin_comms[color] = 0;
    s->nr_DD_blocks_in_comms[color] = 0;
    s->nr_DD_blocks[color] = 0;
    for(i=0; i<s->num_blocks; i++){
      if ( color == s->block[i].color && s->block[i].no_comm ) {
        s->nr_DD_blocks_notin_comms[color]++;
      }
      else if( color == s->block[i].color && !s->block[i].no_comm ){
        s->nr_DD_blocks_in_comms[color]++;
      }
      if( color == s->block[i].color ){
        s->nr_DD_blocks[color]++;
      }
    }
  }
  for(color=0; color<s->num_colors; color++){
    s->DD_blocks_in_comms[color] = NULL;
    MALLOC( s->DD_blocks_in_comms[color], int, s->nr_DD_blocks_in_comms[color] );

    s->DD_blocks_notin_comms[color] = (int*) malloc( s->nr_DD_blocks_notin_comms[color]*sizeof(int) );
    // FIXME: enable (and fix errors associated to) following two lines of code !
    //s->DD_blocks_notin_comms[color] = NULL;
    //MALLOC( s->DD_blocks_notin_comms[color], int, s->nr_DD_blocks_notin_comms[color] );

    s->DD_blocks[color] = NULL;
    MALLOC( s->DD_blocks[color], int, s->nr_DD_blocks[color] );
  }
  for(color=0; color<s->num_colors; color++){
    comms_ctr = 0;
    noncomms_ctr = 0;
    color_ctr = 0;
    for(i=0; i<s->num_blocks; i++){
      if ( color == s->block[i].color && s->block[i].no_comm ) {
        s->DD_blocks_notin_comms[color][noncomms_ctr] = i;
        noncomms_ctr++;
      }
      else if( color == s->block[i].color && !s->block[i].no_comm ){
        s->DD_blocks_in_comms[color][comms_ctr] = i;
        comms_ctr++;
      }
      if( color == s->block[i].color ){
        s->DD_blocks[color][color_ctr] = i;
        color_ctr++;
      }
    }
  }
  MALLOC( (s->cu_s).DD_blocks_in_comms, int*, s->num_colors );
  MALLOC( (s->cu_s).DD_blocks_notin_comms, int*, s->num_colors );
  MALLOC( (s->cu_s).DD_blocks, int*, s->num_colors );
  for (color=0; color<s->num_colors; color++) {
    // assuming that NULL is represented by 0s (think in the byte-sense)
    memset( (s->cu_s).DD_blocks_in_comms, 0, s->num_colors*sizeof(int*) );
    memset( (s->cu_s).DD_blocks_notin_comms, 0, s->num_colors*sizeof(int*) );
    memset( (s->cu_s).DD_blocks, 0, s->num_colors*sizeof(int*) );
  }
  for(color=0; color<s->num_colors; color++){
    CUDA_MALLOC( (s->cu_s).DD_blocks_in_comms[color], int, s->nr_DD_blocks_in_comms[color] );
    // FIXME: the following line should be switched to a CUDA_MALLOC(...)
    cuda_safe_call( cudaMalloc( (void**)(&( (s->cu_s).DD_blocks_notin_comms[color] )), s->nr_DD_blocks_notin_comms[color]*sizeof(int) ) );
    CUDA_MALLOC( (s->cu_s).DD_blocks[color], int, s->nr_DD_blocks[color] );
  }
  for(color=0; color<s->num_colors; color++){
    cuda_safe_call( cudaMemcpy( (s->cu_s).DD_blocks_in_comms[color], s->DD_blocks_in_comms[color],
                                s->nr_DD_blocks_in_comms[color]*sizeof(int), cudaMemcpyHostToDevice) );
    cuda_safe_call( cudaMemcpy( (s->cu_s).DD_blocks_notin_comms[color], s->DD_blocks_notin_comms[color],
                                s->nr_DD_blocks_notin_comms[color]*sizeof(int), cudaMemcpyHostToDevice) );
    cuda_safe_call( cudaMemcpy( (s->cu_s).DD_blocks[color], s->DD_blocks[color],
                                s->nr_DD_blocks[color]*sizeof(int), cudaMemcpyHostToDevice) );
  }

  // ----------------------------------------------------------------------------------------------------------
  // some (relatively) minor data movements:

  // 1. s->block[i].bt_on_gpu

  for(i=1; i<s->num_blocks; i++){
    s->block[i].bt_on_gpu = s->block[0].bt_on_gpu + i * s->block_boundary_length[8];
  }
  for(i=0; i<s->num_blocks; i++){
    cuda_safe_call( cudaMemcpy( s->block[i].bt_on_gpu, s->block[i].bt,
                                s->block_boundary_length[8] * sizeof(int), cudaMemcpyHostToDevice ) );
  }

  // 2. s->block

  cuda_safe_call( cudaMemcpy( (s->cu_s).block, s->block,
                              s->num_blocks*sizeof(block_struct), cudaMemcpyHostToDevice) );

  // 3. s->block_boundary_length[i]

  for( i=0; i<9; i++ ){
    (s->s_on_gpu_cpubuff).block_boundary_length[i] = s->block_boundary_length[i];
  }

  // ----------------------------------------------------------------------------------------------------------
  // CRITICAL DATA MOVEMENT: copying op.oe_clover_vectorized to the GPU

  // instances
  int b, h, nr_DD_sites;

  // definitions
  nr_DD_sites = s->num_block_sites;
  schwarz_PRECISION_struct_on_gpu *out = &(s->s_on_gpu_cpubuff);
  schwarz_PRECISION_struct *in = s;
  cu_cmplx_PRECISION *buf_D_oe_cpu=NULL, *buf_D_oe_cpu_bare=NULL;

  // memory allocs

  if(g.csw != 0.0){
    MALLOC( buf_D_oe_cpu, cu_cmplx_PRECISION, 72 * nr_DD_sites*in->num_blocks );
  }
  else{
    // no buffers needed to re-arrange data, as clover is diagonal when csw=0
  }

  // transactions

  buf_D_oe_cpu_bare = buf_D_oe_cpu;

  if(g.csw != 0.0){

    PRECISION *op_oe_vect_bare = in->op.oe_clover_vectorized;
    PRECISION *op_oe_vect = op_oe_vect_bare;

    PRECISION M_tmp[144];
    PRECISION *M_tmp1, *M_tmp2;
    M_tmp1 = M_tmp;
    M_tmp2 = M_tmp + 72;

    for(b=0; b < in->num_blocks; b++){
      for(h=0; h<nr_DD_sites; h++){
        // the following snippet of code was taken from function sse_site_clover_invert_float(...) in sse_dirac.c
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
  }
  else{
    // no buffers needed to re-arrange data, as clover is diagonal when csw=0
  }

  // making use of Doo^-1 (i.e. buf_D_oe_cpu_bare) being Hermitian to store in reduced form
  // IMPORTANT: this matrix is stored in column-form

  cu_cmplx_PRECISION *buf_D_oe_cpu_gpustorg=NULL, *buf_D_oe_cpu_gpustorg_bare=NULL;

  if(g.csw != 0.0){
    MALLOC( buf_D_oe_cpu_gpustorg, cu_cmplx_PRECISION, 42 * nr_DD_sites*in->num_blocks );
  }
  else{
    // no buffers needed to re-arrange data, as clover is diagonal when csw=0
  }

  buf_D_oe_cpu = buf_D_oe_cpu_bare;
  buf_D_oe_cpu_gpustorg_bare = buf_D_oe_cpu_gpustorg;

  if( g.csw != 0.0 ){
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

        buf_D_oe_cpu += 72;
        buf_D_oe_cpu_gpustorg += 42;
      }
    }
  }
  else{
    // no buffers needed to re-arrange data, as clover is diagonal when csw=0
  }

  if(g.csw != 0.0){
    cuda_safe_call( cudaMemcpy( out->op.oe_clover_vectorized, buf_D_oe_cpu_bare,
                                72*sizeof(cu_cmplx_PRECISION)*nr_DD_sites*in->num_blocks, cudaMemcpyHostToDevice) );
  }
  else{
    cuda_safe_call( cudaMemcpy( out->op.oe_clover_vectorized, in->op.oe_clover_vectorized,
                                12*sizeof(cu_cmplx_PRECISION)*nr_DD_sites*in->num_blocks, cudaMemcpyHostToDevice) );
  }

  if(g.csw != 0.0){
    cuda_safe_call( cudaMemcpy( out->op.oe_clover_gpustorg, buf_D_oe_cpu_gpustorg_bare,
                                42*sizeof(cu_cmplx_PRECISION)*nr_DD_sites*in->num_blocks, cudaMemcpyHostToDevice) );
  }
  else{
    cuda_safe_call( cudaMemcpy(out->op.oe_clover_gpustorg, in->op.oe_clover,
                    12*sizeof(cu_cmplx_PRECISION)*nr_DD_sites*in->num_blocks, cudaMemcpyHostToDevice) );
    buf_D_oe_cpu_gpustorg_bare = NULL;
  }

  // memory de-allocs --> CPU bufs

  if(g.csw != 0.0){
    FREE( buf_D_oe_cpu_bare, cu_cmplx_PRECISION, 72 * nr_DD_sites*in->num_blocks );
  }
  else{
    // no buffers needed to re-arrange data, as clover is diagonal when csw=0
  }
  if(g.csw != 0.0){
    FREE( buf_D_oe_cpu_gpustorg_bare, cu_cmplx_PRECISION, 42 * nr_DD_sites*in->num_blocks );
  }
  else{
    // no buffers needed to re-arrange data, as clover is diagonal when csw=0
  }

  //-------------------------------------------------------------------------------------------------------------------------
  // copying s->op.clover

  int n=l->num_inner_lattice_sites;

  if(g.csw != 0.0){
    cuda_safe_call( cudaMemcpy( out->op.clover_gpustorg, s->op.clover,
                                42 * sizeof(cu_cmplx_PRECISION) * n, cudaMemcpyHostToDevice ) );
  }
  else{
    cuda_safe_call( cudaMemcpy( out->op.clover_gpustorg, s->op.clover,
                                12 * sizeof(cu_cmplx_PRECISION) * n, cudaMemcpyHostToDevice ) );
  }

  //-------------------------------------------------------------------------------------------------------------------------
  // some (relatively) minor data movements:

  int dir, nr_oe_elems_hopp, nr_elems_block_d_plus, type=_SCHWARZ, nls, coupling_site_size, nr_elems_D;

  // 1. s->dir_length_even[dir]

  for( dir=0; dir<4; dir++ ){
    (s->s_on_gpu_cpubuff).dir_length_even[dir] = s->dir_length_even[dir];
    (s->s_on_gpu_cpubuff).dir_length_odd[dir] = s->dir_length_odd[dir];
  }

  // 2. s->oe_index[dir]

  for( dir=0; dir<4; dir++ ){
    nr_oe_elems_hopp = s->dir_length_even[dir] + s->dir_length_odd[dir];
    CUDA_MALLOC( (s->s_on_gpu_cpubuff).oe_index[dir], int, nr_oe_elems_hopp );
    cuda_safe_call( cudaMemcpy( (s->s_on_gpu_cpubuff).oe_index[dir], s->oe_index[dir],
                                nr_oe_elems_hopp*sizeof(int), cudaMemcpyHostToDevice ) );
  }

  // 3. s->dir_length[dir]

  for( dir=0; dir<4; dir++ ){
    (s->s_on_gpu_cpubuff).dir_length[dir] = s->dir_length[dir];
    (s->s_on_gpu_cpubuff).dir_length[dir] = s->dir_length[dir];
  }

  // 4. s->index[dir]

  for( dir=0; dir<4; dir++ ){
    nr_elems_block_d_plus = s->dir_length[dir];
    CUDA_MALLOC( (s->s_on_gpu_cpubuff).index[dir], int, nr_elems_block_d_plus );
    cuda_safe_call( cudaMemcpy( (s->s_on_gpu_cpubuff).index[dir], s->index[dir],
                                nr_elems_block_d_plus*sizeof(int), cudaMemcpyHostToDevice ) );
  }

  // 5. s->op.neighbor_table

  CUDA_MALLOC( (s->s_on_gpu_cpubuff).op.neighbor_table, int, (l->depth==0?4:5)*l->num_inner_lattice_sites );
  cuda_safe_call( cudaMemcpy( (s->s_on_gpu_cpubuff).op.neighbor_table, s->op.neighbor_table,
                              (l->depth==0?4:5)*l->num_inner_lattice_sites*sizeof(int), cudaMemcpyHostToDevice ) );

  // 6. s->op.D

  if ( l->depth == 0 ) {
    coupling_site_size = 4*9;
  } else {
    coupling_site_size = 4*l->num_lattice_site_var*l->num_lattice_site_var;
  }

  nls = (type==_ORDINARY)?l->num_inner_lattice_sites:2*l->num_lattice_sites-l->num_inner_lattice_sites;
  nr_elems_D = coupling_site_size*nls;

  CUDA_MALLOC( (s->s_on_gpu_cpubuff).op.D, cu_config_PRECISION, nr_elems_D );
  cuda_safe_call( cudaMemcpy( (s->s_on_gpu_cpubuff).op.D, (cu_config_PRECISION*)(s->op.D),
                              nr_elems_D*sizeof(cu_config_PRECISION), cudaMemcpyHostToDevice) );

  //-------------------------------------------------------------------------------------------------------------------------
  // transforming s->op.D to optimal form for accesses from the GPU

  // 0, 1, 2, 3 ----> plus-even, plus-odd, minus-even, minus-odd
  cu_cmplx_PRECISION *Dgpu[16]={NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL}, *Dgpu_buff;
  int nr_elems_Dgpu[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  int a1, n1, a2, n2, DD_start, *ind, **index = s->oe_index;

  config_PRECISION Dbuff, D_pt;

  // do PLUS first

  // ... for plus, do EVEN
  //amount = _EVEN_SITES;

  for( dir=0; dir<4; dir++ ){
    a1 = 0; n1 = s->dir_length_even[dir];
    nr_elems_Dgpu[0*8 + 0*4 + dir] =  (n1-a1);
    nr_elems_Dgpu[0*8 + 0*4 + dir] *= s->num_blocks;

    // 0*8 + 0*4 + dir ---> PLUS(0), EVEN(0), dir
    MALLOC( Dgpu[0*8 + 0*4 + dir], cu_cmplx_PRECISION, nr_elems_Dgpu[0*8 + 0*4 + dir] * 9 );
    Dgpu_buff = Dgpu[0*8 + 0*4 + dir];

    for( j=0; j<s->num_blocks; j++ ){
      DD_start = s->block[j].start*l->num_lattice_site_var;
      Dbuff = s->op.D + (DD_start/12)*36;
      ind = index[dir];
      for ( i=a1; i<n1; i++ ) {
        k = ind[i];
        //j = neighbor[4*k+T];
        D_pt = Dbuff + 36*k + 9*dir;
        // copy 9 complex entries: Dgpu_buff <----- D_pt
        for( h=0; h<9; h++ ){
          Dgpu_buff[h] = make_cu_cmplx_PRECISION( creal_PRECISION(D_pt[h]), cimag_PRECISION(D_pt[h]) );
        }
        Dgpu_buff += 9;
      }
    }
    // copy recently created information to the GPU
    CUDA_MALLOC( (s->s_on_gpu_cpubuff).op.Dgpu[0*8 + 0*4 + dir], cu_cmplx_PRECISION, nr_elems_Dgpu[0*8 + 0*4 + dir]*9 );
    cuda_safe_call( cudaMemcpy( (s->s_on_gpu_cpubuff).op.Dgpu[0*8 + 0*4 + dir], Dgpu[0*8 + 0*4 + dir],
                                nr_elems_Dgpu[0*8 + 0*4 + dir] * 9 * sizeof(cu_cmplx_PRECISION),
                                cudaMemcpyHostToDevice) );
  }

  // ... for plus, do ODD
  //amount = _ODD_SITES;

  for( dir=0; dir<4; dir++ ){
    a1 = s->dir_length_even[dir]; n1 = a1 + s->dir_length_odd[dir];
    nr_elems_Dgpu[0*8 + 1*4 + dir] =  (n1-a1);
    nr_elems_Dgpu[0*8 + 1*4 + dir] *= s->num_blocks;

    // 0*8 + 1*4 + dir ---> PLUS(0), ODD(1), dir
    MALLOC( Dgpu[0*8 + 1*4 + dir], cu_cmplx_PRECISION, nr_elems_Dgpu[0*8 + 1*4 + dir] * 9 );
    Dgpu_buff = Dgpu[0*8 + 1*4 + dir];

    for( j=0; j<s->num_blocks; j++ ){
      DD_start = s->block[j].start*l->num_lattice_site_var;
      Dbuff = s->op.D + (DD_start/12)*36;
      ind = index[dir];
      for ( i=a1; i<n1; i++ ) {
        k = ind[i];
        //j = neighbor[4*k+T];
        D_pt = Dbuff + 36*k + 9*dir;
        // copy 9 complex entries: Dgpu_buff <----- D_pt
        for( h=0; h<9; h++ ){
          Dgpu_buff[h] = make_cu_cmplx_PRECISION( creal_PRECISION(D_pt[h]), cimag_PRECISION(D_pt[h]) );
        }
        Dgpu_buff += 9;
      }
    }
    // copy recently created information to the GPU
    CUDA_MALLOC( (s->s_on_gpu_cpubuff).op.Dgpu[0*8 + 1*4 + dir], cu_cmplx_PRECISION, nr_elems_Dgpu[0*8 + 1*4 + dir]*9 );
    cuda_safe_call( cudaMemcpy( (s->s_on_gpu_cpubuff).op.Dgpu[0*8 + 1*4 + dir], Dgpu[0*8 + 1*4 + dir],
                                nr_elems_Dgpu[0*8 + 1*4 + dir] * 9 * sizeof(cu_cmplx_PRECISION),
                                cudaMemcpyHostToDevice) );
  }

  // then do MINUS

  // ... for minus, do EVEN
  //amount = _EVEN_SITES;

  for( dir=0; dir<4; dir++ ){
    a1 = 0; n1 = s->dir_length_even[dir];
    a2 = n1; n2 = a2 + s->dir_length_odd[dir];
    nr_elems_Dgpu[1*8 + 0*4 + dir] =  (n2-a2);
    nr_elems_Dgpu[1*8 + 0*4 + dir] *= s->num_blocks;

    // 1*8 + 0*4 + dir ---> MINUS(1), EVEN(0), dir
    MALLOC( Dgpu[1*8 + 0*4 + dir], cu_cmplx_PRECISION, nr_elems_Dgpu[1*8 + 0*4 + dir] * 9 );
    Dgpu_buff = Dgpu[1*8 + 0*4 + dir];

    for( j=0; j<s->num_blocks; j++ ){
      DD_start = s->block[j].start*l->num_lattice_site_var;
      Dbuff = s->op.D + (DD_start/12)*36;
      ind = index[dir];
      for ( i=a2; i<n2; i++ ) {
        k = ind[i];
        //j = neighbor[4*k+T];
        D_pt = Dbuff + 36*k + 9*dir;
        // copy 9 complex entries: Dgpu_buff <----- D_pt
        for( h=0; h<9; h++ ){
          Dgpu_buff[h] = make_cu_cmplx_PRECISION( creal_PRECISION(D_pt[h]), cimag_PRECISION(D_pt[h]) );
        }
        Dgpu_buff += 9;
      }
    }
    // copy recently created information to the GPU
    CUDA_MALLOC( (s->s_on_gpu_cpubuff).op.Dgpu[1*8 + 0*4 + dir], cu_cmplx_PRECISION, nr_elems_Dgpu[1*8 + 0*4 + dir]*9 );
    cuda_safe_call( cudaMemcpy( (s->s_on_gpu_cpubuff).op.Dgpu[1*8 + 0*4 + dir], Dgpu[1*8 + 0*4 + dir],
                                nr_elems_Dgpu[1*8 + 0*4 + dir] * 9 * sizeof(cu_cmplx_PRECISION),
                                cudaMemcpyHostToDevice) );
  }

  // ... for minus, do ODD
  //amount = _ODD_SITES;

  for( dir=0; dir<4; dir++ ){
    a1 = s->dir_length_even[dir]; n1 = a1 + s->dir_length_odd[dir];
    a2 = 0; n2 = a1;
    nr_elems_Dgpu[1*8 + 1*4 + dir] =  (n2-a2);
    nr_elems_Dgpu[1*8 + 1*4 + dir] *= s->num_blocks;

    // 1*8 + 1*4 + dir ---> MINUS(1), ODD(1), dir
    MALLOC( Dgpu[1*8 + 1*4 + dir], cu_cmplx_PRECISION, nr_elems_Dgpu[1*8 + 1*4 + dir] * 9 );
    Dgpu_buff = Dgpu[1*8 + 1*4 + dir];

    for( j=0; j<s->num_blocks; j++ ){
      DD_start = s->block[j].start*l->num_lattice_site_var;
      Dbuff = s->op.D + (DD_start/12)*36;
      ind = index[dir];
      for ( i=a2; i<n2; i++ ) {
        k = ind[i];
        //j = neighbor[4*k+T];
        D_pt = Dbuff + 36*k + 9*dir;
        // copy 9 complex entries: Dgpu_buff <----- D_pt
        for( h=0; h<9; h++ ){
          Dgpu_buff[h] = make_cu_cmplx_PRECISION( creal_PRECISION(D_pt[h]), cimag_PRECISION(D_pt[h]) );
        }
        Dgpu_buff += 9;
      }
    }
    // copy recently created information to the GPU
    CUDA_MALLOC( (s->s_on_gpu_cpubuff).op.Dgpu[1*8 + 1*4 + dir], cu_cmplx_PRECISION, nr_elems_Dgpu[1*8 + 1*4 + dir]*9 );
    cuda_safe_call( cudaMemcpy( (s->s_on_gpu_cpubuff).op.Dgpu[1*8 + 1*4 + dir], Dgpu[1*8 + 1*4 + dir],
                                nr_elems_Dgpu[1*8 + 1*4 + dir] * 9 * sizeof(cu_cmplx_PRECISION),
                                cudaMemcpyHostToDevice) );
  }

  // copying nr_elems_Dgpu to the CPU - GPU-buffer
  memcpy( (s->s_on_gpu_cpubuff).op.nr_elems_Dgpu, nr_elems_Dgpu, 16*sizeof(int) );
  //cuda_safe_call( cudaMemcpy( (s->s_on_gpu_cpubuff).op.nr_elems_Dgpu, nr_elems_Dgpu, 16*sizeof(int), cudaMemcpyHostToDevice ) );

  // de-allocating CPU buffers used for optimal op.D
  for(i=0; i<16; i++){
    FREE( Dgpu[i], cu_cmplx_PRECISION, nr_elems_Dgpu[i]*9 );
  }

  //-------------------------------------------------------------------------------------------------------------------------
  // copying gamma matrices

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

  // copying gamma-matrices info to constant GPU memory
  cudaMemcpyToSymbol(gamma_info_vals_PRECISION, gamma_info_vals_buff, sizeof(cu_cmplx_PRECISION)*16, 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(gamma_info_coo_PRECISION, gamma_info_coo_buff, sizeof(int)*16, 0, cudaMemcpyHostToDevice);

  //-------------------------------------------------------------------------------------------------------------------------
  // copying some minor variables

  (s->s_on_gpu_cpubuff).num_block_even_sites = s->num_block_even_sites;
  (s->s_on_gpu_cpubuff).num_block_odd_sites = s->num_block_odd_sites;
  (s->s_on_gpu_cpubuff).block_vector_size = s->block_vector_size;

  //-------------------------------------------------------------------------------------------------------------------------
  // creation of some extra information in s to facilitate GPU calls

  // The following variable is an indirect indication of the necessary work
  // over boundaries
  int *bbl = s->block_boundary_length;
  s->tot_num_boundary_work = 0;
  for( dir=0; dir<4; dir++ ){
    // +
    s->num_boundary_sites[dir*2] = (bbl[2*dir+1] - bbl[2*dir])/2;
    s->tot_num_boundary_work += s->num_boundary_sites[dir*2];
    // -
    s->num_boundary_sites[dir*2+1] = (bbl[2*dir+2] - bbl[2*dir+1])/2;
    s->tot_num_boundary_work += s->num_boundary_sites[dir*2+1];
  }

  //-------------------------------------------------------------------------------------------------------------------------
  // After all the allocations and definitions associated to s->s_on_gpu_cpubuff, it's time to move to the GPU

  cuda_safe_call( cudaMemcpy( s->s_on_gpu, &(s->s_on_gpu_cpubuff), 1*sizeof(schwarz_PRECISION_struct_on_gpu),
                              cudaMemcpyHostToDevice) );

  //-------------------------------------------------------------------------------------------------------------------------
  // Assign a different sub-set of DD blocks to each OpenMP thread
  // Computing the offset of DD-blocks indices and nr of such DD blocks, per color,
  // depending on the nr of OpenMP threads available

  int *nr_thrDD_blocks_notin_comms = s->nr_thrDD_blocks_notin_comms_;
  int *nr_thrDD_blocks_in_comms = s->nr_thrDD_blocks_in_comms_;
  int *DD_thr_offset_notin_comms = s->DD_thr_offset_notin_comms_;
  int *DD_thr_offset_in_comms = s->DD_thr_offset_in_comms_;  

  // VALIDATE that these are actually the right variables now.
  nr_thrDD_blocks_notin_comms[0] = s->nr_thrDD_blocks_notin_comms_[0];
  nr_thrDD_blocks_notin_comms[1] = s->nr_thrDD_blocks_notin_comms_[1];
  nr_thrDD_blocks_in_comms[0]    = s->nr_thrDD_blocks_in_comms_[0];
  nr_thrDD_blocks_in_comms[1]    = s->nr_thrDD_blocks_in_comms_[1];

  DD_thr_offset_notin_comms[0]   = 0;
  DD_thr_offset_notin_comms[1]   = 0;
  DD_thr_offset_in_comms[0]      = 0;
  DD_thr_offset_in_comms[1]      = 0;

  // disabled for now !

  /*

  struct Thread *threading = l->threading;

  if( threading->n_core > 1 ){
    int rest_offset=0;
    for( int color=0; color<2; color++ ){
      // Non-"residual" nr of DD blocks
      DD_thr_offset_notin_comms[color] = s->nr_DD_blocks_notin_comms[color] / threading->n_core;
      rest_offset = s->nr_DD_blocks_notin_comms[color] - DD_thr_offset_notin_comms[color] * threading->n_core;
      // Number of DD blocks, including "residual" (residual in the send of this simple division)
      nr_thrDD_blocks_notin_comms[color] = DD_thr_offset_notin_comms[color];
      if( threading->core < rest_offset ){ nr_thrDD_blocks_notin_comms[color]++; }
      // Include "residual" for these offsets
      DD_thr_offset_notin_comms[color] *= threading->core;
      if( threading->core < rest_offset ){
        DD_thr_offset_notin_comms[color] += threading->core;
      }
      else{
        DD_thr_offset_notin_comms[color] += rest_offset;
      }
      DD_thr_offset_in_comms[color] = s->nr_DD_blocks_in_comms[color] / threading->n_core;
      rest_offset = s->nr_DD_blocks_in_comms[color] - DD_thr_offset_in_comms[color] * threading->n_core;
      nr_thrDD_blocks_in_comms[color] = DD_thr_offset_in_comms[color];
      if( threading->core < rest_offset ){ nr_thrDD_blocks_in_comms[color]++; }
      DD_thr_offset_in_comms[color] *= threading->core;
      if( threading->core < rest_offset ){
        DD_thr_offset_in_comms[color] += threading->core;
      }
      else{
        DD_thr_offset_in_comms[color] += rest_offset;
      }
      // For core=0, the offset is always zero
      if( threading->core == 0 ){
        DD_thr_offset_notin_comms[color] = 0;
        DD_thr_offset_in_comms[color] = 0;
      }
    }
  }

  */

  //-------------------------------------------------------------------------------------------------------------------------
  // copying (s->op.c).boundary_table[inv_mu_dir] ---> for ghost exchanges !

  // TODO: are the following copyings really necessary ?

  int mu_dir, inv_mu_dir, mu, boundary_size;

  for( mu=0; mu<4; mu++ ){
    //-1
    dir = -1;
    mu_dir = 2*mu-MIN(dir,0);
    inv_mu_dir = 2*mu+1+MIN(dir,0);
    boundary_size = (s->op.c).num_boundary_sites[mu_dir];
    cuda_safe_call( cudaMemcpy( (s->op.c).boundary_table_gpu[inv_mu_dir], (s->op.c).boundary_table[inv_mu_dir],
                                boundary_size*sizeof(int), cudaMemcpyHostToDevice ) );
    //+1
    dir = 1;
    mu_dir = 2*mu-MIN(dir,0);
    inv_mu_dir = 2*mu+1+MIN(dir,0);
    boundary_size = (s->op.c).num_boundary_sites[mu_dir];
    cuda_safe_call( cudaMemcpy( (s->op.c).boundary_table_gpu[inv_mu_dir], (s->op.c).boundary_table[inv_mu_dir],
                                boundary_size*sizeof(int), cudaMemcpyHostToDevice ) );
  }

}


void schwarz_PRECISION_CUDA( vector_PRECISION phi, vector_PRECISION D_phi, vector_PRECISION eta, const int cycles, int res,
                             schwarz_PRECISION_struct *s, level_struct *l, struct Thread *threading ) {

  // TODO-s:

  //	2. reconsider all the calls to cudaDeviceSynchronize() and
  //	   SYNC_CORES(threading): reduce synchronizations as much
  //	   as possible (go over all of the FIXME-s)
  //	3. add local profiling
  //	4. fix unknown (minor) memory leakage

  //printf("(%d) very beginning of SAP...\n", g.my_rank);

  START_NO_HYPERTHREADS(threading)

  // Assign a different sub-set of DD blocks to each OpenMP thread
  // Computing the offset of DD-blocks indices and nr of such DD blocks, per color,
  // depending on the nr of OpenMP threads available

  int *nr_thrDD_blocks_notin_comms = s->nr_thrDD_blocks_notin_comms_;
  int *nr_thrDD_blocks_in_comms = s->nr_thrDD_blocks_in_comms_;
  int *DD_thr_offset_notin_comms = s->DD_thr_offset_notin_comms_;
  int *DD_thr_offset_in_comms = s->DD_thr_offset_in_comms_;  

  int color, k, mu, nb = s->num_blocks, init_res = res, i;

  // spinors on the GPU to be used on computations
  cuda_vector_PRECISION r_dev = (s->cu_s).buf1, x_dev = (s->cu_s).buf3,
                        latest_iter_dev = (s->cu_s).buf2, Dphi_dev = (s->cu_s).buf4,
                        eta_dev = (s->cu_s).buf5, D_phi_dev = (s->cu_s).buf6;

  // disabling thread splitting for now
  int nb_thread_start = 0;
  int nb_thread_end = nb;
  //compute_core_start_end_custom(0, nb, &nb_thread_start, &nb_thread_end, l, threading, 1);

  // Creation of CUDA Events, for time measurement and GPU sync
  cudaEvent_t start_event_copy, stop_event_copy, start_event_comp, stop_event_comp;
  cuda_safe_call( cudaEventCreate(&start_event_copy) );
  cuda_safe_call( cudaEventCreate(&stop_event_copy) );
  cuda_safe_call( cudaEventCreate(&start_event_comp) );
  cuda_safe_call( cudaEventCreate(&stop_event_comp) );
  cudaStream_t *streams_schwarz = s->streams;

  cuda_vector_PRECISION_copy( (void*)eta_dev, (void*)eta, nb_thread_start*s->block_vector_size,
                              (nb_thread_end-nb_thread_start)*s->block_vector_size, l, _H2D, _CUDA_SYNC,
                              threading->core, streams_schwarz );

  if ( res == _NO_RES ) {
    cuda_vector_PRECISION_copy( (void*)r_dev, (void*)eta_dev, nb_thread_start*s->block_vector_size,
                                (nb_thread_end-nb_thread_start)*s->block_vector_size, l, _D2D, _CUDA_SYNC,
                                threading->core, streams_schwarz );
    cuda_vector_PRECISION_define( x_dev, make_cu_cmplx_PRECISION(0,0), nb_thread_start*s->block_vector_size,
                                  (nb_thread_end-nb_thread_start)*s->block_vector_size, l, _CUDA_SYNC,
                                  threading->core, streams_schwarz );
  } else {
    cuda_vector_PRECISION_copy( (void*)x_dev, (void*)phi, nb_thread_start*s->block_vector_size,
                                (nb_thread_end-nb_thread_start)*s->block_vector_size, l, _H2D, _CUDA_SYNC,
                                threading->core, streams_schwarz );
  }

  //START_MASTER(threading)
  if ( res == _NO_RES ) {
    // setting boundary values to zero
    cuda_vector_PRECISION_define( x_dev, make_cu_cmplx_PRECISION(0,0), l->inner_vector_size,
                                  (l->schwarz_vector_size - l->inner_vector_size), l, _CUDA_SYNC,
                                  threading->core, streams_schwarz );
  }
  //END_MASTER(threading)

  // FIXME ?
  //cuda_safe_call( cudaDeviceSynchronize() );
  //SYNC_CORES(threading)

  //printf("(%d) BEFORE!! \n", g.my_rank);

  for ( k=0; k<cycles; k++ ) {
    for ( color=0; color<s->num_colors; color++ ) {
      if ( res == _RES ) {
        //START_LOCKED_MASTER(threading)
        for ( mu=0; mu<4; mu++ ) {
          cuda_ghost_update_PRECISION( (k==0 && init_res == _RES)?x_dev:latest_iter_dev, mu, +1, &(s->op.c), l );
          cuda_ghost_update_PRECISION( (k==0 && init_res == _RES)?x_dev:latest_iter_dev, mu, -1, &(s->op.c), l );
        }
        //END_LOCKED_MASTER(threading)
      } else {
        // we need a barrier between black and white blocks
        // FIXME ?
        //SYNC_CORES(threading)
        cuda_safe_call( cudaDeviceSynchronize() );
      }

      if ( res == _RES ) {
        if ( k==0 && init_res == _RES ) {

          //cuda_block_d_plus_clover_PRECISION( Dphi_dev, x_dev, nr_thrDD_blocks_notin_comms[color], s, l,
          //                                    no_threading, threading->core, streams_schwarz, color,
          //                                    (s->cu_s).DD_blocks_notin_comms[color] + DD_thr_offset_notin_comms[color],
          //                                    s->DD_blocks_notin_comms[color] +DD_thr_offset_notin_comms[color] );

          //cuda_block_PRECISION_boundary_op( Dphi_dev, x_dev, nr_thrDD_blocks_notin_comms[color], s, l,
          //                                  no_threading, threading->core, streams_schwarz, color,
          //                                  (s->cu_s).DD_blocks_notin_comms[color] + DD_thr_offset_notin_comms[color],
          //                                  s->DD_blocks_notin_comms[color] +DD_thr_offset_notin_comms[color] );

          //cuda_block_vector_PRECISION_minus( r_dev, eta_dev, Dphi_dev, nr_thrDD_blocks_notin_comms[color], s, l,
          //                                   no_threading, threading->core, streams_schwarz, color,
          //                                   (s->cu_s).DD_blocks_notin_comms[color] + DD_thr_offset_notin_comms[color],
          //                                   s->DD_blocks_notin_comms[color] + DD_thr_offset_notin_comms[color] );

          cuda_block_d_plus_clover_PRECISION( Dphi_dev, x_dev, s->nr_DD_blocks_notin_comms[color], s, l,
                                              no_threading, 0, streams_schwarz, color,
                                              (s->cu_s).DD_blocks_notin_comms[color], s->DD_blocks_notin_comms[color] );
          cuda_block_PRECISION_boundary_op( Dphi_dev, x_dev, s->nr_DD_blocks_notin_comms[color], s, l,
                                            no_threading, 0, streams_schwarz, color,
                                            (s->cu_s).DD_blocks_notin_comms[color], s->DD_blocks_notin_comms[color] );
          cuda_block_vector_PRECISION_minus( r_dev, eta_dev, Dphi_dev, s->nr_DD_blocks_notin_comms[color], s, l,
                                             no_threading, 0, streams_schwarz, color,
                                             (s->cu_s).DD_blocks_notin_comms[color], s->DD_blocks_notin_comms[color] );

        } else {

          //cuda_n_block_PRECISION_boundary_op( r_dev, latest_iter_dev, nr_thrDD_blocks_notin_comms[color], s, l,
          //                                    no_threading, threading->core, streams_schwarz, color,
          //                                    (s->cu_s).DD_blocks_notin_comms[color] + DD_thr_offset_notin_comms[color],
          //                                    s->DD_blocks_notin_comms[color] + DD_thr_offset_notin_comms[color] );

          cuda_n_block_PRECISION_boundary_op( r_dev, latest_iter_dev, s->nr_DD_blocks_notin_comms[color], s, l,
                                              no_threading, 0, streams_schwarz, color,
                                              (s->cu_s).DD_blocks_notin_comms[color], s->DD_blocks_notin_comms[color] );

        }
      }

      // local minres updates x, r and latest iter
      //cuda_block_solve_oddeven_PRECISION( (cuda_vector_PRECISION)x_dev, (cuda_vector_PRECISION)r_dev,
      //                                    (cuda_vector_PRECISION)latest_iter_dev, 0, nr_thrDD_blocks_notin_comms[color],
      //                                    s, l, no_threading, threading->core, streams_schwarz, 0, color,
      //                                    (s->cu_s).DD_blocks_notin_comms[color] + DD_thr_offset_notin_comms[color],
      //                                    s->DD_blocks_notin_comms[color] + DD_thr_offset_notin_comms[color] );

      cuda_block_solve_oddeven_PRECISION( (cuda_vector_PRECISION)x_dev, (cuda_vector_PRECISION)r_dev,
                                          (cuda_vector_PRECISION)latest_iter_dev, 0, s->nr_DD_blocks_notin_comms[color],
                                          s, l, no_threading, 0, streams_schwarz, 0, color,
                                          (s->cu_s).DD_blocks_notin_comms[color], s->DD_blocks_notin_comms[color] );

      if ( res == _RES ) {
        //START_LOCKED_MASTER(threading)
        for ( mu=0; mu<4; mu++ ) {
          cuda_ghost_update_wait_PRECISION( (k==0 && init_res == _RES)?x_dev:latest_iter_dev, mu, +1, &(s->op.c), l );
          cuda_ghost_update_wait_PRECISION( (k==0 && init_res == _RES)?x_dev:latest_iter_dev, mu, -1, &(s->op.c), l );
          //cuda_safe_call( cudaDeviceSynchronize() );
        }
        //END_LOCKED_MASTER(threading)
      } else {
        // we need a barrier between black and white blocks
        // FIXME ?
        //SYNC_CORES(threading)
        //cuda_safe_call( cudaDeviceSynchronize() );
      }

      // FIXME ?
      //SYNC_CORES(threading)
      //cuda_safe_call( cudaDeviceSynchronize() );

      if ( res == _RES ) {
        if ( k==0 && init_res == _RES ) {

          //cuda_block_d_plus_clover_PRECISION( Dphi_dev, x_dev, nr_thrDD_blocks_in_comms[color], s, l, no_threading, threading->core,
          //                                    streams_schwarz, color,
          //                                    (s->cu_s).DD_blocks_in_comms[color] + DD_thr_offset_in_comms[color],
          //                                    s->DD_blocks_in_comms[color] + DD_thr_offset_in_comms[color] );

          //cuda_block_PRECISION_boundary_op( Dphi_dev, x_dev, nr_thrDD_blocks_in_comms[color], s, l, no_threading, threading->core,
          //                                  streams_schwarz, color,
          //                                  (s->cu_s).DD_blocks_in_comms[color] + DD_thr_offset_in_comms[color],
          //                                  s->DD_blocks_in_comms[color] + DD_thr_offset_in_comms[color] );

          //cuda_block_vector_PRECISION_minus( r_dev, eta_dev, Dphi_dev, nr_thrDD_blocks_in_comms[color], s, l, no_threading, threading->core,
          //                                   streams_schwarz, color,
          //                                   (s->cu_s).DD_blocks_in_comms[color] + DD_thr_offset_in_comms[color],
          //                                   s->DD_blocks_in_comms[color] + DD_thr_offset_in_comms[color] );

          cuda_block_d_plus_clover_PRECISION( Dphi_dev, x_dev, s->nr_DD_blocks_in_comms[color], s, l, no_threading, 0,
                                              streams_schwarz, color, (s->cu_s).DD_blocks_in_comms[color],
                                              s->DD_blocks_in_comms[color] );
          cuda_block_PRECISION_boundary_op( Dphi_dev, x_dev, s->nr_DD_blocks_in_comms[color], s, l, no_threading, 0,
                                            streams_schwarz, color, (s->cu_s).DD_blocks_in_comms[color],
                                            s->DD_blocks_in_comms[color] );
          cuda_block_vector_PRECISION_minus( r_dev, eta_dev, Dphi_dev, s->nr_DD_blocks_in_comms[color], s, l, no_threading, 0,
                                             streams_schwarz, color, (s->cu_s).DD_blocks_in_comms[color],
                                             s->DD_blocks_in_comms[color] );

        } else {

          //cuda_n_block_PRECISION_boundary_op( r_dev, latest_iter_dev, nr_thrDD_blocks_in_comms[color], s, l, no_threading, threading->core,
          //                                    streams_schwarz, color,
          //                                    (s->cu_s).DD_blocks_in_comms[color] + DD_thr_offset_in_comms[color],
          //                                    s->DD_blocks_in_comms[color] + DD_thr_offset_in_comms[color] );

          cuda_n_block_PRECISION_boundary_op( r_dev, latest_iter_dev, s->nr_DD_blocks_in_comms[color], s, l, no_threading, 0,
                                              streams_schwarz, color, (s->cu_s).DD_blocks_in_comms[color],
                                              s->DD_blocks_in_comms[color] );

        }
      }

      // local minres updates x, r and latest iter
      //cuda_block_solve_oddeven_PRECISION( (cuda_vector_PRECISION)x_dev, (cuda_vector_PRECISION)r_dev,
      //                                    (cuda_vector_PRECISION)latest_iter_dev, 0, nr_thrDD_blocks_in_comms[color],
      //                                    s, l, no_threading, threading->core, streams_schwarz, 0, color,
      //                                    (s->cu_s).DD_blocks_in_comms[color] + DD_thr_offset_in_comms[color],
      //                                    s->DD_blocks_in_comms[color] + DD_thr_offset_in_comms[color] );

      cuda_block_solve_oddeven_PRECISION( (cuda_vector_PRECISION)x_dev, (cuda_vector_PRECISION)r_dev,
                                          (cuda_vector_PRECISION)latest_iter_dev, 0, s->nr_DD_blocks_in_comms[color],
                                          s, l, no_threading, 0, streams_schwarz, 0, color,
                                          (s->cu_s).DD_blocks_in_comms[color], s->DD_blocks_in_comms[color] );

      res = _RES;
    }
  }

  //printf("(%d) AFTER!! \n", g.my_rank);

  // FIXME ?
  cuda_safe_call( cudaDeviceSynchronize() );
  //SYNC_CORES(threading)

  if ( l->relax_fac != 1.0 ){
    cuda_vector_PRECISION_copy( (void*)phi, (void*)x_dev, nb_thread_start*s->block_vector_size,
                                (nb_thread_end-nb_thread_start)*s->block_vector_size, l, _D2H, _CUDA_SYNC, threading->core, streams_schwarz );
    for ( i=nb_thread_start; i<nb_thread_end; i++ ) {
      vector_PRECISION_scale( phi, phi, l->relax_fac, s->block[i].start*l->num_lattice_site_var,
                              s->block[i].start*l->num_lattice_site_var+s->block_vector_size, l );
    }
  } else {
    cuda_vector_PRECISION_copy( (void*)phi, (void*)x_dev, nb_thread_start*s->block_vector_size,
                                (nb_thread_end-nb_thread_start)*s->block_vector_size, l, _D2H, _CUDA_SYNC, threading->core, streams_schwarz );
  }

  if( D_phi != NULL ){

    START_LOCKED_MASTER(threading)
    for ( mu=0; mu<4; mu++ ) {
      cuda_ghost_update_PRECISION( latest_iter_dev, mu, +1, &(s->op.c), l );
      cuda_ghost_update_PRECISION( latest_iter_dev, mu, -1, &(s->op.c), l );
    }
    END_LOCKED_MASTER(threading)

    SYNC_CORES(threading)

    // 1. for all those DD blocks of color=0 and NOT INVOLVED in comms: n_boundary_op( ... ), vector_PRECISION_minus( ... ),
    // vector_PRECISION_scale( ... )

    //cuda_n_block_PRECISION_boundary_op( r_dev, latest_iter_dev, nr_thrDD_blocks_notin_comms[0], s, l, no_threading, threading->core,
    //                                    streams_schwarz, 0,
    //                                    (s->cu_s).DD_blocks_notin_comms[0] + DD_thr_offset_notin_comms[0],
    //                                    s->DD_blocks_notin_comms[0] + DD_thr_offset_notin_comms[0] );

    //cuda_block_vector_PRECISION_minus( D_phi_dev, eta_dev, r_dev, nr_thrDD_blocks_notin_comms[0], s, l, no_threading, threading->core,
    //                                   streams_schwarz, 0,
    //                                   (s->cu_s).DD_blocks_notin_comms[0] + DD_thr_offset_notin_comms[0],
    //                                   s->DD_blocks_notin_comms[0] + DD_thr_offset_notin_comms[0] );

    //if( l->relax_fac != 1.0 ){
    //  cuda_block_vector_PRECISION_scale( D_phi_dev, D_phi_dev, make_cu_cmplx_PRECISION( l->relax_fac, 0 ),
    //                                     nr_thrDD_blocks_notin_comms[0], s, l, no_threading, threading->core,
    //                                     streams_schwarz, 0,
    //                                     (s->cu_s).DD_blocks_notin_comms[0] + DD_thr_offset_notin_comms[0],
    //                                     s->DD_blocks_notin_comms[0] + DD_thr_offset_notin_comms[0] );
    //}

    cuda_n_block_PRECISION_boundary_op( r_dev, latest_iter_dev, s->nr_DD_blocks_notin_comms[0], s, l, no_threading, 0,
                                        streams_schwarz, 0, (s->cu_s).DD_blocks_notin_comms[0], s->DD_blocks_notin_comms[0] );
    cuda_block_vector_PRECISION_minus( D_phi_dev, eta_dev, r_dev, s->nr_DD_blocks_notin_comms[0], s, l, no_threading, 0,
                                       streams_schwarz, 0, (s->cu_s).DD_blocks_notin_comms[0], s->DD_blocks_notin_comms[0] );
    if( l->relax_fac != 1.0 ){
      cuda_block_vector_PRECISION_scale( D_phi_dev, D_phi_dev, make_cu_cmplx_PRECISION( l->relax_fac, 0 ),
                                         s->nr_DD_blocks_notin_comms[0], s, l, no_threading, 0,
                                         streams_schwarz, 0, (s->cu_s).DD_blocks_notin_comms[0], s->DD_blocks_notin_comms[0] );
    }

    // FIXME ?
    cuda_safe_call( cudaDeviceSynchronize() );
    //SYNC_CORES(threading)

    // 2. for all those DD blocks of color=1: vector_PRECISION_minus( ... ), vector_PRECISION_scale( ... )

    //cuda_safe_call( cudaDeviceSynchronize() );
    //SYNC_CORES(threading)

    cuda_block_vector_PRECISION_minus( D_phi_dev, eta_dev, r_dev, nr_thrDD_blocks_notin_comms[1],
                                       s, l, no_threading, threading->core,
                                       streams_schwarz, 1,
                                       (s->cu_s).DD_blocks_notin_comms[1] + DD_thr_offset_notin_comms[1],
                                       s->DD_blocks_notin_comms[1] + DD_thr_offset_notin_comms[1] );

    cuda_block_vector_PRECISION_minus( D_phi_dev, eta_dev, r_dev, nr_thrDD_blocks_in_comms[1],
                                       s, l, no_threading, threading->core,
                                       streams_schwarz, 1,
                                       (s->cu_s).DD_blocks_in_comms[1] + DD_thr_offset_in_comms[1],
                                       s->DD_blocks_in_comms[1] + DD_thr_offset_in_comms[1] );

    if( l->relax_fac != 1.0 ){

      cuda_block_vector_PRECISION_scale( D_phi_dev, D_phi_dev, make_cu_cmplx_PRECISION( l->relax_fac, 0 ),
                                         nr_thrDD_blocks_in_comms[1],
                                         s, l, no_threading, threading->core,
                                         streams_schwarz, 1,
                                         (s->cu_s).DD_blocks_in_comms[1] + DD_thr_offset_in_comms[1],
                                         s->DD_blocks_in_comms[1] + DD_thr_offset_in_comms[1] );

      cuda_block_vector_PRECISION_scale( D_phi_dev, D_phi_dev, make_cu_cmplx_PRECISION( l->relax_fac, 0 ),
                                         nr_thrDD_blocks_notin_comms[1],
                                         s, l, no_threading, threading->core,
                                         streams_schwarz, 1,
                                         (s->cu_s).DD_blocks_notin_comms[1] + DD_thr_offset_notin_comms[1],
                                         s->DD_blocks_notin_comms[1] + DD_thr_offset_notin_comms[1] );

      //cuda_block_vector_PRECISION_scale( D_phi_dev, D_phi_dev, make_cu_cmplx_PRECISION( l->relax_fac, 0 ),
      //                                   s->nr_DD_blocks[1], s, l, no_threading, threading->core,
      //                                   streams_schwarz, 1, (s->cu_s).DD_blocks[1], s->DD_blocks[1] );
    }

    //START_LOCKED_MASTER(threading)
    for ( mu=0; mu<4; mu++ ) {
      cuda_ghost_update_wait_PRECISION( latest_iter_dev, mu, +1, &(s->op.c), l );
      cuda_ghost_update_wait_PRECISION( latest_iter_dev, mu, -1, &(s->op.c), l );
    }
    //END_LOCKED_MASTER(threading)

    // FIXME ?
    cuda_safe_call( cudaDeviceSynchronize() );
    //SYNC_CORES(threading)

    // 3. for all those DD blocks of color=0 and INVOLVED in comms: n_boundary_op( ... ), vector_PRECISION_minus( ... ),
    // vector_PRECISION_scale( ... )

    //cuda_n_block_PRECISION_boundary_op( r_dev, latest_iter_dev, nr_thrDD_blocks_in_comms[0], s, l, no_threading, threading->core,
    //                                    streams_schwarz, 0,
    //                                    (s->cu_s).DD_blocks_in_comms[0] + DD_thr_offset_in_comms[0],
    //                                    s->DD_blocks_in_comms[0] + DD_thr_offset_in_comms[0] );

    //cuda_block_vector_PRECISION_minus( D_phi_dev, eta_dev, r_dev, nr_thrDD_blocks_in_comms[0], s, l, no_threading, threading->core,
    //                                   streams_schwarz, 0,
    //                                   (s->cu_s).DD_blocks_in_comms[0] + DD_thr_offset_in_comms[0],
    //                                   s->DD_blocks_in_comms[0] + DD_thr_offset_in_comms[0] );

    //if( l->relax_fac != 1.0 ){
    //  cuda_block_vector_PRECISION_scale( D_phi_dev, D_phi_dev, make_cu_cmplx_PRECISION( l->relax_fac, 0 ),
    //                                     nr_thrDD_blocks_in_comms[0], s, l, no_threading, threading->core,
    //                                     streams_schwarz, 0,
    //                                     (s->cu_s).DD_blocks_in_comms[0] + DD_thr_offset_in_comms[0],
    //                                     s->DD_blocks_in_comms[0] + DD_thr_offset_in_comms[0] );
    //}

    cuda_n_block_PRECISION_boundary_op( r_dev, latest_iter_dev, s->nr_DD_blocks_in_comms[0], s, l, no_threading, 0,
                                        streams_schwarz, 0, (s->cu_s).DD_blocks_in_comms[0], s->DD_blocks_in_comms[0] );
    cuda_block_vector_PRECISION_minus( D_phi_dev, eta_dev, r_dev, s->nr_DD_blocks_in_comms[0], s, l, no_threading, 0, \
                                       streams_schwarz, 0, (s->cu_s).DD_blocks_in_comms[0], s->DD_blocks_in_comms[0] );
    if( l->relax_fac != 1.0 ){
      cuda_block_vector_PRECISION_scale( D_phi_dev, D_phi_dev, make_cu_cmplx_PRECISION( l->relax_fac, 0 ),
                                         s->nr_DD_blocks_in_comms[0], s, l, no_threading, 0,
                                         streams_schwarz, 0, (s->cu_s).DD_blocks_in_comms[0], s->DD_blocks_in_comms[0] );
    }

  }

  // FIXME ?
  cuda_safe_call( cudaDeviceSynchronize() );
  //SYNC_CORES(threading)

  if( D_phi != NULL ){
    cuda_vector_PRECISION_copy( (void*)D_phi, (void*)D_phi_dev, nb_thread_start*s->block_vector_size,
                                (nb_thread_end-nb_thread_start)*s->block_vector_size, l, _D2H, _CUDA_SYNC, threading->core,
                                streams_schwarz );
  }

  // FIXME ?
  cuda_safe_call( cudaDeviceSynchronize() );
  //SYNC_CORES(threading)

#ifdef SCHWARZ_RES
  // TODO: portion pending to port to GPU/CUDA
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

void schwarz_PRECISION_def_CUDA( schwarz_PRECISION_struct *s, operator_double_struct *op, level_struct *l ) {

  if( l->depth==0 && g.odd_even ){
    schwarz_PRECISION_alloc_CUDA( s, l );
    schwarz_PRECISION_setup_CUDA( s, op, l );
  }

}

#endif
