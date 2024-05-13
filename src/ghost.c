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
 
void neighbor_define( level_struct *l ) {
  
  int mu, neighbor_coords[4];
  
  for ( mu=0; mu<4; mu++ ) {
    neighbor_coords[mu] = g.my_coords[mu];
  }
  
  for ( mu=0; mu<4; mu++ ) {    
    neighbor_coords[mu]+=l->comm_offset[mu];
    MPI_Cart_rank( g.comm_cart, neighbor_coords, &(l->neighbor_rank[2*mu]) );
    neighbor_coords[mu]-=2*l->comm_offset[mu];
    MPI_Cart_rank( g.comm_cart, neighbor_coords, &(l->neighbor_rank[2*mu+1]) );
    neighbor_coords[mu]+=l->comm_offset[mu];
  }
}


void predefine_rank( void ) {
  MPI_Comm_rank( MPI_COMM_WORLD, &(g.my_rank) );
}


void cart_define( level_struct *l ) {
  
  int mu, num_processes;
  
  MPI_Comm_size( MPI_COMM_WORLD, &num_processes ); 
  if (num_processes != g.num_processes) {
    error0("Error: Number of processes has to be %d\n", g.num_processes);
  }
  MPI_Cart_create( MPI_COMM_WORLD, 4, l->global_splitting, l->periodic_bc, 1, &(g.comm_cart) );
  MPI_Comm_rank( g.comm_cart, &(g.my_rank) );
  MPI_Cart_coords( g.comm_cart, g.my_rank, 4, g.my_coords );
  
  for ( mu=0; mu<4; mu++ ) {
    l->num_processes_dir[mu] = l->global_lattice[mu]/l->local_lattice[mu];
    l->comm_offset[mu] = 1;
  }

  neighbor_define( l );
  MPI_Comm_group( g.comm_cart, &(g.global_comm_group) );

#ifdef CUDA_OPT
  // Set the device for each MPI process, in correspondence with
  // the <local rank>

  // Based on:
  //		https://cvw.cac.cornell.edu/MPIAdvTopics/splitting
  //		https://stackoverflow.com/questions/27908813/requirements-for-use-of-cuda-aware-mpi (MPI needs to be CUDA-aware)
  //		6-Wrap-Up.pdf ----> GPU notes from 1st STIMULATE workshop
  int local_rank = 0;
  g.num_devices = 0;

  // one alternative to get the <local rank>
  //local_rank = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));

  // another alternative to get the <local rank>
  MPI_Comm loc_comm;
  MPI_Comm_split_type( g.comm_cart, MPI_COMM_TYPE_SHARED, g.my_rank, MPI_INFO_NULL, &loc_comm );
  MPI_Comm_rank( loc_comm, &local_rank );
  MPI_Comm_free( &loc_comm );

  cuda_safe_call( cudaGetDeviceCount( &(g.num_devices) ) );
  // Using pragma omp here to raise persistent thread-to-GPU linkage

#pragma omp parallel num_threads(g.num_openmp_processes)
  {
    cuda_safe_call( cudaSetDevice( local_rank % g.num_devices ) );
  }

  g.device_id = local_rank % g.num_devices;

  // Run async ghost exchanges once, to remove offset setup time introduced
  // by the MPI-Aware implementation of OpenMPI
  {
    MPI_Request setup_reqs[2];
    int dir, mu, mu_dir, inv_mu_dir;
    dir=1;
    mu=0;
    mu_dir = 2*mu-MIN(dir,0);
    inv_mu_dir = 2*mu+1+MIN(dir,0);

    if( l->neighbor_rank[mu_dir]==g.my_rank || l->neighbor_rank[inv_mu_dir]==g.my_rank ){
      //error0("mu=%d is not a good direction for ghost-exch initial test! FIXME!\n", mu);
      //warning0("mu=%d is not a good direction for ghost-exch initial test! FIXME!\n", mu);
    }

    // buffers for test exchange --> this text exchange is necessary to 'eliminate' an initial overhead
    float *recv_buff;
    float *send_buff;

    // 8 is just to make sure we don't go over mem size
    cuda_safe_call( cudaMalloc( (void**) (&( recv_buff )), 16*sizeof(float) ) );
    cuda_safe_call( cudaMalloc( (void**) (&( send_buff )), 16*sizeof(float) ) );
    cuda_safe_call( cudaMemset(send_buff, 0, 16*sizeof(float)) );

    MPI_Isend( send_buff, 2, MPI_FLOAT,
               l->neighbor_rank[inv_mu_dir], mu_dir, g.comm_cart, &(setup_reqs[1]) );

    MPI_Irecv( recv_buff, 2, MPI_FLOAT,
               l->neighbor_rank[mu_dir], mu_dir, g.comm_cart, &(setup_reqs[0]) );

    MPI_Wait( &(setup_reqs[0]), MPI_STATUS_IGNORE );
    MPI_Wait( &(setup_reqs[1]), MPI_STATUS_IGNORE );

    // release test buffers
    cuda_safe_call( cudaFree( recv_buff ) );
    cuda_safe_call( cudaFree( send_buff ) );
  }
#endif
}


void cart_free( level_struct *l ) {
  MPI_Group_free( &(g.global_comm_group) );
  MPI_Comm_free( &(g.comm_cart) );  
}
