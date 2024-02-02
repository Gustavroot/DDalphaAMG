#ifndef LEVEL_STRUCT_H
#define LEVEL_STRUCT_H

#ifndef IMPORT_FROM_EXTERN_C
#include <mpi.h>
#endif
#include "algorithm_structs_float.h"
#include "algorithm_structs_double.h"
#include "profiling_old_float.h"
#include "profiling_old_double.h"


typedef struct level_struct
{

  // distributed: non-idling processes of previos level
  // gathered: non-idling processes of current level

  // distributed
  operator_double_struct op_double;
  operator_float_struct op_float;
  // odd_even
  operator_double_struct oe_op_double;
  operator_float_struct oe_op_float;
  // gathered / schwarz
  schwarz_double_struct s_double;
  schwarz_float_struct s_float;
  // interpolation / aggregation
  interpolation_double_struct is_double;
  interpolation_float_struct is_float;
  // gathering parameters and buffers
  gathering_double_struct gs_double;
  gathering_float_struct gs_float;
  // k cycle
  gmres_float_struct p_float;
  gmres_double_struct p_double;
  // gmres as a smoother
  gmres_float_struct sp_float;
  gmres_double_struct sp_double;
  // dummy gmres struct
  gmres_float_struct dummy_p_float;
  gmres_double_struct dummy_p_double;
  // profiling
  profiling_float_struct prof_float;
  profiling_double_struct prof_double;

  // communication
  MPI_Request *reqs;
  int parent_rank, idle, neighbor_rank[8], num_processes, num_processes_dir[4];
  // lattice
  int *global_lattice;
  int *local_lattice;
  int *block_lattice;
  int num_eig_vect;
  int coarsening[4];
  int global_splitting[4];
  int periodic_bc[4];
  int comm_offset[4];
  // degrees of freedom on a site
  // 12 on fine lattice (i.e., complex d.o.f.)
  // 2*num_eig_vect on coarser lattices
  int num_lattice_site_var;
  int level;
  int depth;
  // number of sites in local volume + ghost shell (either fw or bw)
  int num_lattice_sites;
  // number of sites in local volume
  int num_inner_lattice_sites;
  int num_boundary_sites[4];
  // complex d.o.f. in local volume + ghost shell = num_lattice_sites * num_lattice_site_var
  int vector_size;
  // complex d.o.f. in local volume = num_inner_lattice_sites * num_lattice_site_var
  int inner_vector_size;
  int schwarz_vector_size;
  int D_size;
  int clover_size;
  // operator
  double real_shift;
  complex_double dirac_shift, even_shift, odd_shift;
  // buffer vectors
  vector_float vbuf_float[9], sbuf_float[2];
  vector_double vbuf_double[9], sbuf_double[2];
  // storage + daggered-operator bufferes
  vector_double x;
  // local solver parameters
  double tol, relax_fac;
  int n_cy, post_smooth_iter, block_iter, setup_iter;

  // next coarser level
  struct level_struct *next_level;

  struct Thread *threading;

#if defined(GCRODR) || defined(POLYPREC)
  // 'bool', if on H will be copied
  int dup_H;
#endif

} level_struct;

#endif // LEVEL_STRUCT_H
