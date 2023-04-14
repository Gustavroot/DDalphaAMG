#ifndef GLOBAL_STRUCT_H
#define GLOBAL_STRUCT_H

#ifndef IMPORT_FROM_EXTERN_C
#include <mpi.h>
#endif

#include <stdio.h>
#include "global_defs.h"
#include "algorithm_structs_double.h"
#include "algorithm_structs_float.h"
#include "algorithm_structs.h"
#include "var_table.h"
#include "dd_alpha_amg_parameters.h"
#include "dd_alpha_amg_setup_status.h"

typedef struct global_struct
{

    FILE *logfile;

    gmres_double_struct p;
    gmres_MP_struct p_MP;
    operator_double_struct op_double;
    operator_float_struct op_float;

    // communication
    MPI_Comm comm_cart;
    MPI_Group global_comm_group;
    MPI_Request sreqs[8], rreqs[8];
    int num_processes, my_rank, my_coords[4], two_cnfgs, tv_io_single_file, num_openmp_processes;
    // string buffers
    char in[STRINGLENGTH], in_clov[STRINGLENGTH], source_list[STRINGLENGTH], tv_io_file_name[STRINGLENGTH];
    // geometry, method parameters
    int num_levels, num_desired_levels, process_grid[4], in_format,
        **global_lattice, **local_lattice, **block_lattice,
        *post_smooth_iter, *block_iter, *setup_iter, *ncycle,
        method, odd_even, anti_pbc, rhs, propagator_coords[4],
        interpolation, randomize, *num_eig_vect, num_coarse_eig_vect, kcycle, mixed_precision,
        restart, max_restart, kcycle_restart, kcycle_max_restart, coarse_iter, coarse_restart;
    double tol, coarse_tol, kcycle_tol, csw, rho, *relax_fac;

    // profiling, analysis, output
    int coarse_iter_count, iter_count, iterator, print, conf_flag, setup_flag, in_setup;
    double coarse_time, prec_time, *output_table[8], cur_storage, max_storage, total_time,
        plaq_hopp, plaq_clov, norm_res, plaq, setup_m0, solve_m0, bicgstab_tol;

#ifdef CUDA_OPT
    double cur_gpu_storage, max_gpu_storage;

    // some CUDA-specific values
    int warp_size;
    int num_devices;
    int device_id;

    int *CUDA_threads_per_CUDA_block_type1;
    int *CUDA_threads_per_lattice_site_type1;
    int *CUDA_threads_per_CUDA_block_type2;
    int *CUDA_threads_per_lattice_site_type2;
#endif

    // index functions for external usage
    int (*conf_index_fct)(), (*vector_index_fct)();
    int *odd_even_table;

    // bc: 0 dirichlet, 1 periodic, 2 anti-periodic
    int bc;

    complex_double **gamma, g5D_shift;
    var_table vt;

    int on_solve;
    int nr_threads;

    struct dd_alpha_amg_parameters amg_params;
    struct dd_alpha_amg_setup_status mg_setup_status;
    double mass_for_next_solve;

} global_struct;

extern global_struct g;

#endif // GLOBAL_STRUCT_H
