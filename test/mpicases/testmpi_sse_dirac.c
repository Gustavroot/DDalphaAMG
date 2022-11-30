#include "dirac_double.h"
#include "alloc_control.h"
#include "complex_types_double.h"
#include "threading.h"
#include "ghost.h"
#include "init.h"
#include "global_struct.h"
#include "io.h"
#include "dirac.h"
#include "top_level.h"

int main()
{
    int argc = 2;
    char **argv = NULL;

    struct common_thread_data *commonthreaddata;

    MALLOC(argv, char *, 2);
    MALLOC(argv[0], char, 1);
    argv[0][0] = '\0';
    argv[1] = "test/test.ini";

    vector_double eta = NULL;
    vector_double phi = NULL;
    level_struct l;

    config_double hopp = NULL;

    MALLOC(eta, complex_double, 12);
    MALLOC(phi, complex_double, 12);

    MPI_Init(&argc, &argv);
    predefine_rank();
    method_init(&argc, &argv, &l);

    MALLOC(no_threading, Thread, 1);
    setup_no_threading(no_threading, &l);

    // populate global state for clover and hopper term
    MALLOC(hopp, complex_double, 3 * l.inner_vector_size);
    read_conf((double *)(hopp), g.in, &(g.plaq_hopp), &l);

    dirac_setup(hopp, NULL, &l);

    FREE(hopp, complex_double, 3 * l.inner_vector_size);

    MALLOC(commonthreaddata, struct common_thread_data, 1);
    init_common_thread_data(commonthreaddata);

#pragma omp parallel num_threads(g.num_openmp_processes)
    {
        g.on_solve=0;
        struct Thread threading;
        l.threading = &threading;
        setup_threading(&threading, commonthreaddata, &l);
        setup_no_threading(no_threading, &l);

        // TODO: move this line to a better place !
        g.nr_threads = threading.n_core;

        method_setup( NULL, &l, &threading );

        method_update( l.setup_iter, &l, &threading );
        
        g.on_solve=1;
        solve_driver(&l, &threading );
        // d_plus_clover_double(eta, phi, &g.op_double, &l, no_threading);
    }

    finalize_common_thread_data(commonthreaddata);
    finalize_no_threading(no_threading);

    method_free(&l);
    method_finalize(&l);

    MPI_Finalize();

    FREE(eta, complex_double, 1);
    FREE(phi, complex_double, 1);

    FREE(commonthreaddata, struct common_thread_data, 1);
    FREE(no_threading, Thread, 1);

    FREE(argv[0], char, 1);
    FREE(argv, char *, 2);
    return 0;
}