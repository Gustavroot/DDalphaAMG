#include <criterion/criterion.h>
#include "dirac_double.h"
#include "alloc_control.h"
#include "complex_types_double.h"
#include "threading.h"
#include "ghost.h"
#include "init.h"
#include "global_struct.h"


Test(dirac, can_include)
{
}

Test(dirac, callSize1)
{
    int argc = 2;
    char ** argv = NULL;

    MALLOC(argv, char *, 2);
    MALLOC(argv[0], char, 1);
    argv[0][0] = '\0';
    argv[1] = "test/test.ini";

    vector_double eta = NULL;
    vector_double phi = NULL;
    level_struct l;
    Thread no_threading;

    MALLOC(eta, complex_double, 1);
    MALLOC(phi, complex_double, 1);

    MPI_Init(&argc, &argv);
    predefine_rank();
    method_init( &argc, &argv, &l );
    setup_no_threading(&no_threading, &l);

    #pragma omp parallel num_threads(2)
    {
        d_plus_clover_double(eta, phi, &g.op_double, &l, &no_threading);
    }

    FREE(eta, complex_double, 1);
    FREE(phi, complex_double, 1);
}