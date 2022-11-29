#include "dirac_double.h"
#include "alloc_control.h"
#include "complex_types_double.h"
#include "threading.h"
#include "ghost.h"
#include "init.h"

int main()
{
    int argc = 2;
    char **argv = NULL;

    MALLOC(argv, char *, 2);
    MALLOC(argv[0], char, 1);
    argv[0][0] = '\0';
    argv[1] = "test/test.ini";

    vector_double eta = NULL;
    vector_double phi = NULL;
    operator_double_struct op;
    level_struct l;
    Thread no_threading;

    MALLOC(eta, complex_double, 12);
    MALLOC(phi, complex_double, 12);

    MPI_Init(&argc, &argv);
    predefine_rank();
    method_init(&argc, &argv, &l);
    setup_no_threading(&no_threading, &l);

#pragma omp parallel num_threads(1)
    {
        d_plus_clover_double(eta, phi, &op, &l, &no_threading);
    }

    FREE(eta, complex_double, 1);
    FREE(phi, complex_double, 1);

    FREE(argv[0], char, 1);
    FREE(argv, char *, 2);
    return 0;
}