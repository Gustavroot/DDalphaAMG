#ifndef ALGORITHM_STRUCTS_H
#define ALGORITHM_STRUCTS_H

#include "algorithm_structs_double.h"
#include "algorithm_structs_float.h"

typedef struct
{
    gmres_float_struct sp;
    gmres_double_struct dp;
} gmres_MP_struct;

#endif // ALGORITHM_STRUCTS_H