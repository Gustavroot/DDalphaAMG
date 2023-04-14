#ifndef ALGORITHM_STRUCTS_H
#define ALGORITHM_STRUCTS_H

#include "algorithm_structs_double.h"
#include "algorithm_structs_float.h"

typedef struct {
  gmres_float_struct float_section;
  gmres_double_struct double_section;
} gmres_MP_struct;

#endif  // ALGORITHM_STRUCTS_H
