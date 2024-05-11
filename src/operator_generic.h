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

#ifndef OPERATOR_PRECISION_HEADER
#define OPERATOR_PRECISION_HEADER
#include "algorithm_structs_PRECISION.h"
#include "level_struct.h"

struct Thread;

void cpu_operator_PRECISION_init(operator_PRECISION_struct *op);
void cpu_operator_PRECISION_alloc(operator_PRECISION_struct *op, const int type, level_struct *l);
void operator_PRECISION_define(operator_PRECISION_struct *op, level_struct *l);
void cpu_operator_PRECISION_free(operator_PRECISION_struct *op, const int type, level_struct *l);

void operator_PRECISION_test_routine(operator_PRECISION_struct *op, level_struct *l, struct Thread *threading);


#endif
