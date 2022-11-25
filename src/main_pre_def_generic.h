/*
 * Copyright (C) 2016, Matthias Rottmann, Artur Strebel, Gustavo Ramirez, Simon Heybrock, Simone Bacchio, Bjoern Leder, Issaku Kanamori.
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

#ifndef MAIN_PRE_DEF_PRECISION_HEADER
  #define MAIN_PRE_DEF_PRECISION_HEADER

// FIXME temporary includes
#ifdef CUDA_OPT
#include "cuda_dirac_PRECISION.h"
#endif


// FIXME temporary includes
#include "communication_PRECISION.h"
#include "profiling_old_PRECISION.h"

#ifdef CUDA_OPT
//FIXME temporary inlcude
# include "cuda_schwarz_PRECISION.h"
#endif
  
#endif
