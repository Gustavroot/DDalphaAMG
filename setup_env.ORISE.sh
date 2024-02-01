#!/bin/bash

module purge

module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
module load compiler/rocm/dtk/23.04

# source /public/software/compiler/dtk/dtk-23.04/env.sh
source /public/software/compiler/dtk/dtk-23.04/cuda/env.sh
export C_INCLUDE_PATH=/public/software/compiler/dtk/dtk-23.04/cuda/include:${C_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=/public/software/compiler/dtk/dtk-23.04/cuda/include:${CPLUS_INCLUDE_PATH}

module load compiler/automake/1.15

