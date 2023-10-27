module purge

module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
module load compiler/rocm/dtk/23.04
# source /public/software/compiler/dtk/dtk-23.04/env.sh
source /public/software/compiler/dtk/dtk-23.04/cuda/env.sh
export C_INCLUDE_PATH=/public/software/compiler/dtk/dtk-23.04/cuda/include:${C_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=/public/software/compiler/dtk/dtk-23.04/cuda/include:${CPLUS_INCLUDE_PATH}


# module load compiler/devtoolset/9.3.1
# module load mpi/openmpi/4.1.5-gcc9.3.0
# DTK_PATH="/public/software/compiler/dtk/dtk-23.04"
# source ${DTK_PATH}/env.sh
# source ${DTK_PATH}/cuda/env.sh
# export C_INCLUDE_PATH=${DTK_PATH}/include:${C_INCLUDE_PATH}
# export C_INCLUDE_PATH=${DTK_PATH}/cuda/include:${C_INCLUDE_PATH}


# module load compiler/intel/2021.3.0
# module load mpi/intelmpi/2021.3.0
# module load mpi/hpcx/2.11.0/intel-2017.5.239
# module load compiler/intel/2017.5.239
# module load compiler/rocm/dtk/23.04
# DTK_PATH="/public/software/compiler/dtk/dtk-23.04"
# CMP_PATH="/opt/hpc/software/compiler/intel/intel-compiler-2017.5.239"
# MPI_PATH="/opt/hpc/software/mpi/hpcx/v2.11.0/intel-2017.5.239"
# source ${DTK_PATH}/env.sh
# source ${DTK_PATH}/cuda/env.sh
# export C_INCLUDE_PATH=${CMP_PATH}/include:${MPI_PATH}/include:${DTK_PATH}/include:${DTK_PATH}/cuda/include:${C_INCLUDE_PATH}


# module load compiler/intel/2021.3.0 
# module load mpi/intelmpi/2021.3.0
# DTK_PATH="/public/home/kelong/software/compiler/dtk/dtk-23.04.1"
# source ${DTK_PATH}/env.sh
# source ${DTK_PATH}/cuda/env.sh
# export C_INCLUDE_PATH=${DTK_PATH}/include:${C_INCLUDE_PATH}
# export C_INCLUDE_PATH=${DTK_PATH}/cuda/include:${C_INCLUDE_PATH}


# module load compiler/devtoolset/7.3.1
# module load mpi/hpcx/2.11.0/gcc-7.3.1
# module load compiler/rocm/dtk/23.04
# DTK_PATH="/public/home/kelong/software/compiler/dtk/dtk-23.04.1"
# source ${DTK_PATH}/env.sh
# source ${DTK_PATH}/cuda/env.sh
# export C_INCLUDE_PATH=${DTK_PATH}/cuda/include:${C_INCLUDE_PATH}
# export C_INCLUDE_PATH=${DTK_PATH}/include:${C_INCLUDE_PATH}


# module load compiler/devtoolset/9.3.1
# module load mpi/openmpi/4.1.5-gcc9.3.0
# DTK_PATH="/public/home/kelong/software/compiler/dtk/dtk-23.04.1"
# source ${DTK_PATH}/env.sh
# source ${DTK_PATH}/cuda/env.sh
# export C_INCLUDE_PATH=${DTK_PATH}/cuda/include:${C_INCLUDE_PATH}
# export C_INCLUDE_PATH=${DTK_PATH}/include:${C_INCLUDE_PATH}


# module load compiler/intel/2021.3.0 
# module load mpi/intelmpi/2021.3.0
# DTK_PATH="/public/home/kelong/software/compiler/dtk/dtk-23.04.1"
# source ${DTK_PATH}/env.sh
# source ${DTK_PATH}/cuda/env.sh
# export C_INCLUDE_PATH=${DTK_PATH}/include:${DTK_PATH}/cuda/include:${C_INCLUDE_PATH}
# export CPLUS_INCLUDE_PATH=${DTK_PATH}/include:${DTK_PATH}/cuda/include:${CPLUS_INCLUDE_PATH}

module load compiler/automake/1.15

