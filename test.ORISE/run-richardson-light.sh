#!/bin/bash
#SBATCH -J T96L48
#SBATCH -p normal
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --ntasks-per-node=4
#SBATCH --gres=dcu:4
#SBATCH --exclusive
#SBATCH --comment=WRF
#SBATCH -o light.T96L48.%j.out
#SBATCH -e light.T96L48.%j.out

export OMP_NUM_THEADS=1

#LIME_DIR="/public/home/kelong/software/usqcd/c-lime"
#export LD_LIBRARY_PATH=${LIME_DIR}/lib:${LD_LIBRARY_PATH}
#SOFT_DIR="/public/home/kelong/software/DDalphaAMG/04_DDalphaAMG_GPU"
SOFT_DIR="/public/home/kelong/software/DDalphaAMG/02_DDalphaAMG_AVX_GPU"
export LD_LIBRARY_PATH=${SOFT_DIR}/lib:${LD_LIBRARY_PATH}

source ${SOFT_DIR}/setup_env.ORISE.sh

echo ${LD_LIBRARY_PATH}
# export OMPI_MCA_opal_cuda_support=0
# export UCX_MEMTYPE_CACHE=n

# 6 3 3 3
FILE_I="./Richardson-light.ini"

AMGEXE=${SOFT_DIR}/dd_alpha_amg

echo " "
echo "======================================================="
echo " "
echo "Begin"
date 
echo "SLURM_NODELIST=${SLURM_NODELIST}"
mpirun -n 16  ${AMGEXE}  ${FILE_I}  
date
echo "End"


