#!/bin/bash
#SBATCH -J T96L48
#SBATCH -p normal
#SBATCH -N 8
#SBATCH -n 32
#SBATCH --ntasks-per-node=4
#SBATCH --gres=dcu:4
#SBATCH --exclusive
#SBATCH --comment=WRF
#SBATCH -o light.T96L48.%j.out
#SBATCH -e light.T96L48.%j.out

export OMP_NUM_THEADS=1

#SOFT_DIR="/public/home/kelong/software/DDalphaAMG/02_DDalphaAMG_AVX_GPU"
SOFT_DIR="../"
export LD_LIBRARY_PATH=${SOFT_DIR}/lib:${LD_LIBRARY_PATH}

source ${SOFT_DIR}/setup_env.ORISE.sh

echo ${LD_LIBRARY_PATH}

FILE_I="./T96L48-light.ini"

AMGEXE=${SOFT_DIR}/dd_alpha_amg

echo " "
echo "======================================================="
echo " "
echo "Begin"
date 
echo "SLURM_NODELIST=${SLURM_NODELIST}"
mpirun -n 32  ${AMGEXE}  ${FILE_I}  
date
echo "End"


