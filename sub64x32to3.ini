#!/bin/bash

#SBATCH --account=mul-tra
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --output=mpi_out_%j.txt
#SBATCH --error=mpi_err_%j.txt
#SBATCH --time=00:29:00
#SBATCH --gres=gpu:4 --partition=develbooster



# IMPORTANT : this latice i.e. 8to4 is used for development purposes



#module load GCC ParaStationMPI MPI-settings/CUDA UCX-settings/RC-CUDA

module load nano CUDA GCC OpenMPI MPI-settings/CUDA

jutil env activate -p chwu29

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun --distribution=block:cyclic:fcyclic --cpus-per-task=${SLURM_CPUS_PER_TASK} \
    dd_alpha_amg sample64x32to3.ini
