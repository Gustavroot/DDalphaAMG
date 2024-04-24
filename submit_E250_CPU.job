#!/bin/bash

#SBATCH --account=mul-tra
#SBATCH --nodes=72
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --output=mpi_out_%j.txt
#SBATCH --error=mpi_err_%j.txt
#SBATCH --time=00:59:00
#SBATCH --partition=batch



# when using GPUS
#module load nano CUDA GCC OpenMPI MPI-settings/CUDA

# CPU-only
module load nano GCC OpenMPI

#jutil env activate -p chwu29

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun --distribution=block:cyclic:fcyclic --cpus-per-task=${SLURM_CPUS_PER_TASK} \
    dd_alpha_amg sample_E250_CPU.ini
