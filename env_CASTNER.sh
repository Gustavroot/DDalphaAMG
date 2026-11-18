# environment for GPU runs on CASTNER: load with   source env_CASTNER.sh
export OMP_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=1
export MPIRUN_OPTS="-c $OMP_NUM_THREADS --cpu-bind=cores --gpus-per-task=1"
export PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/profilers/13.0/Nsight_Systems/target-linux-sbsa-armv8:$PATH
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS  MPICH_GPU_SUPPORT_ENABLED=$MPICH_GPU_SUPPORT_ENABLED  MPIRUN_OPTS=$MPIRUN_OPTS"
