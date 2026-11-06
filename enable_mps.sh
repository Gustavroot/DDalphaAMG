#!/bin/bash

GPUSID=${GPUSID:-"0 1 2 3"}
GPUSID=(${GPUSID})
NGPUS=${#GPUSID[@]}

# Tasks per node: use SLURM_NTASKS_PER_NODE if set, otherwise derive it
NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-$(( SLURM_NTASKS / SLURM_NNODES ))}
LOCALID=${SLURM_LOCALID}

mps_prefix="/tmp/$(id -un)/slurm-${SLURM_JOBID}.${SLURM_STEPID}/nvidia"
export CUDA_MPS_PIPE_DIRECTORY=${mps_prefix}-mps
export CUDA_MPS_LOG_DIRECTORY=${mps_prefix}-log

use_mps=0
if [ "${NTASKS_PER_NODE}" -gt "${NGPUS}" ]; then
  use_mps=1
fi

# Local rank 0 starts the MPS control daemon (only if ranks > GPUs)
if [ "${LOCALID}" -eq 0 ] && [ "${use_mps}" -eq 1 ]; then
  rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
  mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
  nvidia-cuda-mps-control -d
fi

# All ranks wait until the daemon is up
if [ "${use_mps}" -eq 1 ]; then
  mps_pid_file_timeout=120
  pid_file="${CUDA_MPS_PIPE_DIRECTORY}/nvidia-cuda-mps-control.pid"
  if ! timeout ${mps_pid_file_timeout} bash -c "until [[ -f \"${pid_file}\" ]]; do sleep 1; done"; then
    echo "MPS wrapper: timed out waiting for ${pid_file} on rank ${SLURM_PROCID}"
    exit 1
  fi
fi

# Map consecutive local ranks to the same GPU (block mapping, works for any count)
gpu_idx=$(( LOCALID * NGPUS / NTASKS_PER_NODE ))
export CUDA_VISIBLE_DEVICES=${GPUSID[$gpu_idx]}

echo "MPS wrapper: rank ${SLURM_PROCID} (local ${LOCALID}/${NTASKS_PER_NODE}) -> GPU ${CUDA_VISIBLE_DEVICES}, mps=${use_mps}"

exec "$@"
