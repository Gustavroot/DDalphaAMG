#!/bin/bash
#SBATCH -J ddamg
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 72
#SBATCH --gpus-per-task=1
#SBATCH -t 02:00:00
#SBATCH -o ddamg_%j.out
#
# Runs ./dd_alpha_amg through run_CASTNER on CASTNER.
# usage (from the repository folder):   sbatch sbatch_CASTNER.sh
#
# - CPU or GPU: set CUDA_ENABLER in MakeSettings_CASTNER.mk and rebuild;
#   the script detects the build from the binary itself.
# - OpenMP threads: taken from "number of openmp threads" in the input file (max 72).
# - MPI ranks: #SBATCH -n must match the lattice decomposition of the input file
#   (max 4 GPUs per node: for more than 4 ranks increase -N as well).

INPUT=3lvl_sample64_CASTNER.ini          # <-- input file

cd "$SLURM_SUBMIT_DIR" || exit 1

# --- checks --------------------------------------------------------------------
[ -f "$INPUT" ]       || { echo "ERROR: input file $INPUT not found"; exit 1; }
[ -x dd_alpha_amg ]   || { echo "ERROR: ./dd_alpha_amg not found, build it first"; exit 1; }
[ -x run_CASTNER ]    || { echo "ERROR: ./run_CASTNER not found"; exit 1; }

# CPU or GPU build: decided by the binary
if ldd ./dd_alpha_amg | grep -qE "libcudart|libmpi_gtl_cuda"; then BUILD=gpu; else BUILD=cpu; fi
SETTING=$(grep -E '^CUDA_ENABLER' MakeSettings_CASTNER.mk | cut -d= -f2 | tr -d ' ')
if { [ "$BUILD" = gpu ] && [ "$SETTING" != yes ]; } || { [ "$BUILD" = cpu ] && [ "$SETTING" != no ]; }; then
  echo "WARNING: dd_alpha_amg is a $BUILD build, but MakeSettings_CASTNER.mk has CUDA_ENABLER=$SETTING (not rebuilt?)"
fi

# OpenMP threads from the input file
T=$(grep -E '^number of openmp threads:' "$INPUT" | cut -d: -f2 | tr -d ' ')
T=${T:-1}
if [ "$T" -gt "$SLURM_CPUS_PER_TASK" ]; then
  echo "ERROR: input asks for $T threads, job has $SLURM_CPUS_PER_TASK cores per task (#SBATCH -c)"; exit 1
fi

# MPI ranks from the input file (same computation as run_CASTNER)
read -a G <<< "$(grep -E '^d0 global lattice:' "$INPUT" | cut -d: -f2)"
read -a L <<< "$(grep -E '^d0 local lattice:'  "$INPUT" | cut -d: -f2)"
NP=1
for i in 0 1 2 3; do NP=$(( NP * (G[i] / L[i]) )); done
if [ "$NP" -ne "$SLURM_NTASKS" ]; then
  echo "ERROR: input needs $NP MPI ranks, job has $SLURM_NTASKS (change #SBATCH -n, and -N if more than 4)"; exit 1
fi

# --- environment -----------------------------------------------------------------
export OMP_NUM_THREADS=$T
if [ "$BUILD" = gpu ]; then
  export MPICH_GPU_SUPPORT_ENABLED=1
  export MPIRUN_OPTS="-c $T --cpu-bind=cores --gpus-per-task=1"
else
  export MPICH_GPU_SUPPORT_ENABLED=0
  export MPIRUN_OPTS="-c $T --cpu-bind=cores"
fi

OUT=result_$(basename "$INPUT" .ini)_${BUILD}_omp${T}_${SLURM_JOB_ID}.txt

echo "date:        $(date)"
echo "node(s):     $SLURM_JOB_NODELIST"
echo "folder:      $PWD"
echo "git commit:  $(git rev-parse --short HEAD 2>/dev/null) ($(git rev-parse --abbrev-ref HEAD 2>/dev/null))"
echo "build:       $BUILD"
echo "input:       $INPUT"
echo "MPI ranks:   $NP"
echo "OMP threads: $T per rank"
echo "output:      $OUT"
echo

# --- run -------------------------------------------------------------------------
./run_CASTNER -i "$INPUT" > "$OUT" 2>&1
echo "exit code: $?"

# --- short summary -----------------------------------------------------------------
echo "iterations: $(grep -c 'approx. rel. res' "$OUT")"
grep "exact relative residual" "$OUT"
grep -E "elapsed (wall clock )?time" "$OUT"
