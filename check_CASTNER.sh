#!/bin/bash
#SBATCH -J check
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH -c 72
#SBATCH --gpus-per-node=2
#SBATCH --exclusive
#SBATCH -t 01:30:00
#SBATCH -o check_%j.out
#
# Quick check of a build: 64x32^3, method 4, 2 MPI ranks with one GPU each, 1 OpenMP thread
# per rank (same placement as Series A, rank_analysis/seriesA_1gpu_per_rank_m4/2_ranks).
# Prints iterations, residual, unit-test lines and the times
# (total = initial definition + iterative setup + solve, as in rank_analysis/*/summary.txt).
#
# usage:  sbatch check_CASTNER.sh                  (binary ./dd_alpha_amg)
#         EXE=./other_binary sbatch check_CASTNER.sh

INPUT=${INPUT:-3lvl_sample64_CASTNER_m4.ini}
EXE=${EXE:-./dd_alpha_amg}

cd "$SLURM_SUBMIT_DIR" || exit 1
unset SLURM_GPUS_PER_TASK SLURM_TRES_PER_TASK
[ -x "$EXE" ]   || { echo "ERROR: $EXE not found"; exit 1; }
[ -f "$INPUT" ] || { echo "ERROR: $INPUT not found"; exit 1; }

OUT=check_${SLURM_JOB_ID}.txt
echo "binary: $EXE   input: $INPUT   node: $(hostname)   $(date)"
echo "git:    $(git rev-parse --short HEAD 2>/dev/null) ($(git rev-parse --abbrev-ref HEAD 2>/dev/null))"

MPICH_GPU_SUPPORT_ENABLED=1 OMP_NUM_THREADS=1 \
  srun -N 1 -n 2 -c 72 --cpu-bind=cores --gpus-per-task=1 --gpu-bind=closest \
       "$EXE" "$INPUT" > "$OUT" 2>&1
echo "exit code: $?   full output: $OUT"

echo "--- unit tests (only in test builds)"
grep -A2 -E "^\[rank 0\] (interpolate|restrict) test" "$OUT" || echo "(none)"
echo "--- CUDA errors: $(grep -ci 'cuda error' "$OUT")"

python3 - "$OUT" << 'PYEOF'
import re, sys
txt = open(sys.argv[1], errors="replace").read()
el  = [float(x) for x in re.findall(r"^elapsed time: ([\d.]+) seconds", txt, re.M)]
wc  = [float(x) for x in re.findall(r"elapsed wall clock time: ([\d.]+)", txt)]
it  = len(re.findall(r"approx\. rel\. res", txt))
res = re.findall(r"exact relative residual: \|\|r\|\|/\|\|b\|\| = ([\d.e+-]+)", txt)
print(f"--- iterations: {it}   residual: {res[-1] if res else 'n/a'}")
if len(el) >= 2 and wc:
    print(f"--- times (s): init {el[0]:.2f}   setup {el[1]:.2f}   solve {wc[-1]:.2f}   TOTAL {el[0]+el[1]+wc[-1]:.2f}")
    print( "    reference, Series A 2 ranks: GPU vanilla 673.71, GPU + restriction 647.22")
else:
    print("--- times: not found (run failed?)")
PYEOF
