#!/bin/bash
#SBATCH -J rank_analysis_gpr
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 72
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -t 08:00:00
#SBATCH -o rank_analysis/rank_analysis_gpu_per_rank_%j.out
#
# Rank-scaling comparison on CASTNER (64x32^3, 1 OpenMP thread per rank):
#   CPU vanilla, GPU vanilla, GPU + restriction, with 2, 4, 8, 16 MPI ranks.
#   In the GPU runs EVERY rank has ITS OWN GH200 (no MPS). A node has 4 GH200, so
#   2 and 4 ranks run on 1 node, 8 ranks on 2 nodes, 16 ranks on 4 nodes (4 ranks per node).
#   Each rank gets a whole Grace CPU (72 cores, 1 used) and the GPU attached to it.
#
# usage:  cd /lus/bnchlu5/e75014/DDalphaAMG
#         sbatch rank_analysis_gpu_per_rank_CASTNER.sh                      # 2 4 8 16 ranks (4 nodes)
#         sbatch -N 1 rank_analysis_gpu_per_rank_CASTNER.sh 2 4             # only 2 and 4 ranks (1 node)
#
# Results go to restriction_rank_analysis_1gpu_per_rank/ (see README.txt there).
# Submit from a shell with the usual modules loaded (PrgEnv-gnu, cray-mpich, cuda):
# the job inherits that environment.

# --- settings ------------------------------------------------------------------
WORKDIR=/lus/bnchlu5/e75014/DDalphaAMG
VANILLA=/home/users/e75014/vanilla_DDalphaAMG
BASE_INI=3lvl_sample64_CASTNER_m4.ini   # <-- input file (method 4: FGMRES + GMRES smoother)
OUTROOT=${OUTROOT:-restriction_rank_analysis_1gpu_per_rank}
RANKS=${*:-"2 4 8 16"}
STEP_TIME=90                      # time limit of each single run, in minutes

BIN_CPU_SRC=$VANILLA/dd_alpha_amg_cpu_vanilla
BIN_GPU_SRC=$VANILLA/dd_alpha_amg_gpu_vanilla
BIN_RES_SRC=$WORKDIR/dd_alpha_amg

# local lattices (depth 0 | depth 1 | depth 2) for each number of ranks
decomp() {
  case $1 in
    2)  echo "32 32 32 32|8 8 8 8|2 2 2 2" ;;
    4)  echo "16 32 32 32|4 8 8 8|1 2 2 2" ;;
    8)  echo "16 16 32 32|4 4 8 8|1 1 2 2" ;;
    16) echo "16 16 16 32|4 4 4 8|1 1 1 2" ;;
    *)  return 1 ;;
  esac
}

die() { echo "ERROR: $*"; exit 1; }

# 4 GH200 per node: ranks per node and number of nodes for N ranks
GPUS_PER_NODE=4
tpn()    { [ "$1" -lt "$GPUS_PER_NODE" ] && echo "$1" || echo "$GPUS_PER_NODE"; }
nnodes() { echo $(( ($1 + GPUS_PER_NODE - 1) / GPUS_PER_NODE )); }

# options inherited from the submitting shell (e.g. sbatch launched inside a salloc with
# --gpus-per-task) would be applied to every srun: drop them
unset SLURM_GPUS_PER_TASK SLURM_TRES_PER_TASK

# --- checks ----------------------------------------------------------------------
cd "$WORKDIR" || die "cannot cd to $WORKDIR"
[ -e "$OUTROOT" ] && die "$WORKDIR/$OUTROOT already exists: rename or remove it first"
[ -f "$BASE_INI" ] || die "$BASE_INI not found"
T=$(grep -E '^number of openmp threads:' "$BASE_INI" | cut -d: -f2 | tr -d ' ')
[ "$T" = "1" ] || die "$BASE_INI must have 'number of openmp threads: 1' (found: $T)"
METHOD=$(grep -E '^method:' "$BASE_INI" | cut -d: -f2 | tr -d ' ')
case $METHOD in
  1) METHOD_TXT="FGMRES + additive Schwarz smoother" ;;
  2) METHOD_TXT="FGMRES + red-black Schwarz smoother" ;;
  3) METHOD_TXT="FGMRES + 16 color Schwarz smoother" ;;
  4) METHOD_TXT="FGMRES + GMRES smoother" ;;
  *) METHOD_TXT="see input file" ;;
esac

for b in "$BIN_CPU_SRC" "$BIN_GPU_SRC" "$BIN_RES_SRC"; do
  [ -x "$b" ] || die "binary $b not found"
done
ldd "$BIN_CPU_SRC" | grep -q libcudart && die "$BIN_CPU_SRC is a GPU build (expected CPU)"
ldd "$BIN_GPU_SRC" | grep -q libcudart || die "$BIN_GPU_SRC is not a GPU build"
ldd "$BIN_RES_SRC" | grep -q libcudart || die "$BIN_RES_SRC is not a GPU build (set CUDA_ENABLER = yes and rebuild)"
strings "$BIN_GPU_SRC" | grep -q cuda_restrict && die "$BIN_GPU_SRC contains the ported restriction (expected vanilla)"
strings "$BIN_RES_SRC" | grep -q cuda_restrict || die "$BIN_RES_SRC does not contain the ported restriction"
[ "$(strings "$BIN_RES_SRC" | grep -c 'restrict test')" = "0" ] || die "$BIN_RES_SRC contains the unit test (built with -DCUDA_RESTRICT_TEST)"


for N in $RANKS; do
  decomp "$N" > /dev/null || die "no lattice decomposition defined for $N ranks (allowed: 2 4 8 16)"
  [ "$(nnodes "$N")" -le "$SLURM_NNODES" ] || die "$N ranks need $(nnodes "$N") nodes, job has $SLURM_NNODES (sbatch -N)"
done

# --- output folder, frozen copies of binaries --------------------------------------
mkdir -p "$OUTROOT/bin"
cp "$BIN_CPU_SRC" "$OUTROOT/bin/dd_alpha_amg_cpu_vanilla"
cp "$BIN_GPU_SRC" "$OUTROOT/bin/dd_alpha_amg_gpu_vanilla"
cp "$BIN_RES_SRC" "$OUTROOT/bin/dd_alpha_amg_gpu_restrict"
cp "$0" "$OUTROOT/bin/rank_analysis_gpu_per_rank_CASTNER.sh" 2> /dev/null
BIN_CPU=$PWD/$OUTROOT/bin/dd_alpha_amg_cpu_vanilla
BIN_GPU=$PWD/$OUTROOT/bin/dd_alpha_amg_gpu_vanilla
BIN_RES=$PWD/$OUTROOT/bin/dd_alpha_amg_gpu_restrict

RUNLOG=$OUTROOT/runs.log
NODE=$SLURM_JOB_NODELIST

# --- README (description; results are appended at the end of the job) ----------------
COMMIT_RES=$(git -C "$WORKDIR" rev-parse --short HEAD 2>/dev/null)
BRANCH_RES=$(git -C "$WORKDIR" rev-parse --abbrev-ref HEAD 2>/dev/null)
COMMIT_VAN=$(git -C "$VANILLA" rev-parse --short HEAD 2>/dev/null)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null)

cat > "$OUTROOT/README.txt" << EOF
RESTRICTION GPU PORT - RANK SCALING, ONE GPU PER RANK
=====================================================

What this is
------------
Comparison of three builds of DDalphaAMG on the same lattice and on the same set of nodes,
with 2, 4, 8 and 16 MPI ranks, 1 OpenMP thread per rank:

  cpu_vanilla   vanilla code, CPU only                     (no GPU)
  gpu_vanilla   vanilla code, CUDA build + the two fixes needed to run it here
                (method 4 on more than 2 levels in vcycle_generic.c; unused CUDA Dynamic
                 Parallelism code disabled, link with -cudadevrt none)
  gpu_restrict  CUDA build + restriction ported to GPU     (branch ${BRANCH_RES}, commit ${COMMIT_RES})

In the GPU runs EVERY rank has its OWN GPU (no MPS, no GPU sharing). Each CASTNER node has
4 GH200 superchips (Grace CPU + Hopper GPU), so 4 ranks per node are used:

  ranks   nodes   ranks per node   GPUs (GPU builds)
  2       1       2                2
  4       1       4                4
  8       2       4                8
  16      4       4                16

Every rank gets one whole Grace CPU (srun -c 72) and the GPU of the same superchip
(--gpu-bind=closest); with 1 OpenMP thread it uses 1 core of it. The CPU runs use exactly the
same placement (same nodes, same ranks per node, 1 core per rank), so for a given rank count the
three builds use the same CPU resources; the GPU builds have one GPU per rank in addition.

Companion analysis: restriction_rank_analysis/ (same runs, but all ranks sharing ONE GPU via MPS).

Job
---
date:            $(date)
Slurm job id:    ${SLURM_JOB_ID}
nodes:           ${NODE} (allocated exclusively)
GPU:             ${GPU_NAME}
working folder:  ${WORKDIR}
vanilla folder:  ${VANILLA} (commit ${COMMIT_VAN})

Test case
---------
configuration:   /home/users/e75014/configs/qcdsf.632.01000.mg2, lattice 64x32^3 (T Z Y X)
input:           ${BASE_INI}, only the local lattices changed for each rank count
                 (3 levels, method ${METHOD} = ${METHOD_TXT}, 24 test vectors,
                  4 setup iterations, mixed precision, odd-even, K-cycle, tolerance 1e-10,
                  mu coarsest 0.0001)
OpenMP threads:  1 per rank (number of openmp threads: 1 in the input)

Lattice decomposition (local lattice per rank, T Z Y X)
--------------------------------------------------------
  ranks   depth 0         depth 1       depth 2
  2       32 32 32 32     8 8 8 8       2 2 2 2
  4       16 32 32 32     4 8 8 8       1 2 2 2
  8       16 16 32 32     4 4 8 8       1 1 2 2
  16      16 16 16 32     4 4 4 8       1 1 1 2
Global lattices: depth 0 64 32 32 32, depth 1 16 8 8 8, depth 2 4 2 2 2 (blocks 4^4).

How the runs are launched
-------------------------
  K = ranks per node, M = nodes (table above)
  CPU:  MPICH_GPU_SUPPORT_ENABLED=0 srun -N M -n N --ntasks-per-node=K -c 72 --cpu-bind=cores \\
            bin/dd_alpha_amg_cpu_vanilla input.ini
  GPU:  MPICH_GPU_SUPPORT_ENABLED=1 srun -N M -n N --ntasks-per-node=K -c 72 --cpu-bind=cores \\
            --gpus-per-task=1 --gpu-bind=closest bin/dd_alpha_amg_gpu_<vanilla|restrict> input.ini
Runs are executed one after the other, in the order: for each rank count
cpu_vanilla, gpu_vanilla, gpu_restrict.

Folder structure
----------------
  README.txt            this file (results table at the end)
  summary.txt           results table (also regenerated with: python3 make_summary.py)
  make_summary.py       script that reads the result files and writes summary.txt
  runs.log              one line per run: rank count, build, exit code, wall time
  bin/                  exact binaries and job script used for these runs
  <N>_ranks/
     input_<N>ranks.ini      input file used
     result_cpu_vanilla.txt  full program output (stdout+stderr) of each run
     result_gpu_vanilla.txt
     result_gpu_restrict.txt

Where the numbers come from (in each result_*.txt)
--------------------------------------------------
  config reading (excluded) : FIRST line "| elapsed wall clock time: ... seconds |"
  initial definition        : first  line "elapsed time: ... seconds"
  iterative setup           : second line "elapsed time: ... seconds (... seconds on coarse grid)"
  solve                     : LAST line "| elapsed wall clock time: ... seconds |"
  total                     : initial definition + iterative setup + solve
  outer iterations          : number of lines "approx. rel. res. after N iterations"
  final residual            : line "exact relative residual: ||r||/||b|| = ..."
  quick look:  grep -E "elapsed (wall clock )?time" result_gpu_restrict.txt

Notes
-----
- All runs: same allocation (same nodes), one after the other. 2 and 4 ranks use the first
  node only, 8 ranks the first 2 nodes, 16 ranks all 4.
- 8 and 16 ranks communicate between nodes (Slingshot network); 2 and 4 ranks only within
  one node. This is part of what is measured when going from 4 to 8 ranks.
- With 1 thread per rank the CPU part of every build runs on N cores in total.
- For 4 or more ranks some local lattices at depth 1 contain a single 4^4 block in one
  direction, and the coarsest level has very few sites per rank (2 sites with 16 ranks).
  If a configuration is rejected by DDalphaAMG, the error is in the corresponding
  result_*.txt and the run is marked FAILED in summary.txt.
- Previous report (method 2 = red-black Schwarz, 2 ranks on 2 GPUs, 1 thread): CPU vanilla
  1171.6 s, GPU vanilla 638.9 s, GPU + restriction 613.9 s, 18 iterations. Those numbers are
  NOT directly comparable with these runs if a different method is used here.

EOF

{
  echo "Modules loaded at submission:"
  module list 2>&1 | sed 's/^/  /'
  echo
} >> "$OUTROOT/README.txt" 2> /dev/null

# --- summary script ----------------------------------------------------------------------
cat > "$OUTROOT/make_summary.py" << 'PYEOF'
#!/usr/bin/env python3
# Reads <N>_ranks/result_*.txt and writes summary.txt (lives in the analysis folder)
import os, re, glob

BUILDS = ["cpu_vanilla", "gpu_vanilla", "gpu_restrict"]
here = os.path.dirname(os.path.abspath(__file__))

def phases(fname):
    txt = open(fname, errors="replace").read()
    el  = [float(x) for x in re.findall(r"^elapsed time: ([\d.]+) seconds", txt, re.M)]
    cg  = re.findall(r"^elapsed time: [\d.]+ seconds \(([\d.]+) seconds on coarse grid\)", txt, re.M)
    wc  = [float(x) for x in re.findall(r"elapsed wall clock time: ([\d.]+)", txt)]
    it  = len(re.findall(r"approx\. rel\. res", txt))
    res = re.findall(r"exact relative residual: \|\|r\|\|/\|\|b\|\| = ([\d.e+-]+)", txt)
    ci  = re.findall(r"avg coarsest iters\s*=\s*([\d.]+)", txt)
    cerr = len(re.findall(r"CUDA error|cudaError", txt))
    if len(el) < 2 or len(wc) < 2 or not res:
        return None
    return dict(init=el[0], setup=el[1], coarse=float(cg[0]) if cg else float("nan"),
                solve=wc[-1], total=el[0] + el[1] + wc[-1], iter=it, res=res[-1],
                ci=float(ci[-1]) if ci else float("nan"), cerr=cerr)

ranks = sorted(int(os.path.basename(d).split("_")[0])
               for d in glob.glob(os.path.join(here, "*_ranks")))

lines = []
hdr = f"{'ranks':>5}  {'build':12s} {'init':>8} {'setup':>9} {'(coarse)':>9} {'solve':>8} {'TOTAL':>9}  {'iter':>4}  {'residual':>12}  {'speedup vs CPU':>14}  {'coarsest it':>11}  {'CUDA err':>8}"
lines += ["Times in seconds (total = initial definition + iterative setup + solve).",
          "Speedup vs CPU: CPU vanilla total / build total, same number of ranks.",
          "coarsest it: avg coarsest iters printed by DDalphaAMG (must be similar for the 3 builds).",
          "CUDA err: number of CUDA error lines in the output (must be 0).", "", hdr, "-" * len(hdr)]
data = {}
for n in ranks:
    for b in BUILDS:
        f = os.path.join(here, f"{n}_ranks", f"result_{b}.txt")
        p = phases(f) if os.path.isfile(f) else None
        data[(n, b)] = p
        if p is None:
            lines.append(f"{n:>5}  {b:12s} {'FAILED or missing':>45}")
            continue
        cpu = data.get((n, "cpu_vanilla"))
        sp = f"x{cpu['total']/p['total']:.2f}" if cpu else "-"
        lines.append(f"{n:>5}  {b:12s} {p['init']:8.2f} {p['setup']:9.2f} {p['coarse']:9.2f} "
                     f"{p['solve']:8.2f} {p['total']:9.2f}  {p['iter']:>4}  {p['res']:>12}  {sp:>14}  {p['ci']:11.2f}  {p['cerr']:>8}")
    lines.append("")

lines += ["Restriction port vs GPU vanilla (same number of ranks):"]
for n in ranks:
    a, b = data.get((n, "gpu_vanilla")), data.get((n, "gpu_restrict"))
    if a and b:
        d = b["total"] - a["total"]
        lines.append(f"  {n:>2} ranks: {d:+8.2f} s ({100*d/a['total']:+6.2f}%)")
    else:
        lines.append(f"  {n:>2} ranks: n/a")

lines += ["", "Scaling with the number of ranks (total time, relative to 2 ranks):"]
for b in BUILDS:
    ref = data.get((2, b))
    row = []
    for n in ranks:
        p = data.get((n, b))
        row.append(f"{n}r: " + (f"{p['total']:.1f}s" + (f" (x{ref['total']/p['total']:.2f})" if ref else "") if p else "n/a"))
    lines.append(f"  {b:12s} " + "   ".join(row))

out = "\n".join(lines) + "\n"
open(os.path.join(here, "summary.txt"), "w").write(out)
print(out)
PYEOF
chmod +x "$OUTROOT/make_summary.py"

# --- helpers ------------------------------------------------------------------------------
run_one() {   # run_one <ranks> <build> <folder> <input>
  local N=$1 B=$2 D=$3 INI=$4 OUT=$3/result_$2.txt rc t0 t1
  local K M; K=$(tpn "$N"); M=$(nnodes "$N")
  t0=$(date +%s)
  echo "=== $N ranks ($M node(s), $K per node), $B: started $(date +%T)"
  case $B in
    cpu_vanilla)
      MPICH_GPU_SUPPORT_ENABLED=0 OMP_NUM_THREADS=1 \
        srun -N "$M" -n "$N" --ntasks-per-node="$K" -c 72 --cpu-bind=cores -t "$STEP_TIME" \
             "$BIN_CPU" "$INI" > "$OUT" 2>&1
      rc=$? ;;
    gpu_vanilla|gpu_restrict)
      local EXE=$BIN_GPU; [ "$B" = gpu_restrict ] && EXE=$BIN_RES
      MPICH_GPU_SUPPORT_ENABLED=1 OMP_NUM_THREADS=1 \
        srun -N "$M" -n "$N" --ntasks-per-node="$K" -c 72 --cpu-bind=cores \
             --gpus-per-task=1 --gpu-bind=closest -t "$STEP_TIME" \
             "$EXE" "$INI" > "$OUT" 2>&1
      rc=$? ;;
  esac
  t1=$(date +%s)
  printf "%3d ranks  %-12s  exit %3d  wall %6d s  %d node(s) x %d\n" "$N" "$B" "$rc" "$((t1 - t0))" "$M" "$K" >> "$RUNLOG"
  echo "    finished, exit code $rc, $((t1 - t0)) s"
}

# --- runs ----------------------------------------------------------------------------------
echo "nodes: $NODE   GPU: $GPU_NAME   rank counts: $RANKS"
printf "%-9s  %-12s  %-8s  %-11s  %s\n" "ranks" "build" "exit" "wall" "placement" > "$RUNLOG"

for N in $RANKS; do
  D=$OUTROOT/${N}_ranks
  mkdir -p "$D"
  IFS='|' read -r L0 L1 L2 <<< "$(decomp "$N")"
  INI=$D/input_${N}ranks.ini
  sed -e "s/^d0 local lattice:.*/d0 local lattice: $L0/" \
      -e "s/^d1 local lattice:.*/d1 local lattice: $L1/" \
      -e "s/^d2 local lattice:.*/d2 local lattice: $L2/" \
      "$BASE_INI" > "$INI"

  # same computation of the number of processes as run_CASTNER
  read -a G <<< "$(grep -E '^d0 global lattice:' "$INI" | cut -d: -f2)"
  read -a L <<< "$(grep -E '^d0 local lattice:'  "$INI" | cut -d: -f2)"
  NP=1; for i in 0 1 2 3; do NP=$(( NP * (G[i] / L[i]) )); done
  if [ "$NP" -ne "$N" ]; then
    echo "ERROR: $INI gives $NP ranks instead of $N, skipping"; continue
  fi

  for B in cpu_vanilla gpu_vanilla gpu_restrict; do
    run_one "$N" "$B" "$D" "$PWD/$INI"
  done
done

# --- summary ---------------------------------------------------------------------------------
python3 "$OUTROOT/make_summary.py" > /dev/null
{
  echo "Results (copy of summary.txt, written at $(date))"
  echo "--------------------------------------------------------------"
  cat "$OUTROOT/summary.txt"
} >> "$OUTROOT/README.txt"

echo
cat "$OUTROOT/summary.txt"
echo "all done: $WORKDIR/$OUTROOT"
