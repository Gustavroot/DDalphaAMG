#!/bin/bash
# Compares the CPU-side cost of the restriction copies and kernel launches between
# Nsight Systems profiles (rank 0), e.g. pageable vs pinned host memory.
# For every call it joins the CUDA API call (CPU side) with its GPU operation (CorrId)
# and prints, per category, the number of calls and median / mean durations:
#   API  = time spent by the CPU inside cudaMemcpy / cudaLaunchKernel
#   GPU  = duration of the copy / kernel on the GPU
# Categories are recognised by the size of the copy (64x32^3, 2 ranks, 24 test vectors):
#   depth 0: HtoD phi 100663296 B, DtoH phi_c 1572864 B
#   depth 1: HtoD phi   1572864 B, DtoH phi_c    6144 B
#
# usage (login node):  ./compare_restrict_copies_CASTNER.sh nsys_mps_pageable.tar.gz nsys_mps.tar.gz

NSYS_DIR=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/profilers/13.0/Nsight_Systems/target-linux-sbsa-armv8
export PATH=$NSYS_DIR:$PATH
command -v nsys > /dev/null || { echo "ERROR: nsys not found"; exit 1; }
[ $# -ge 1 ] || { echo "usage: $0 archive1.tar.gz [archive2.tar.gz ...]"; exit 1; }

TMP=$(mktemp -d compare_copies_XXXX)
trap 'rm -rf "$TMP"' EXIT

for a in "$@"; do
  [ -f "$a" ] || { echo "ERROR: $a not found"; exit 1; }
  tag=$(basename "$a" .tar.gz)
  mkdir -p "$TMP/$tag"
  rep=$(tar -tzf "$a" | grep 'report_rank0.nsys-rep$' | head -1)
  [ -n "$rep" ] || { echo "ERROR: no report_rank0.nsys-rep in $a"; exit 1; }
  tar -xzf "$a" -C "$TMP/$tag" "$rep"
  echo "exporting $a ..."
  nsys stats --force-export=true --report cuda_api_trace,cuda_gpu_trace --format csv \
       --output "$TMP/$tag/t" "$TMP/$tag/$rep" > "$TMP/$tag/stats.log" 2>&1
  ls "$TMP/$tag"/t_cuda_api_trace.csv "$TMP/$tag"/t_cuda_gpu_trace.csv > /dev/null 2>&1 \
    || { echo "ERROR: nsys stats failed for $a, see log:"; tail -5 "$TMP/$tag/stats.log"; exit 1; }
done

python3 - "$TMP" "$@" << 'PYEOF'
import csv, os, sys, statistics

tmp, archives = sys.argv[1], sys.argv[2:]

def col(hdr, *names):
    low = [h.lower() for h in hdr]
    for n in names:
        if n.lower() in low:
            return low.index(n.lower())
    raise KeyError(f"none of {names} in {hdr}")

def load(path):
    rows = list(csv.reader(open(path)))
    i = next(k for k, r in enumerate(rows) if any(c.lower().startswith("start") for c in r))
    return rows[i], rows[i+1:]

CATS = [("depth 0  HtoD phi",   "memcpy", "HtoD", 100663296),
        ("depth 0  DtoH phi_c", "memcpy", "DtoH", 1572864),
        ("depth 1  HtoD phi",   "memcpy", "HtoD", 1572864),
        ("depth 1  DtoH phi_c", "memcpy", "DtoH", 6144),
        ("restrict launch",     "kernel", None,   None)]

def med(x):  return statistics.median(x) if x else float("nan")
def mean(x): return statistics.fmean(x) if x else float("nan")

for a in archives:
    tag = os.path.basename(a)[:-len(".tar.gz")]
    ah, ar = load(os.path.join(tmp, tag, "t_cuda_api_trace.csv"))
    gh, gr = load(os.path.join(tmp, tag, "t_cuda_gpu_trace.csv"))
    a_dur, a_name, a_corr = col(ah, "Duration (ns)"), col(ah, "Name"), col(ah, "CorrID", "CorrId")
    g_dur, g_name, g_corr = col(gh, "Duration (ns)"), col(gh, "Name"), col(gh, "CorrId", "CorrID")
    try:
        g_bytes = col(gh, "Bytes (MB)"); scale = 1e6
    except KeyError:
        g_bytes = col(gh, "Bytes (B)", "Bytes"); scale = 1.0
    api = {}
    for r in ar:
        if len(r) > max(a_dur, a_name, a_corr):
            api[r[a_corr]] = (r[a_name], float(r[a_dur].replace(",", "")))
    res = {c[0]: ([], []) for c in CATS}
    for r in gr:
        if len(r) <= max(g_dur, g_name, g_corr):
            continue
        name, corr = r[g_name], r[g_corr]
        if corr not in api:
            continue
        a_n, a_d = api[corr]
        g_d = float(r[g_dur].replace(",", ""))
        if "cuda_restrict" in name:
            res["restrict launch"][0].append(a_d); res["restrict launch"][1].append(g_d)
            continue
        if "memcpy" not in name.lower() or not a_n.startswith("cudaMemcpy"):
            continue
        direction = "HtoD" if "host-to-device" in name.lower() else "DtoH" if "device-to-host" in name.lower() else None
        try:
            nbytes = float(r[g_bytes].replace(",", "")) * scale
        except ValueError:
            continue
        for label, kind, d, size in CATS:
            if kind == "memcpy" and d == direction and abs(nbytes - size) < max(0.01*size, 0.0006*scale):
                res[label][0].append(a_d); res[label][1].append(g_d)
    # whole restriction as seen by the CPU: from the start of the HtoD copy call before the
    # launch to the end of the call that waits for the result (DtoH cudaMemcpy or
    # cudaStreamSynchronize), per depth (depth from the size of the HtoD copy)
    a_start = col(ah, "Start (ns)")
    a_tid = col(ah, "Tid", "TID")
    g_size = {}
    for r in gr:
        if len(r) > max(g_corr, g_bytes):
            try: g_size[r[g_corr]] = float(r[g_bytes].replace(",", "")) * scale
            except ValueError: pass
    restrict_corr = set()
    for r in gr:
        if len(r) > max(g_name, g_corr) and "cuda_restrict" in r[g_name]:
            restrict_corr.add(r[g_corr])
    by_tid = {}
    for r in ar:
        if len(r) > max(a_start, a_dur, a_name, a_corr, a_tid):
            by_tid.setdefault(r[a_tid], []).append((float(r[a_start].replace(",", "")),
                                                   float(r[a_dur].replace(",", "")), r[a_name], r[a_corr]))
    span = {"depth 0": [], "depth 1": []}
    for calls in by_tid.values():
        calls.sort()
        for k, (t0, d0, n0, c0) in enumerate(calls):
            if c0 not in restrict_corr: continue
            j = k - 1
            while j >= 0 and not calls[j][2].startswith("cudaMemcpy"): j -= 1
            if j < 0: continue
            size = g_size.get(calls[j][3], -1)
            depth = "depth 0" if abs(size - 100663296) < 0.01*100663296 else \
                    "depth 1" if abs(size - 1572864) < 0.01*1572864 else None
            if depth is None: continue
            m = k + 1
            while m < len(calls) and calls[m][2] not in ("cudaStreamSynchronize", "cudaMemcpy"): m += 1
            if m == len(calls): continue
            span[depth].append((calls[m][0] + calls[m][1] - calls[j][0]) / 1000)
    print(f"\n=== {a}   (rank 0, times in microseconds)")
    for dpt in ("depth 0", "depth 1"):
        x = span[dpt]
        print(f"whole restriction, {dpt}: {len(x):5d} calls, median {med(x):8.1f}, mean {mean(x):8.1f}  (CPU, copy+launch+copy+wait)")
    print(f"{'category':22s} {'calls':>6} {'API median':>11} {'API mean':>10} {'GPU median':>11} {'API-GPU median':>15}")
    for label, *_ in CATS:
        A, G = res[label]
        A = [x/1000 for x in A]; G = [x/1000 for x in G]
        gap = [x - y for x, y in zip(A, G)]
        g = f"{med(gap):15.1f}" if label != "restrict launch" else f"{'-':>15}"
        print(f"{label:22s} {len(A):>6} {med(A):11.1f} {mean(A):10.1f} {med(G):11.1f} {g}")
print("\nAPI-GPU for the DtoH copies includes the wait for the kernel (synchronous copy).")
PYEOF
