"""Benchmark harness for Assignment 2 (extends the Assignment 1 harness).

Studies (choose with --study; scripts/run_all.py runs all of them in order):
  blocks  Part A. Tile-side sweep of the two blocked methods at n = BLOCK_N.
  dense   Part A. Every dense method at every size of DENSE_SIZES, random
          dense inputs generated exactly as in Assignment 1.
  sparse  Part B. Matched comparison at n = SPARSE_N: for each distribution
          and density the same logical A and B are multiplied by the CSR method
          (CSR input), by the CSR method with in-program conversion from dense
          input, by the best dense method and (SPARSE_IJK densities only) by
          the baseline.
  scale   Part B. CSR method alone at sizes whose dense storage would exceed
          the memory budget.

Protocol (as Assignment 1): one fresh process per configuration, WARMUP
warm-up samples followed by REPS measured samples; short dense kernels are
batched to ~2^25 multiply-adds per sample and reported per multiplication.
The memory metric is the process's peak working set, read by each program
with GetProcessMemoryInfo before it exits.

Budgets (declared before running):
  * time: TIME_BUDGET_S per single multiplication. When the cubic (dense) or
    measured-ratio (scale study) extrapolation from the previous size predicts
    more than that, the configuration is run once as a probe under a hard
    timeout of TIME_BUDGET_S; on timeout larger sizes are skipped.
  * memory: MEM_BUDGET_MB per process, enforced with a Windows Job Object
    process-memory limit, so an allocation beyond it fails and is recorded as
    "memory_failure".

Outputs: results/raw_<study>/samples.csv and runs.csv.
"""
import argparse
import csv
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DEFAULT_BLOCK, DENSE_METHODS, ROOT, dense_cmd, input_files,  # noqa: E402
                    sparse_cmd, sparse_files)

WARMUP = 2
REPS = 5
TIME_BUDGET_S = 150.0
MEM_BUDGET_MB = 4096
MIN_BATCH_OPS = 2 ** 25

DENSE_SIZES = [64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072]
BLOCK_SIZES = [32, 64, 128, 256, 512]
BLOCK_N = [2048]
SPARSE_N = [2048]
DENSITIES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0]
DISTRIBUTIONS = ["uniform", "banded"]
BEST_DENSE = "blocked_unroll"
# The baseline's runtime does not depend on the values; it is run on two
# densities only to check that, which saves about 2 h of runtime.
SPARSE_IJK = {2048: [0.01, 0.5]}
SCALE = {  # series name -> list of (n, density)
    "uniform_d0.001": [(n, 0.001) for n in (4096, 8192, 16384, 32768, 65536, 131072)],
    "uniform_16_per_row": [(n, 16 / n) for n in (16384, 65536, 262144, 1048576, 2097152)],
}

SAMPLE_COLS = ["study", "method", "n", "distribution", "density_target", "block", "phase", "rep",
               "batch", "elapsed_ns", "time_per_mult_s", "convert_in_s", "convert_out_s",
               "checksum", "nnz_c", "numeric_type", "threads"]
RUN_COLS = ["study", "method", "n", "distribution", "density_target", "block", "status", "warmup",
            "reps", "batch", "nnz_a", "nnz_b", "nnz_c", "csr_bytes", "peak_rss_kb",
            "process_wall_s", "note"]


def batch_for(n):
    return max(1, round(MIN_BATCH_OPS / n ** 3))


def windows_memory_job(proc):
    """Put the child in a Job Object whose per-process committed-memory
    limit is MEM_BUDGET_MB. Returns the job handle
    (kept alive while the child runs) or None if the limit could not be set."""
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(f, ctypes.c_ulonglong) for f in ("ReadOperationCount", "WriteOperationCount",
                    "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC_LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC_LIMIT), ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = EXTENDED_LIMIT()
    info.BasicLimitInformation.LimitFlags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.ProcessMemoryLimit = MEM_BUDGET_MB * 2 ** 20
    ok = kernel32.SetInformationJobObject(wintypes.HANDLE(job), 9,  # JobObjectExtendedLimitInformation
                                          ctypes.byref(info), ctypes.sizeof(info))
    ok = ok and kernel32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(int(proc._handle)))
    if not ok:
        print("    (warning: could not apply the Windows memory limit)", flush=True)
    return job


def run_process(cmd, timeout):
    """Run cmd under the memory limit and a wall-clock timeout; return (status, stdout, wall_s)."""
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        t0 = time.monotonic()
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, text=True)
        job = windows_memory_job(proc)
        status = None
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = proc.wait()
            status = "timeout"
        if job:
            import ctypes
            ctypes.windll.kernel32.CloseHandle(job)
        wall = time.monotonic() - t0
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(), err.read()
        if status is None:
            if code == 0:
                status = "ok"
            elif code == 2 or "out of memory" in stderr:
                status = "memory_failure"
            else:
                status = f"error({code})"
            if code != 0:
                sys.stderr.write(stderr)
        return status, stdout, wall


class Recorder:
    def __init__(self, study, tag=""):
        self.study = study
        out = ROOT / "results" / f"raw_{study}{tag}"
        out.mkdir(parents=True, exist_ok=True)
        self.sf = open(out / "samples.csv", "w", newline="")
        self.rf = open(out / "runs.csv", "w", newline="")
        self.samples = csv.DictWriter(self.sf, SAMPLE_COLS)
        self.runs = csv.DictWriter(self.rf, RUN_COLS)
        self.samples.writeheader()
        self.runs.writeheader()

    def run(self, method, cmd, n, timeout, batch=1, warmup=WARMUP, reps=REPS, distribution="dense",
            density=1.0, block="", note=""):
        """Run one configuration, record it, return (status, median time per multiplication)."""
        print(f"[{time.strftime('%H:%M:%S')}] {self.study} {method} n={n} {distribution} "
              f"d={density:g} block={block} batch={batch} w={warmup} r={reps} {note}", flush=True)
        status, stdout, wall = run_process(cmd, timeout)
        info = {"nnz_a": "", "nnz_b": "", "nnz_c": "", "csr_bytes": "", "peak_rss_kb": ""}
        measured = []
        base = {"study": self.study, "method": method, "n": n, "distribution": distribution,
                "density_target": f"{density:g}", "block": block}
        for line in stdout.strip().splitlines():
            f = line.split(",")
            if f[0] == "peak_rss_kb":
                info["peak_rss_kb"] = int(f[1])
            elif f[0] == "csr_bytes":
                info["csr_bytes"] = int(f[1])
            elif f[0] == "nnz":
                info["nnz_a"], info["nnz_b"], info["nnz_c"] = f[1:4]
            elif f[0] in ("warmup", "measure"):
                t = int(f[2]) / batch / 1e9
                row = dict(base, phase=f[0], rep=f[1], batch=batch, elapsed_ns=f[2],
                           time_per_mult_s=f"{t:.9f}", checksum=f[3], numeric_type="float64",
                           threads=1, nnz_c=f[4] if len(f) > 4 else "",
                           convert_in_s=f"{int(f[5]) / 1e9:.9f}" if len(f) > 5 and f[5] != "-1" else "",
                           convert_out_s=f"{int(f[6]) / 1e9:.9f}" if len(f) > 6 and f[6] != "-1" else "")
                self.samples.writerow(row)
                if f[0] == "measure":
                    measured.append(t)
        med = sorted(measured)[len(measured) // 2] if measured else None
        if status == "ok" and med is not None and med > TIME_BUDGET_S:
            status = "over_time_budget"
        self.runs.writerow(dict(base, status=status, warmup=warmup, reps=reps, batch=batch,
                                process_wall_s=f"{wall:.3f}", note=note, **info))
        self.sf.flush()
        self.rf.flush()
        print(f"    -> {status} rss={info['peak_rss_kb']} kB nnz_c={info['nnz_c']} wall={wall:.1f} s "
              f"median={med if med is not None else float('nan'):.4f} s", flush=True)
        return status, med

    def close(self):
        self.sf.close()
        self.rf.close()


def study_dense(rec, methods, sizes, tile):
    stopped, last = set(), {}
    for n in sizes:
        pa, pb = input_files(n)
        batch = batch_for(n)
        for m in methods:
            if m in stopped:
                continue
            warmup, reps, note = WARMUP, REPS, ""
            timeout = TIME_BUDGET_S * (warmup + reps) * batch + 60
            if m in last:
                predicted = last[m][1] * (n / last[m][0]) ** 3
                if predicted > TIME_BUDGET_S:
                    warmup, reps, timeout = 0, 1, TIME_BUDGET_S
                    note = f"budget probe (cubic prediction {predicted:.0f} s)"
            block = tile if m.startswith("blocked") else ""
            status, med = rec.run(m, dense_cmd(m, pa, pb, n, warmup, reps, batch, block or 0), n,
                                  timeout, batch, warmup, reps, block=block, note=note)
            if status != "ok":
                stopped.add(m)
            elif med is not None:
                last[m] = (n, med)


def study_blocks(rec):
    for n in BLOCK_N:
        pa, pb = input_files(n)
        for m in ("blocked", "blocked_unroll"):
            for bs in BLOCK_SIZES:
                rec.run(m, dense_cmd(m, pa, pb, n, WARMUP, REPS, 1, bs), n,
                        TIME_BUDGET_S * (WARMUP + REPS) + 60, block=bs)


def study_sparse(rec, densities, tile):
    timeout = TIME_BUDGET_S * (WARMUP + REPS) + 60
    for n in SPARSE_N:
        for dist in DISTRIBUTIONS:
            for d in densities:
                ca, cb, da, db = sparse_files(n, d, dist, dense=True)
                kw = dict(distribution=dist, density=d)
                rec.run("csr", sparse_cmd("csr", ca, cb, n, WARMUP, REPS), n, timeout, **kw)
                rec.run("csr_from_dense", sparse_cmd("dense", da, db, n, WARMUP, REPS), n, timeout, **kw)
                rec.run(BEST_DENSE, dense_cmd(BEST_DENSE, da, db, n, WARMUP, REPS, 1, tile),
                        n, timeout, block=tile, **kw)
                if d in SPARSE_IJK[n]:
                    rec.run("ijk", dense_cmd("ijk", da, db, n, WARMUP, REPS, 1), n, timeout, **kw)
                for p in (da, db):  # dense copies are only needed for this configuration
                    p.unlink()


def study_scale(rec):
    for series, points in SCALE.items():
        last = None
        for n, d in points:
            warmup, reps, note = WARMUP, REPS, series
            timeout = TIME_BUDGET_S * (warmup + reps) + 60
            ca, cb = sparse_files(n, d, "uniform")
            if last is not None:
                # SpGEMM work ~ nnz_a * (nnz per row of B); scale the last median by that ratio
                work = n * (n * d) ** 2
                predicted = last[1] * work / last[0]
                if predicted > TIME_BUDGET_S:
                    warmup, reps, timeout = 0, 1, TIME_BUDGET_S
                    note += f"; budget probe (prediction {predicted:.0f} s)"
            status, med = rec.run("csr", sparse_cmd("csr", ca, cb, n, warmup, reps), n, timeout,
                                  warmup=warmup, reps=reps, distribution="uniform", density=d,
                                  note=note)
            for p in (ca, cb):  # large inputs: free the disk space once measured
                p.unlink()
            if status != "ok":
                break
            last = (n * (n * d) ** 2, med)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, choices=["dense", "blocks", "sparse", "scale"])
    ap.add_argument("--methods", nargs="+", default=DENSE_METHODS)
    ap.add_argument("--sizes", nargs="+", type=int, default=DENSE_SIZES)
    ap.add_argument("--densities", nargs="+", type=float, default=DENSITIES)
    ap.add_argument("--tag", default="", help="suffix of the output directory")
    ap.add_argument("--block", type=int, default=DEFAULT_BLOCK,
                    help="tile side of the blocked methods (run_all.py passes the sweep's best)")
    args = ap.parse_args()
    rec = Recorder(args.study, args.tag)
    if args.study == "dense":
        study_dense(rec, args.methods, args.sizes, args.block)
    elif args.study == "blocks":
        study_blocks(rec)
    elif args.study == "sparse":
        study_sparse(rec, args.densities, args.block)
    else:
        study_scale(rec)
    rec.close()


if __name__ == "__main__":
    main()
