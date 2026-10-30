"""Run the whole study end to end on this Windows machine.

    python scripts\\run_all.py           build, validate, record the environment,
                                         run the four benchmark studies, make the
                                         tables and figures (about 1-1.5 h)
    python scripts\\run_all.py --check   only build and validate (about 1 min)

Steps:
  1. build.py        compile build\\dense.exe and build\\sparse.exe
  2. validate.py     correctness tests; the run stops if any test fails
  3. environment.py  hardware/software record -> results\\environment.txt
  4. benchmark.py --study blocks   tile-side sweep; the tile with the lowest
                                   median time of blocked_unroll is used below
  5. benchmark.py --study dense    Part A
  6. benchmark.py --study sparse   Part B, matched comparison
  7. benchmark.py --study scale    Part B, large sparse sizes
  8. analyze.py      results\\summary_*.md/csv and results\\figures\\*.png

A study that finished is not repeated if the script is started again (its
directory contains a DONE file), so an interrupted run can be resumed. Console
output of each study is also saved in results\\benchmark_<study>_log.txt.
Keep the laptop plugged in and do not use it while the benchmarks run.
"""
import argparse
import csv
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"


def step(title):
    print(f"\n=== {title} ===", flush=True)


def run(args, log=None):
    """Run a Python script, echo its output and optionally save it to `log`."""
    cmd = [sys.executable, "-u", *args]
    with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          encoding="utf-8", errors="replace") as proc:
        out = open(log, "w", encoding="utf-8") if log else None
        for line in proc.stdout:
            print(line, end="", flush=True)
            if out:
                out.write(line)
                out.flush()
        if out:
            out.close()
    if proc.returncode != 0:
        sys.exit(f"\n{args[0]} failed (exit code {proc.returncode}); see the messages above.")


def best_tile():
    """Tile side with the lowest median time of blocked_unroll in the sweep."""
    times = defaultdict(list)
    with open(RESULTS / "raw_blocks" / "samples.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["phase"] == "measure" and r["method"] == "blocked_unroll":
                times[int(r["block"])].append(float(r["time_per_mult_s"]))
    return min(times, key=lambda b: statistics.median(times[b]))


def study(name, *extra):
    done = RESULTS / f"raw_{name}" / "DONE"
    if done.exists():
        print(f"(already finished, skipped; delete {done.relative_to(ROOT)} to run it again)")
        return
    t0 = time.monotonic()
    run([str(SCRIPTS / "benchmark.py"), "--study", name, *extra],
        log=RESULTS / f"benchmark_{name}_log.txt")
    done.write_text(f"finished in {time.monotonic() - t0:.0f} s\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only build and validate")
    args = ap.parse_args()
    if "onedrive" in str(ROOT).lower():
        print("WARNING: this folder is inside OneDrive. The benchmarks write several GB of input\n"
              "files to data\\ and OneDrive would try to upload them. Copy the project to a folder\n"
              "such as C:\\BD\\assignment2 and run it from there.", flush=True)
        if not args.check:
            sys.exit(1)
    RESULTS.mkdir(exist_ok=True)
    start = time.monotonic()

    step("1/8 Build")
    run([str(SCRIPTS / "build.py")])
    step("2/8 Correctness tests")
    run([str(SCRIPTS / "validate.py")])
    if args.check:
        print("\nBuild and validation OK. Run without --check to start the benchmarks.")
        return
    step("3/8 Environment")
    run([str(SCRIPTS / "environment.py")])
    step("4/8 Tile-side sweep")
    study("blocks")
    tile = best_tile()
    (RESULTS / "chosen_tile.txt").write_text(f"{tile}\n")
    print(f"Best tile side for blocked_unroll: {tile}")
    step("5/8 Part A: dense methods")
    study("dense", "--block", str(tile))
    step("6/8 Part B: sparse against dense")
    study("sparse", "--block", str(tile))
    step("7/8 Part B: large sparse sizes")
    study("scale")
    step("8/8 Tables and figures")
    run([str(SCRIPTS / "analyze.py")])
    print(f"\nAll done in {(time.monotonic() - start) / 60:.0f} min. Results are in the results folder.")


if __name__ == "__main__":
    main()
