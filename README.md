# Assignment 2: Optimised Dense Matrix Multiplication and Sparse Matrices

Big Data (GCID, ULPGC), academic year 2026-2027. Author: Diego Ruiz Rodríguez.
Repository: https://github.com/DiegoRuiz16/BD-Individual_addignment_2

Two research questions:

* **Part A.** How much do loop interchange, transposition of B, cache blocking and register
  blocking (unroll-and-jam) change the runtime and memory of the sequential dense O(n³)
  multiplication, relative to the Assignment 1 baseline, as n grows?
* **Part B.** For which tested densities and nonzero distributions does a CSR sparse
  multiplication beat the best dense method, and how much of that advantage survives
  conversion costs and the fill-in of the product?

Everything was developed and measured on Windows 11. The report is in `report/`.

## Reused material

The baseline is the C implementation of Assignment 1: its i-j-k kernel is copied verbatim as
method `ijk` in `src/c/dense.c`. The dense input generator and the design of the benchmark,
validation and analysis scripts are extended from Assignment 1.

## Layout

```
src/c/dense.c            Dense methods: ijk, transpose, ikj, blocked, blocked_unroll
src/c/sparse.c           CSR x CSR multiplication (Gustavson), own code, no library
scripts/run_all.py       Runs everything below in order (build -> validate -> benchmarks -> analysis)
scripts/common.py        Input generators (dense and sparse), CSR file format, command lines
scripts/build.py         Compiles with the fixed flags; records GCC's vectorisation report
scripts/validate.py      Correctness tests -> results/validation.csv
scripts/benchmark.py     Benchmark harness, 4 studies -> results/raw_<study>/
scripts/analyze.py       Tables and figures -> results/summary_*.{csv,md}, results/figures/
scripts/environment.py   Hardware/software record -> results/environment.txt
scripts/build_report.py  report.md -> report.html -> report.pdf (pandoc + Edge)
report/                  Report (Markdown source, HTML, PDF)
results/                 Raw measurements, summaries, figures, environment, logs
data/                    Generated inputs (not committed; regenerated on demand)
```

## Requirements (Windows 10/11)

* **GCC (MinGW-w64)**: `winget install BrechtSanders.WinLibs.POSIX.UCRT`, then open a new
  terminal and check `gcc --version`.
* **Python 3.12+** (`winget install Python.Python.3.12`) with
  `python -m pip install numpy scipy matplotlib`. NumPy generates the inputs and is the
  correctness reference; SciPy only cross-checks nonzero counts in `validate.py`. Neither is
  used inside any benchmarked kernel.
* About 3 GB of free disk for generated inputs. Run the project from a folder **outside
  OneDrive** (for example `C:\BD\assignment2`), otherwise OneDrive tries to upload them.
* `pandoc` and Microsoft Edge only to rebuild the PDF report.

## Reproduce

In PowerShell, inside the project folder:

```powershell
python scripts\run_all.py --check   # build + correctness tests, about 1 min; must print ALL PASSED
python scripts\run_all.py           # whole study, about 1-1.5 h; keep the laptop plugged in and idle
```

`run_all.py` runs, in order: `build.py`, `validate.py`, `environment.py`,
`benchmark.py --study blocks|dense|sparse|scale` and `analyze.py`. The tile side of the blocked
methods is the one with the lowest median time in the tile sweep (saved in
`results/chosen_tile.txt`). A finished study is skipped if the script is started again.

Main results on the measurement machine (Intel Core i7-12700H, Windows 11, whole run 110 min):
`blocked_unroll` (tile 512) is the fastest dense method at every n (13.0x the baseline at
n = 2048, 3.8-4.3 GFLOP/s); the CSR kernel beats it up to 20 % density for uniform inputs and up
to 44 % for banded inputs, and loses from 50 % and 61 % respectively. Details in the report.

Run one configuration by hand:

```powershell
python -c "import sys; sys.path.insert(0,'scripts'); import common; print(common.input_files(1024))"
build\dense.exe blocked_unroll data\A_1024.bin data\B_1024.bin 1024 2 5 1 256
#  arguments: method A B n warmup reps batch [tile] [C_out.bin]
build\sparse.exe csr data\SA_uniform_2048_0.01.csr data\SB_uniform_2048_0.01.csr 2048 2 5
#  arguments: mode(csr|dense) A B n warmup reps [C_out_dense.bin]
```

## Inputs

* **Dense (Part A)**: generated as in Assignment 1. NumPy PCG64 seeded with `[2026, n]`, uniform
  [0, 1), float64, `data/A_<n>.bin`, `data/B_<n>.bin` (little-endian, row-major).
* **Sparse (Part B)**: NumPy PCG64 seeded with `[2026, n, round(density·10⁶), dist]`
  (dist 1 = uniform, 2 = banded); A is drawn first, then B, from the same stream. Values are
  uniform on (0, 1] (never an explicit zero).
  * `uniform`: each row gets k ~ Binomial(n, d) distinct columns chosen uniformly at random.
  * `banded`: all entries with |i − j| ≤ w are nonzero, w = round((d·n − 1)/2).
  * CSR file: int64 n, int64 nnz, int64 row_ptr[n+1], int32 col[nnz], float64 val[nnz].
    For dense methods the same matrices are expanded to dense files.

Density = percentage of nonzero entries; sparsity = 100 % − density.

## Fixed settings

| Setting | Value |
|---|---|
| Language / compiler | C11, `gcc -O2 -std=c11` (MinGW-w64), no `-march`, no `-ffast-math` |
| Numeric type | IEEE-754 binary64 |
| Threads | 1 (no OpenMP, no BLAS) |
| Timer | `QueryPerformanceCounter` |
| Dense timed region | the multiplication only (inputs loaded and C, Bᵀ allocated before timing; zeroing C and transposing B are inside) |
| Sparse timed region | symbolic + allocation of C + numeric pass; conversions dense→CSR and CSR→dense timed separately in `dense` mode |
| Repetitions | 2 warm-up + 5 measured per configuration, fresh process per configuration |
| Batching | dense: `max(1, round(2^25 / n³))` multiplications per sample |
| Memory metric | peak working set (`GetProcessMemoryInfo`); sparse also reports CSR array bytes |
| Budgets | 150 s per multiplication; 4 GiB per process (Job Object memory limit) |

## Raw data format

`results/raw_<study>/samples.csv`: `study, method, n, distribution, density_target, block, phase
(warmup|measure), rep, batch, elapsed_ns, time_per_mult_s, convert_in_s, convert_out_s, checksum,
nnz_c, numeric_type, threads`.

`results/raw_<study>/runs.csv`: one row per process: `study, method, n, distribution,
density_target, block, status (ok|timeout|memory_failure|...), warmup, reps, batch, nnz_a, nnz_b,
nnz_c, csr_bytes, peak_rss_kb, process_wall_s, note`.

Method names: `ijk`, `transpose`, `ikj`, `blocked`, `blocked_unroll` (dense);
`csr` (CSR input, kernel only); `csr_from_dense` (dense input, conversions timed separately).

## Use of generative AI

An AI assistant (Claude, Anthropic) helped write the code, scripts and report text. All
reported numbers are outputs of these scripts, kept unmodified in `results/raw_*`.
