"""Shared configuration: input generators, file formats and program command lines.

Dense inputs (Part A) are exactly those of Assignment 1: NumPy PCG64 seeded
with [2026, n], uniform [0, 1), float64, row-major little-endian files.

Sparse inputs (Part B) are generated here (see `sparse_pair`) and stored as
CSR files; when a dense method needs the same logical matrix, the CSR matrix
is expanded into a dense file of the Assignment 1 format.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BUILD = ROOT / "build"
EXE = ".exe"

SEED = 2026
VALUE_RANGE = (0.0, 1.0)
DENSE_METHODS = ["ijk", "transpose", "ikj", "blocked", "blocked_unroll"]
DEFAULT_BLOCK = 256  # used only if no tile-side sweep is available; run_all.py picks the best tile
DISTRIBUTIONS = {"uniform": 1, "banded": 2}


def dense_cmd(method, pa, pb, n, warmup, reps, batch, block=DEFAULT_BLOCK, out=None):
    cmd = [str(BUILD / f"dense{EXE}"), method, str(pa), str(pb), str(n), str(warmup), str(reps),
           str(batch), str(block)]
    return cmd + [str(out)] if out else cmd


def sparse_cmd(mode, pa, pb, n, warmup, reps, out=None):
    cmd = [str(BUILD / f"sparse{EXE}"), mode, str(pa), str(pb), str(n), str(warmup), str(reps)]
    return cmd + [str(out)] if out else cmd


# ---------------------------------------------------------------- dense inputs
def generate(n, seed=SEED):
    """Deterministic A and B for size n (identical to Assignment 1)."""
    rng = np.random.default_rng([seed, n])
    lo, hi = VALUE_RANGE
    a = rng.uniform(lo, hi, size=(n, n))
    b = rng.uniform(lo, hi, size=(n, n))
    return a, b


def write_matrix(path, m):
    np.asarray(m, dtype="<f8").tofile(path)


def read_matrix(path, n):
    return np.fromfile(path, dtype="<f8").reshape(n, n)


def input_files(n):
    """Write (once) and return the dense input files for size n."""
    DATA.mkdir(exist_ok=True)
    pa, pb = DATA / f"A_{n}.bin", DATA / f"B_{n}.bin"
    if not (pa.exists() and pb.exists()):
        a, b = generate(n)
        write_matrix(pa, a)
        write_matrix(pb, b)
    return pa, pb


# --------------------------------------------------------------- sparse inputs
class CSR:
    """Minimal CSR container (row_ptr int64, col int32, val float64, sorted columns)."""

    def __init__(self, n, row_ptr, col, val):
        self.n, self.row_ptr, self.col, self.val = n, row_ptr, col, val

    @property
    def nnz(self):
        return int(self.row_ptr[-1])

    def to_dense(self):
        d = np.zeros((self.n, self.n))
        rows = np.repeat(np.arange(self.n), np.diff(self.row_ptr))
        d[rows, self.col] = self.val
        return d

    @classmethod
    def from_dense(cls, d):
        n = d.shape[0]
        rows, cols = np.nonzero(d)
        row_ptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(rows, minlength=n), out=row_ptr[1:])
        return cls(n, row_ptr, cols.astype(np.int32), d[rows, cols].astype(np.float64))


def write_csr(path, m):
    with open(path, "wb") as f:
        np.array([m.n, m.nnz], dtype="<i8").tofile(f)
        m.row_ptr.astype("<i8").tofile(f)
        m.col.astype("<i4").tofile(f)
        m.val.astype("<f8").tofile(f)


def read_csr(path):
    with open(path, "rb") as f:
        n, nnz = np.fromfile(f, dtype="<i8", count=2)
        row_ptr = np.fromfile(f, dtype="<i8", count=n + 1)
        col = np.fromfile(f, dtype="<i4", count=nnz)
        val = np.fromfile(f, dtype="<f8", count=nnz)
    return CSR(int(n), row_ptr, col, val)


def _values(rng, k):
    # uniform on (0, 1]: an explicitly stored entry is never an exact zero
    return 1.0 - rng.random(k)


def sparse_matrix(rng, n, density, distribution):
    """One n x n sparse matrix with the requested target density (fraction of nonzeros).

    uniform: every row draws k ~ Binomial(n, density) distinct columns uniformly
             at random (so each entry is nonzero independently with prob. density).
    banded : all entries with |i - j| <= w are nonzero, w = round((density*n - 1) / 2),
             so the band holds about density * n^2 entries (fewer at the corners).
    """
    if distribution == "uniform":
        counts = rng.binomial(n, density, size=n)
        cols = [np.sort(rng.choice(n, size=k, replace=False)) for k in counts]
        col = np.concatenate(cols).astype(np.int32) if n else np.zeros(0, np.int32)
    elif distribution == "banded":
        w = max(0, round((density * n - 1) / 2))
        cols = [np.arange(max(0, i - w), min(n, i + w + 1)) for i in range(n)]
        counts = np.array([len(c) for c in cols])
        col = np.concatenate(cols).astype(np.int32)
    else:
        raise ValueError(distribution)
    row_ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=row_ptr[1:])
    return CSR(n, row_ptr, col, _values(rng, len(col)))


def sparse_seed(n, density, distribution, seed=SEED):
    return [seed, n, round(density * 1_000_000), DISTRIBUTIONS[distribution]]


def sparse_pair(n, density, distribution, seed=SEED):
    """Deterministic A and B (CSR); seed [2026, n, density in ppm, distribution id]."""
    rng = np.random.default_rng(sparse_seed(n, density, distribution, seed))
    return sparse_matrix(rng, n, density, distribution), sparse_matrix(rng, n, density, distribution)


def sparse_files(n, density, distribution, dense=False):
    """Write (once) and return CSR files, plus dense files of the same matrices if asked."""
    DATA.mkdir(exist_ok=True)
    tag = f"{distribution}_{n}_{density:g}"
    pa, pb = DATA / f"SA_{tag}.csr", DATA / f"SB_{tag}.csr"
    da, db = DATA / f"SA_{tag}.bin", DATA / f"SB_{tag}.bin"
    if not (pa.exists() and pb.exists()):
        a, b = sparse_pair(n, density, distribution)
        write_csr(pa, a)
        write_csr(pb, b)
    if dense and not (da.exists() and db.exists()):
        write_matrix(da, read_csr(pa).to_dense())
        write_matrix(db, read_csr(pb).to_dense())
    return (pa, pb, da, db) if dense else (pa, pb)
