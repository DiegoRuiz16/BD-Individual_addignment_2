"""Correctness checks for the four dense methods and the sparse (CSR) method.

Every case runs each method once (no warm-up, one repetition) and compares its
output with an exact known answer or with NumPy's matmul (trusted reference).
For sparse cases the number of nonzeros of the product is also compared with
SciPy's sparse product, and the CSR program is run in both input modes (CSR
files, and dense files converted inside the program).

Tolerance (as in Assignment 1): rtol = atol = 1e-12. Values lie in [0, 1], so
|C_ij| <= n and the rounding error of a length-n dot product is bounded by about
n * 2^-53 * sum_k |a_ik b_kj|, below 1e-12 relative for the tested sizes.

All methods add the products of each C_ij in increasing k order without
reassociation (no -ffast-math, and GCC does not vectorise these loops at -O2),
and the sparse method only skips exact zeros, so the outputs are expected to be
bit-for-bit identical to the Assignment 1 ijk kernel. This is checked too.
Results go to results/validation.csv.
"""
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CSR, DENSE_METHODS, ROOT, dense_cmd, generate, read_matrix,  # noqa: E402
                    sparse_cmd, sparse_pair, write_csr, write_matrix)

RTOL = 1e-12
ATOL = 1e-12
BLOCKS = [16, 64, 256]


def dense_cases():
    yield "2x2 known", np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]]), \
        np.array([[19.0, 22.0], [43.0, 50.0]])
    a, _ = generate(16, seed=1)
    yield "identity x A (n=16)", np.eye(16), a, a
    yield "A x identity (n=16)", a, np.eye(16), a
    yield "zero x A (n=16)", np.zeros((16, 16)), a, np.zeros((16, 16))
    for n in (3, 50, 64, 100, 257):  # 50 < tile, 100 and 257 not multiples of the tile side or of 4
        a, b = generate(n, seed=7)
        yield f"random vs NumPy (n={n})", a, b, a @ b


def sparse_cases():
    a, _ = generate(40, seed=3)
    yield "sparse identity x A (n=40)", CSR.from_dense(np.eye(40)), CSR.from_dense(a)
    yield "sparse zero x A (n=40)", CSR.from_dense(np.zeros((40, 40))), CSR.from_dense(a)
    yield "sparse fully dense (n=40)", CSR.from_dense(a), CSR.from_dense(a.T.copy())
    for dist, n, d in [("uniform", 200, 0.01), ("uniform", 200, 0.05), ("uniform", 300, 0.2),
                       ("banded", 200, 0.05), ("banded", 301, 0.2)]:
        a, b = sparse_pair(n, d, dist)
        yield f"{dist} d={d:g} (n={n})", a, b


def record(rows, name, method, n, c, expected, ref_ijk, extra=None):
    err = np.abs(c - expected)
    nz = expected != 0
    rel = (err[nz] / np.abs(expected[nz])).max() if nz.any() else 0.0
    passed = bool(np.allclose(c, expected, rtol=RTOL, atol=ATOL))
    identical = bool(np.array_equal(c, ref_ijk))
    row = {"case": name, "method": method, "n": n, "max_abs_err": f"{err.max():.3e}",
           "max_rel_err": f"{rel:.3e}", "passed": passed, "bitwise_identical_to_ijk": identical,
           "nnz_c": "", "nnz_c_scipy": ""}
    row.update(extra or {})
    rows.append(row)
    print(f"{name:30s} {method:14s} max_abs={err.max():.2e} pass={passed} identical={identical} "
          f"{extra or ''}")
    return passed


def run_dense(method, a, b, tmp, block=64):
    n = a.shape[0]
    pa, pb, pc = tmp / "A.bin", tmp / "B.bin", tmp / "C.bin"
    write_matrix(pa, a)
    write_matrix(pb, b)
    subprocess.run(dense_cmd(method, pa, pb, n, 0, 1, 1, block, pc), check=True,
                   capture_output=True, text=True)
    return read_matrix(pc, n)


def run_sparse(mode, a, b, tmp):
    n = a.n
    if mode == "csr":
        pa, pb = tmp / "A.csr", tmp / "B.csr"
        write_csr(pa, a)
        write_csr(pb, b)
    else:
        pa, pb = tmp / "A.bin", tmp / "B.bin"
        write_matrix(pa, a.to_dense())
        write_matrix(pb, b.to_dense())
    pc = tmp / "C.bin"
    out = subprocess.run(sparse_cmd(mode, pa, pb, n, 0, 1, pc), check=True,
                         capture_output=True, text=True).stdout
    nnz_c = next(int(l.split(",")[3]) for l in out.splitlines() if l.startswith("nnz,"))
    return read_matrix(pc, n), nnz_c


def main():
    rows, ok = [], True
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for name, a, b, expected in dense_cases():
            n = a.shape[0]
            ref = run_dense("ijk", a, b, tmp)
            for m in DENSE_METHODS:
                for bs in (BLOCKS if m.startswith("blocked") else [None]):
                    c = run_dense(m, a, b, tmp, bs or 64)
                    ok &= record(rows, name, m if bs is None else f"{m}(b={bs})", n, c,
                                 expected, ref)
        for name, a, b in sparse_cases():
            da, db = a.to_dense(), b.to_dense()
            expected = da @ db
            ref = run_dense("ijk", da, db, tmp)
            scipy_nnz = int((sp.csr_matrix(da) @ sp.csr_matrix(db)).count_nonzero())
            for mode in ("csr", "dense"):
                c, nnz_c = run_sparse(mode, a, b, tmp)
                ok &= nnz_c == scipy_nnz
                ok &= record(rows, name, f"sparse({mode} in)", a.n, c, expected, ref,
                             {"nnz_c": nnz_c, "nnz_c_scipy": scipy_nnz})
    out = ROOT / "results" / "validation.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    all_identical = all(r["bitwise_identical_to_ijk"] for r in rows)
    print(f"bitwise identical to ijk in every case: {all_identical}")
    print("ALL PASSED" if ok else "FAILURES", "->", out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
