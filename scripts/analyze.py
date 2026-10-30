"""Turn the raw measurements into summary tables and figures.

Inputs : results/raw_{dense,blocks,sparse,scale}/{samples,runs}.csv
Outputs: results/summary_<study>.{csv,md}, results/figures/*.png
Only "measure" samples enter the statistics (median and IQR = Q3 - Q1);
warm-up samples stay in the raw files.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DENSE_METHODS, ROOT, read_csr, sparse_files  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"
LABEL = {"ijk": "ijk (A1 baseline)", "transpose": "transposed B", "ikj": "ikj (loop interchange)",
         "blocked": "blocked ikj", "blocked_unroll": "blocked + 4-row unroll-and-jam",
         "csr": "CSR SpGEMM (kernel)", "csr_total": "CSR incl. conversions"}
COLOR = {"ijk": "#2a78d6", "transpose": "#eb6834", "ikj": "#1baf7a", "blocked": "#9b59b6",
         "blocked_unroll": "#d4a106", "csr": "#c0392b", "csr_total": "#7f8c8d"}
MARKER = {"ijk": "o", "transpose": "v", "ikj": "s", "blocked": "D", "blocked_unroll": "^",
          "csr": "P", "csr_total": "X"}

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


DENSITY_TICKS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]


def density_axis(ax):
    ax.set_xticks(DENSITY_TICKS, [f"{t:g}" for t in DENSITY_TICKS])
    ax.minorticks_off()


def products(n, density, distribution):
    """Exact number of scalar products of Gustavson's algorithm: sum_k nnz(A[:, k]) * nnz(B[k, :]).

    Regenerates the CSR inputs deterministically if they are not on disk."""
    pa, pb = sparse_files(n, density, distribution)
    a, b = read_csr(pa), read_csr(pb)
    return int(np.dot(np.bincount(a.col, minlength=n).astype(np.int64), np.diff(b.row_ptr)))


def key(r):
    return (r["method"], int(r["n"]), r["distribution"], r["density_target"], r["block"])


def load(study):
    dirs = [d for d in (RES / f"raw_{study}", RES / f"raw_{study}_high") if (d / "runs.csv").exists()]
    if not dirs:
        return None, None
    samples = defaultdict(lambda: defaultdict(list))
    runs = []
    for d in dirs:
        load_dir(d, samples, runs)
    return samples, runs


def load_dir(d, samples, runs):
    with open(d / "samples.csv") as f:
        for r in csv.DictReader(f):
            if r["phase"] != "measure":
                continue
            s = samples[key(r)]
            s["t"].append(float(r["time_per_mult_s"]))
            if r["convert_in_s"]:
                s["cin"].append(float(r["convert_in_s"]))
                s["cout"].append(float(r["convert_out_s"]))
                s["total"].append(float(r["time_per_mult_s"]) + float(r["convert_in_s"])
                                  + float(r["convert_out_s"]))
    with open(d / "runs.csv") as f:
        runs.extend(csv.DictReader(f))


def stats(x):
    if not len(x):
        return None, None
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return med, q3 - q1


def g(x, digits=4):
    if x is None or x == "":
        return ""
    return x if isinstance(x, str) else f"{x:.{digits}g}"


def write_tables(name, rows, cols, md_cols):
    with open(RES / f"summary_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(RES / f"summary_{name}.md", "w") as f:
        f.write("| " + " | ".join(h for h, _ in md_cols) + " |\n")
        f.write("|" + "|".join("---" if i < 1 else "---:" for i in range(len(md_cols))) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(fn(r) for _, fn in md_cols) + " |\n")


# ------------------------------------------------------------------ Part A
def dense_part():
    samples, runs = load("dense")
    if runs is None:
        return
    rows, base = [], {}
    for run in runs:
        n = int(run["n"])
        t = samples[key(run)]["t"]
        med, iqr = stats(t)
        row = {"method": run["method"], "n": n, "status": run["status"], "reps": len(t),
               "batch": run["batch"], "block": run["block"], "median_s": med, "iqr_s": iqr,
               "iqr_rel": iqr / med if med else None, "gflops": 2 * n ** 3 / med / 1e9 if med else None,
               "peak_rss_mb": int(run["peak_rss_kb"]) / 1024 if run["peak_rss_kb"] else None,
               "note": run["note"]}
        if run["status"] != "ok":
            med = None
            row["median_s"] = None
        if run["method"] == "ijk" and med:
            base[n] = med
        rows.append(row)
    for r in rows:
        r["speedup_vs_ijk"] = base[r["n"]] / r["median_s"] if r["median_s"] and r["n"] in base else None
    rows.sort(key=lambda r: (r["n"], DENSE_METHODS.index(r["method"])))
    cols = ["method", "n", "status", "reps", "batch", "block", "median_s", "iqr_s", "iqr_rel", "gflops",
            "speedup_vs_ijk", "peak_rss_mb", "note"]
    write_tables("dense", rows, cols, [
        ("Method", lambda r: r["method"]), ("n", lambda r: str(r["n"])),
        ("Status", lambda r: r["status"] + (f" ({r['note']})" if r["status"] != "ok" and r["note"] else "")),
        ("Reps", lambda r: str(r["reps"])), ("Median (s)", lambda r: g(r["median_s"])),
        ("IQR (s)", lambda r: g(r["iqr_s"], 2)),
        ("IQR/med", lambda r: f"{100 * r['iqr_rel']:.1f}%" if r["iqr_rel"] is not None and r["median_s"] else ""),
        ("GFLOP/s", lambda r: f"{r['gflops']:.2f}" if r["median_s"] else ""),
        ("Speed-up vs ijk", lambda r: f"{r['speedup_vs_ijk']:.2f}" if r["speedup_vs_ijk"] else ""),
        ("Peak RSS (MB)", lambda r: f"{r['peak_rss_mb']:.1f}" if r["peak_rss_mb"] else "")])

    by = defaultdict(list)
    for r in rows:
        if r["median_s"]:
            by[r["method"]].append(r)
    # time vs n
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in DENSE_METHODS:
        pts = by[m]
        ax.errorbar([p["n"] for p in pts], [p["median_s"] for p in pts], yerr=[p["iqr_s"] / 2 for p in pts],
                    marker=MARKER[m], color=COLOR[m], label=LABEL[m], ms=4, lw=1.3, capsize=2)
    for r in rows:
        if r["status"] == "timeout":
            ax.scatter([r["n"]], [150], marker="x", color=COLOR[r["method"]], s=40, zorder=5)
    ns = np.array([64, 4096])
    ax.plot(ns, by["ikj"][0]["median_s"] * (ns / by["ikj"][0]["n"]) ** 3, "--", color="#999", lw=1,
            label="∝ n³")
    ax.set(xscale="log", yscale="log", xlabel="n (matrix side)", ylabel="median time per multiplication (s)")
    ax.set_xticks([64, 128, 256, 512, 1024, 2048, 4096], ["64", "128", "256", "512", "1024", "2048", "4096"])
    ax.legend(fontsize=8)
    fig.savefig(FIG / "dense_time_vs_n.png")
    plt.close(fig)
    # GFLOP/s vs n
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m in DENSE_METHODS:
        pts = by[m]
        ax.plot([p["n"] for p in pts], [p["gflops"] for p in pts], marker=MARKER[m], color=COLOR[m],
                label=LABEL[m], ms=4, lw=1.3)
    ax.set(xscale="log", xlabel="n (matrix side)", ylabel="GFLOP/s (2n³ / median time)", ylim=(0, None))
    ax.set_xticks([64, 128, 256, 512, 1024, 2048, 4096], ["64", "128", "256", "512", "1024", "2048", "4096"])
    ax.legend(fontsize=8)
    fig.savefig(FIG / "dense_gflops_vs_n.png")
    plt.close(fig)
    # memory vs n
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m in DENSE_METHODS:
        pts = [r for r in rows if r["method"] == m and r["peak_rss_mb"]]
        ax.plot([p["n"] for p in pts], [p["peak_rss_mb"] for p in pts], marker=MARKER[m], color=COLOR[m],
                label=LABEL[m], ms=4, lw=1.2)
    ns = np.array([64, 128, 256, 512, 1024, 2048, 4096])
    ax.plot(ns, 3 * 8 * ns ** 2 / 2 ** 20, "--", color="#999", lw=1, label="3 · 8n² bytes")
    ax.set(xscale="log", yscale="log", xlabel="n (matrix side)", ylabel="peak RSS (MiB)")
    ax.set_xticks(ns, [str(x) for x in ns])
    ax.legend(fontsize=8)
    fig.savefig(FIG / "dense_memory_vs_n.png")
    plt.close(fig)


def blocks_part():
    samples, runs = load("blocks")
    if runs is None:
        return
    rows = []
    for run in runs:
        med, iqr = stats(samples[key(run)]["t"])
        rows.append({"method": run["method"], "n": int(run["n"]), "block": int(run["block"]),
                     "status": run["status"], "median_s": med, "iqr_s": iqr,
                     "gflops": 2 * int(run["n"]) ** 3 / med / 1e9 if med else None})
    write_tables("blocks", rows, list(rows[0]), [
        ("Method", lambda r: r["method"]), ("n", lambda r: str(r["n"])), ("Tile b", lambda r: str(r["block"])),
        ("Median (s)", lambda r: g(r["median_s"])), ("IQR (s)", lambda r: g(r["iqr_s"], 2)),
        ("GFLOP/s", lambda r: f"{r['gflops']:.2f}" if r["gflops"] else "")])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for m in ("blocked", "blocked_unroll"):
        for n, ls in zip(sorted({r["n"] for r in rows}), ("-", "--")):
            pts = [r for r in rows if r["method"] == m and r["n"] == n]
            ax.errorbar([p["block"] for p in pts], [p["gflops"] for p in pts], marker=MARKER[m], color=COLOR[m],
                        ls=ls, label=f"{LABEL[m]}, n={n}", ms=4, lw=1.3)
    ax.set(xscale="log", xlabel="tile side b", ylabel="GFLOP/s", ylim=(0, None))
    ax.set_xticks([16, 32, 64, 128, 256, 512], ["16", "32", "64", "128", "256", "512"])
    ax.minorticks_off()
    ax.legend(fontsize=8)
    fig.savefig(FIG / "block_sweep.png")
    plt.close(fig)


# ------------------------------------------------------------------ Part B
def sparse_part():
    samples, runs = load("sparse")
    if runs is None:
        return
    cfg = defaultdict(dict)
    for run in runs:
        c = cfg[(int(run["n"]), run["distribution"], float(run["density_target"]))]
        s = samples[key(run)]
        c[run["method"]] = {"run": run, "t": stats(s["t"]), "cin": stats(s["cin"]), "cout": stats(s["cout"]),
                            "total": stats(s["total"])}
    rows = []
    for (n, dist, d), c in sorted(cfg.items()):
        csr, conv, dense, ijk = c.get("csr"), c.get("csr_from_dense"), c.get("blocked_unroll"), c.get("ijk")
        r0 = csr["run"]
        nnz_a, nnz_b, nnz_c = int(r0["nnz_a"]), int(r0["nnz_b"]), int(r0["nnz_c"])
        row = {"n": n, "distribution": dist, "density_target": d,
               "density_a": nnz_a / n ** 2, "density_b": nnz_b / n ** 2, "nnz_a": nnz_a, "nnz_b": nnz_b,
               "nnz_c": nnz_c, "density_c": nnz_c / n ** 2,
               "csr_status": r0["status"], "csr_median_s": csr["t"][0], "csr_iqr_s": csr["t"][1],
               "conv_kernel_median_s": conv["t"][0], "convert_in_median_s": conv["cin"][0],
               "convert_out_median_s": conv["cout"][0], "csr_total_median_s": conv["total"][0],
               "csr_total_iqr_s": conv["total"][1],
               "dense_best_median_s": dense["t"][0], "dense_best_iqr_s": dense["t"][1],
               "ijk_median_s": ijk["t"][0] if ijk else None, "ijk_iqr_s": ijk["t"][1] if ijk else None,
               "csr_rss_mb": int(r0["peak_rss_kb"]) / 1024, "csr_bytes_mb": int(r0["csr_bytes"]) / 2 ** 20,
               "conv_rss_mb": int(conv["run"]["peak_rss_kb"]) / 1024,
               "dense_rss_mb": int(dense["run"]["peak_rss_kb"]) / 1024}
        row["products"] = products(n, d, dist)
        row["ns_per_product"] = 1e9 * row["csr_median_s"] / row["products"] if row["products"] else None
        row["dense_ns_per_madd"] = 1e9 * row["dense_best_median_s"] / n ** 3
        row["speedup_kernel"] = row["dense_best_median_s"] / row["csr_median_s"]
        row["speedup_total"] = row["dense_best_median_s"] / row["csr_total_median_s"]
        rows.append(row)
    cols = list(rows[0])
    write_tables("sparse", rows, cols, [
        ("n", lambda r: str(r["n"])), ("Distribution", lambda r: r["distribution"]),
        ("Density A", lambda r: f"{100 * r['density_a']:.2f}%"), ("nnz(A)", lambda r: str(r["nnz_a"])),
        ("nnz(C)", lambda r: str(r["nnz_c"])), ("Density C", lambda r: f"{100 * r['density_c']:.1f}%"),
        ("CSR kernel (s)", lambda r: g(r["csr_median_s"], 3)),
        ("CSR IQR/med", lambda r: f"{100 * r['csr_iqr_s'] / r['csr_median_s']:.0f}%"),
        ("Conv. in (s)", lambda r: g(r["convert_in_median_s"], 3)),
        ("Conv. out (s)", lambda r: g(r["convert_out_median_s"], 3)),
        ("CSR total (s)", lambda r: g(r["csr_total_median_s"], 3)),
        ("Dense best (s)", lambda r: g(r["dense_best_median_s"], 3)),
        ("ijk (s)", lambda r: g(r["ijk_median_s"], 3)),
        ("Products / n³", lambda r: f"{r['products'] / r['n'] ** 3:.3g}"),
        ("ns/product", lambda r: f"{r['ns_per_product']:.2f}" if r["ns_per_product"] else ""),
        ("Dense/CSR kernel", lambda r: f"{r['speedup_kernel']:.3g}"),
        ("Dense/CSR total", lambda r: f"{r['speedup_total']:.3g}"),
        ("RSS CSR (MB)", lambda r: f"{r['csr_rss_mb']:.1f}"), ("RSS dense (MB)", lambda r: f"{r['dense_rss_mb']:.1f}")])

    ns = sorted({r["n"] for r in rows})
    # time vs density, one panel per (n, distribution)
    fig, axes = plt.subplots(len(ns), 2, figsize=(9.5, 3.4 * len(ns)), sharey="row", squeeze=False)
    for i, n in enumerate(ns):
        for j, dist in enumerate(("uniform", "banded")):
            ax = axes[i][j]
            pts = [r for r in rows if r["n"] == n and r["distribution"] == dist]
            x = [100 * r["density_a"] for r in pts]
            ax.errorbar(x, [r["csr_median_s"] for r in pts], yerr=[r["csr_iqr_s"] / 2 for r in pts],
                        marker=MARKER["csr"], color=COLOR["csr"], label=LABEL["csr"], ms=4, lw=1.3)
            ax.errorbar(x, [r["csr_total_median_s"] for r in pts], yerr=[r["csr_total_iqr_s"] / 2 for r in pts],
                        marker=MARKER["csr_total"], color=COLOR["csr_total"], label=LABEL["csr_total"], ms=4, lw=1.3)
            ax.errorbar(x, [r["dense_best_median_s"] for r in pts], yerr=[r["dense_best_iqr_s"] / 2 for r in pts],
                        marker=MARKER["blocked_unroll"], color=COLOR["blocked_unroll"],
                        label="dense: " + LABEL["blocked_unroll"], ms=4, lw=1.3)
            q = [r for r in pts if r["ijk_median_s"]]
            ax.plot([100 * r["density_a"] for r in q], [r["ijk_median_s"] for r in q], marker=MARKER["ijk"],
                    color=COLOR["ijk"], label="dense: " + LABEL["ijk"], ms=4, lw=1.3)
            ax.set(xscale="log", yscale="log", title=f"n = {n}, {dist} nonzeros",
                   xlabel="density of A and B (% nonzero)" if i == len(ns) - 1 else None,
                   ylabel="median time (s)" if j == 0 else None)
            density_axis(ax)
            if i == 0 and j == 0:
                ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "sparse_time_vs_density.png")
    plt.close(fig)
    # output density
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for dist, ls in (("uniform", "-"), ("banded", "--")):
        for n, mk in zip(ns, ("o", "s")):
            pts = [r for r in rows if r["n"] == n and r["distribution"] == dist]
            ax.plot([100 * r["density_a"] for r in pts], [100 * r["density_c"] for r in pts], ls=ls, marker=mk,
                    color=COLOR["csr"] if dist == "uniform" else COLOR["ikj"], label=f"{dist}, n={n}", ms=4)
    dd = np.logspace(-1, np.log10(50), 50) / 100
    for n, lw in zip(ns, (0.8, 1.2)):
        ax.plot(100 * dd, 100 * (1 - (1 - dd ** 2) ** n), ":", color="#777", lw=lw)
    ax.plot([], [], ":", color="#777", label="uniform model 1-(1-d²)ⁿ")
    ax.set(xscale="log", yscale="log", xlabel="input density (% nonzero)", ylabel="density of C = A·B (%)")
    density_axis(ax)
    ax.legend(fontsize=7.5)
    fig.savefig(FIG / "sparse_output_density.png")
    plt.close(fig)
    # memory
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for dist, ls in (("uniform", "-"), ("banded", "--")):
        pts = [r for r in rows if r["n"] == ns[-1] and r["distribution"] == dist]
        x = [100 * r["density_a"] for r in pts]
        ax.plot(x, [r["csr_rss_mb"] for r in pts], ls=ls, marker=MARKER["csr"], color=COLOR["csr"],
                label=f"CSR (CSR input), {dist}", ms=4)
        ax.plot(x, [r["conv_rss_mb"] for r in pts], ls=ls, marker=MARKER["csr_total"], color=COLOR["csr_total"],
                label=f"CSR (dense input + conversion), {dist}", ms=4)
        ax.plot(x, [r["dense_rss_mb"] for r in pts], ls=ls, marker=MARKER["blocked_unroll"],
                color=COLOR["blocked_unroll"], label=f"dense blocked_unroll, {dist}", ms=4)
    ax.set(xscale="log", yscale="log", xlabel="input density (% nonzero)", ylabel="peak RSS (MiB)",
           title=f"n = {ns[-1]}")
    density_axis(ax)
    ax.legend(fontsize=7)
    fig.savefig(FIG / "sparse_memory.png")
    plt.close(fig)


def scale_part():
    samples, runs = load("scale")
    if runs is None:
        return
    rows = []
    for run in runs:
        n, d = int(run["n"]), float(run["density_target"])
        med, iqr = stats(samples[key(run)]["t"])
        rows.append({"series": run["note"].split(";")[0], "n": n, "density_target": d, "status": run["status"],
                     "reps": len(samples[key(run)]["t"]), "nnz_a": run["nnz_a"], "nnz_c": run["nnz_c"],
                     "median_s": med, "iqr_s": iqr,
                     "peak_rss_mb": int(run["peak_rss_kb"]) / 1024 if run["peak_rss_kb"] else None,
                     "csr_bytes_mb": int(run["csr_bytes"]) / 2 ** 20 if run["csr_bytes"] else None,
                     "dense_storage_mb": 3 * 8 * n ** 2 / 2 ** 20, "note": run["note"]})
    write_tables("scale", rows, list(rows[0]), [
        ("Series", lambda r: r["series"]), ("n", lambda r: str(r["n"])),
        ("Density", lambda r: f"{100 * r['density_target']:.4g}%"), ("Status", lambda r: r["status"]),
        ("Reps", lambda r: str(r["reps"])), ("nnz(A)", lambda r: r["nnz_a"]), ("nnz(C)", lambda r: r["nnz_c"]),
        ("Median (s)", lambda r: g(r["median_s"], 3)),
        ("IQR (s)", lambda r: g(r["iqr_s"], 2)),
        ("Peak RSS (MB)", lambda r: f"{r['peak_rss_mb']:.0f}" if r["peak_rss_mb"] else ""),
        ("CSR arrays (MB)", lambda r: f"{r['csr_bytes_mb']:.0f}" if r["csr_bytes_mb"] else ""),
        ("Dense 3·8n² (MB)", lambda r: f"{r['dense_storage_mb']:.0f}")])
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for s, c in zip(sorted({r["series"] for r in rows}), (COLOR["csr"], COLOR["ikj"])):
        pts = [r for r in rows if r["series"] == s and r["median_s"]]
        name = {"uniform_d0.001": "uniform, density 0.1 %", "uniform_16_per_row": "uniform, 16 nonzeros per row"}
        ax.plot([r["n"] for r in pts], [r["median_s"] for r in pts], marker="o", color=c, label=name.get(s, s), ms=4)
        for r in rows:
            if r["series"] == s and r["status"] == "memory_failure":
                ax.axvline(r["n"], color=c, ls=":", lw=1)
                ax.text(r["n"], ax.get_ylim()[0] * 3, " memory\n failure", color=c, fontsize=7)
    ax.set(xscale="log", yscale="log", xlabel="n (matrix side)", ylabel="median CSR kernel time (s)")
    ax.legend(fontsize=8)
    fig.savefig(FIG / "sparse_scale.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    blocks_part()
    dense_part()
    sparse_part()
    scale_part()
    print("summaries and figures written to", RES)


if __name__ == "__main__":
    main()
