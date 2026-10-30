"""Build the dense and sparse programs into build\\dense.exe and build\\sparse.exe.

Compiler: GCC from MinGW-w64 (for example WinLibs, `winget install
BrechtSanders.WinLibs.POSIX.UCRT`), so that `gcc` works in a terminal. Set the
CC environment variable to use another GCC-compatible compiler.
The flags are the fixed settings of every benchmark run (as in Assignment 1:
-O2, no -march, no -ffast-math). GCC's vectorisation report is saved in
results/ as evidence that the dense loops run as scalar code.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFLAGS = ["-O2", "-std=c11", "-Wall", "-Wextra"]
LIBS = ["-lpsapi"]  # GetProcessMemoryInfo


def main():
    cc = os.environ.get("CC", "gcc")
    if shutil.which(cc) is None:
        sys.exit(f"compiler '{cc}' not found: install MinGW-w64 GCC "
                 "(winget install BrechtSanders.WinLibs.POSIX.UCRT) and open a new terminal")
    (ROOT / "build").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    for name in ("dense", "sparse"):
        src = ROOT / "src" / "c" / f"{name}.c"
        out = ROOT / "build" / f"{name}.exe"
        subprocess.run([cc, *CFLAGS, "-o", str(out), str(src), *LIBS], check=True)
        report = subprocess.run([cc, *CFLAGS, "-fopt-info-vec-optimized", "-c", "-o", os.devnull, str(src)],
                                capture_output=True, text=True)
        (ROOT / "results" / f"vectorisation_{name}.txt").write_text(report.stderr)
        print(f"{cc} {' '.join(CFLAGS)} -> build\\{name}.exe")


if __name__ == "__main__":
    main()
