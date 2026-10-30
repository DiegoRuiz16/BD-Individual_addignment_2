"""Record the hardware and software environment in results/environment.txt
(CPU, cores, caches, RAM, Windows version, power plan, compiler, Python, libraries)."""
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HARDWARE = (
    "Get-CimInstance Win32_Processor | Format-List Name,NumberOfCores,NumberOfLogicalProcessors,"
    "MaxClockSpeed,L2CacheSize,L3CacheSize; "
    "'RAM (GiB): ' + [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB, 1); "
    "'OS: ' + (Get-CimInstance Win32_OperatingSystem).Caption + ' ' + "
    "(Get-CimInstance Win32_OperatingSystem).Version"
)


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "(not available)"


def main():
    import matplotlib
    import numpy
    import scipy
    cc = os.environ.get("CC", "gcc")
    version = run([cc, "--version"])
    lines = [
        f"date: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "== Hardware and OS ==", run(["powershell", "-NoProfile", "-Command", HARDWARE]),
        f"platform: {platform.platform()}",
        "== Power plan ==", run(["powercfg", "/getactivescheme"]),
        "== Toolchain ==", version.splitlines()[0] if version else cc,
        "CFLAGS: -O2 -std=c11 -Wall -Wextra (no -march, no -ffast-math)",
        f"{platform.python_implementation()} {platform.python_version()}",
        f"numpy {numpy.__version__} (input generation / reference)",
        f"scipy {scipy.__version__} (reference nnz count only)",
        f"matplotlib {matplotlib.__version__}",
        "== Threads ==", "All benchmarked programs are single-threaded C (no OpenMP, no BLAS).",
    ]
    text = "\n".join(lines) + "\n"
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "environment.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
