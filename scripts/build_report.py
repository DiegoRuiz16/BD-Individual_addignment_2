"""Render report/report.md to HTML (pandoc) and PDF (headless Microsoft Edge or Chrome)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPORT = Path(__file__).resolve().parent.parent / "report"
CANDIDATES = [
    os.environ.get("CHROME", ""), "msedge",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]


def browser():
    for c in CANDIDATES:
        if c and (shutil.which(c) or Path(c).is_file()):
            return shutil.which(c) or c
    sys.exit("Microsoft Edge or Chrome not found: set CHROME to the browser executable")


def main():
    subprocess.run(["pandoc", "report.md", "-s", "--css", "style.css", "--embed-resources",
                    "--metadata", "lang=en", "-o", "report.html"], cwd=REPORT, check=True)
    subprocess.run([browser(), "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={REPORT / 'report.pdf'}", (REPORT / "report.html").as_uri()],
                   check=True, stderr=subprocess.DEVNULL)
    print(REPORT / "report.pdf")


if __name__ == "__main__":
    main()
