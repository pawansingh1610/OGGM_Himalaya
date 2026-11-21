#!/usr/bin/env python
"""Run every figure script (fig01 ... fig14) in order; stop at the first failure.
Usage: python scripts/figures/make_all.py [| tee figures/make_all.log]"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
scripts = sorted(HERE.glob("fig[0-9][0-9]_*.py"))
for s in scripts:
    r = subprocess.run([sys.executable, str(s)], cwd=HERE)
    if r.returncode:
        sys.exit(f"FAILED: {s.name}")
print(f"\nall {len(scripts)} figure scripts done")
