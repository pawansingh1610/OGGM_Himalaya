"""Single place for the compiled run used by every script (read-only, on External2)."""
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from config import OGGM_ROOT  # noqa: E402
from pathlib import Path

RUN = "run_07_tpfix"   # run_06 settings, ERA5-Land re-run with corrected tp (Sep 2022-Feb 2024); other four forcings = run_06
COMPILED = OGGM_ROOT / f"compiled_v3_{RUN}"
PER_FORCING = COMPILED / "per_forcing"
