#!/usr/bin/env python
"""Build compiled_v3_run_07_tpfix = run_06_areacons with ERA5-Land replaced by run_07_era5l_tpfix.

Why: the ERA5-Land monthly precipitation of Sep 2022 - Feb 2024 was wrong (ECMWF known issue); run_07 re-ran ERA5-Land
(all glaciers, same code and settings as run_06: sigma_T = 4, area-consistent targets with the run_04_full_fix factors)
with the corrected months (scripts/era5land_tp_fix.py). Calibration (2000-2019) is not affected by those months, so the
new ERA5-Land must reproduce run_06 up to Aug 2022; this is asserted below (mass balance of every glacier-year up to
2021 within 1e-6 m w.e., parameters identical within 1e-6).
Refuses to overwrite an existing output folder.
"""
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from config import OGGM_ROOT  # noqa: E402
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE = OGGM_ROOT
OLD = BASE / "compiled_v3_run_06_areacons"
NEW = BASE / "compiled_v3_run_07_era5l_tpfix"
OUT = BASE / "compiled_v3_run_07_tpfix"
FORC = ["ERA5", "ERA5-Land", "CRU_TS4", "CHELSA", "HARv2"]
FIX_F = "ERA5-Land"
if OUT.exists():
    sys.exit(f"{OUT} exists - refusing to overwrite.")
tmp = BASE / (OUT.name + ".partial")
if tmp.exists():
    shutil.rmtree(tmp)
log = []
say = lambda s: (print(s, flush=True), log.append(s))
say(f"merge started {datetime.now():%Y-%m-%d %H:%M}\n  base: {OLD}\n  fix : {NEW} ({FIX_F}, all regions)")
metas = []
for f in FORC:
    od = tmp / "per_forcing" / f
    od.mkdir(parents=True)
    if f != FIX_F:
        for fn in ["yearly.parquet", "metadata.csv"]:
            shutil.copy2(OLD / "per_forcing" / f / fn, od / fn)
        m = pd.read_csv(od / "metadata.csv", low_memory=False)
    else:
        mo = pd.read_csv(OLD / "per_forcing" / f / "metadata.csv", low_memory=False)
        mn = pd.read_csv(NEW / "per_forcing" / f / "metadata.csv", low_memory=False)
        assert set(mo.rgi_id) == set(mn.rgi_id), "glacier sets differ between run_06 and run_07 for ERA5-Land"
        j = mo.merge(mn, on="rgi_id", suffixes=("_o", "_n"))
        for c in ["temp_bias", "prcp_fac"]:
            d = (j[c + "_o"] - j[c + "_n"]).abs().max()
            say(f"  {f}: max |{c} run_06 - run_07| = {d:.3e}")
            assert d < 1e-6, f"calibrated {c} changed - the rerun is not equivalent up to 2019"
        assert (mn.prior_sigma_t.astype(float) == 4.0).all() and mn.area_consistent.all()
        yo = pd.read_parquet(OLD / "per_forcing" / f / "yearly.parquet")
        yn = pd.read_parquet(NEW / "per_forcing" / f / "yearly.parquet")
        assert set(yo.columns) == set(yn.columns)
        yn = yn[list(yo.columns)]
        a = yo.set_index(["rgi_id", "year"]).mb_annual_mwe
        b = yn.set_index(["rgi_id", "year"]).mb_annual_mwe
        j2 = pd.concat([a.rename("o"), b.rename("n")], axis=1)
        j2["d"] = (j2.o - j2.n).abs()
        by = j2.groupby(level="year").d.max()
        say("  max |mb_annual_mwe run_06 - run_07| by year (2018-2025): " + ", ".join(f"{y}: {by[y]:.2e}" for y in range(2018, 2026)))
        assert by[by.index <= 2021].max() < 1e-6, "mass balance changed before 2022 - rerun not equivalent"
        assert set(yo.rgi_id) == set(yn.rgi_id)
        yn.sort_values(["rgi_id", "year"]).reset_index(drop=True).to_parquet(od / "yearly.parquet", index=False)
        mn.sort_values("rgi_id").reset_index(drop=True).to_csv(od / "metadata.csv", index=False)
        m = mn
        del yo, yn, j2
    m["forcing_dir"] = f
    metas.append(m)
ma = pd.concat(metas, ignore_index=True)
ma.drop(columns="forcing_dir").to_csv(tmp / "metadata_all.csv", index=False)
bad = ma[ma.prior_sigma_t.astype(float) != 4.0]
assert len(bad) == 0
fo = pd.read_csv(OLD / "failures_all.csv")
fo = fo[fo.forcing != FIX_F]
fn = NEW / "failures_all.csv"
fa = pd.concat([fo] + ([pd.read_csv(fn)] if fn.exists() else []), ignore_index=True)
fa.to_csv(tmp / "failures_all.csv", index=False)
rep = [l for l in (OLD / "compile_report.txt").read_text().splitlines() if not l.startswith(f"{FIX_F}:")]
rep.insert(0, f"MERGED: {FIX_F} from {NEW.name}; everything else from {OLD.name}")
rep += ["--- run_07 compile report ---"] + (NEW / "compile_report.txt").read_text().splitlines()
(tmp / "compile_report.txt").write_text("\n".join(rep) + "\n")
(tmp / "analysis").mkdir()
shutil.copy2(OLD / "analysis" / "glacier_lookup.csv", tmp / "analysis" / "glacier_lookup.csv")
say(f"merge finished {datetime.now():%Y-%m-%d %H:%M}")
(tmp / "merge_report.txt").write_text("\n".join(log) + "\n")
tmp.rename(OUT)
print(f"-> {OUT}")
