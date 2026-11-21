#!/usr/bin/env python
"""Build compiled_v3_run_04_full_fix = run_04_full with ERA5 x RGI60-13 replaced by run_05_era5_r13_sT4.

Why: in run_04_full, ERA5 x RGI60-13 was calibrated with sigma_T = 1.5 (calibration_v3.py was edited
while that pool was running, see context/FULL_RUN_RESULTS_v2.md). run_05 re-ran that block with sigma_T = 4.

Inputs (read-only): compiled_v3_run_04_full/, compiled_v3_run_05_era5_r13_sT4/ (compile_v2.py output).
Output: compiled_v3_run_04_full_fix/ (per_forcing/<F>/{yearly.parquet,metadata.csv}, metadata_all.csv,
failures_all.csv, compile_report.txt, merge_report.txt, analysis/glacier_lookup.csv copied).
Refuses to overwrite an existing output folder. Asserts prior_sigma_t == 4 for every glacier x forcing.
"""
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from config import OGGM_ROOT  # noqa: E402
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = OGGM_ROOT
OLD = BASE / "compiled_v3_run_04_full"
NEW = BASE / "compiled_v3_run_05_era5_r13_sT4"
OUT = BASE / "compiled_v3_run_04_full_fix"
FORC = ["ERA5", "ERA5-Land", "CRU_TS4", "CHELSA", "HARv2"]
FIX_F, FIX_R = "ERA5", "RGI60-13"

if OUT.exists():
    sys.exit(f"{OUT} exists - refusing to overwrite. Delete it by hand if a rebuild is intended.")
assert OLD != OUT and NEW != OUT
tmp = BASE / (OUT.name + ".partial")
if tmp.exists():
    shutil.rmtree(tmp)
log = []
say = lambda s: (print(s, flush=True), log.append(s))
say(f"merge started {datetime.now():%Y-%m-%d %H:%M}\n  base: {OLD}\n  fix : {NEW} ({FIX_F} x {FIX_R})")

metas = []
for f in FORC:
    od = tmp / "per_forcing" / f
    od.mkdir(parents=True)
    src = OLD / "per_forcing" / f
    if f != FIX_F:
        for fn in ["yearly.parquet", "metadata.csv"]:
            shutil.copy2(src / fn, od / fn)
        m = pd.read_csv(od / "metadata.csv", low_memory=False)
    else:
        mo = pd.read_csv(src / "metadata.csv", low_memory=False)
        mn = pd.read_csv(NEW / "per_forcing" / f / "metadata.csv", low_memory=False)
        assert (mn.rgi_id.str[:8] == FIX_R).all(), "run_05 contains glaciers outside RGI60-13"
        old_ids = set(mo.loc[mo.rgi_id.str[:8] == FIX_R, "rgi_id"])
        new_ids = set(mn.rgi_id)
        say(f"  {f}: run_04 {FIX_R} glaciers {len(old_ids):,}; run_05 {len(new_ids):,}; "
            f"only-old {len(old_ids - new_ids)}, only-new {len(new_ids - old_ids)}")
        assert old_ids == new_ids, "glacier sets differ between run_04 and run_05 for ERA5 x RGI60-13"
        m = pd.concat([mo[mo.rgi_id.str[:8] != FIX_R], mn], ignore_index=True)
        m = m.sort_values("rgi_id").reset_index(drop=True)
        assert m.rgi_id.is_unique
        m.to_csv(od / "metadata.csv", index=False)
        yo = pd.read_parquet(src / "yearly.parquet")
        yn = pd.read_parquet(NEW / "per_forcing" / f / "yearly.parquet")
        assert list(yo.columns) == list(yn.columns), (yo.columns, yn.columns)
        y = pd.concat([yo[yo.rgi_id.str[:8] != FIX_R], yn], ignore_index=True)
        y = y.sort_values(["rgi_id", "year"]).reset_index(drop=True)
        assert not y.duplicated(["rgi_id", "year"]).any()
        assert set(y.rgi_id) == set(m.rgi_id)
        say(f"  {f}: yearly rows old {len(yo):,} -> merged {len(y):,} (run_05 rows {len(yn):,})")
        y.to_parquet(od / "yearly.parquet", index=False)
        del yo, yn, y
    m["forcing_dir"] = f
    metas.append(m)

ma = pd.concat(metas, ignore_index=True)
# metadata_all.csv: same layout as compile_v2 (stacked per-forcing metadata); drop helper column
ma.drop(columns="forcing_dir").to_csv(tmp / "metadata_all.csv", index=False)

# ---- assertion: prior sigma_T = 4 everywhere --------------------------------------------------
ma["region"] = ma.rgi_id.str[:8]
cnt = ma.groupby(["forcing_dir", "region"]).agg(
    n=("rgi_id", "size"), sigma_t=("prior_sigma_t", lambda s: ",".join(map(str, sorted(s.unique())))))
say("\nglacier counts and prior_sigma_t per forcing x region:")
say(cnt.to_string())
bad = ma[ma.prior_sigma_t.astype(float) != 4.0]
say(f"\nglacier x forcing rows: {len(ma):,}; with prior_sigma_t != 4: {len(bad)}")
assert len(bad) == 0, bad.groupby(["forcing_dir", "region"]).size()

# ---- failures: run_04 failures, minus any from the replaced block, plus run_05 failures ----
fo = pd.read_csv(OLD / "failures_all.csv")
fo = fo[~((fo.forcing == FIX_F) & (fo.region == FIX_R))]
fn = NEW / "failures_all.csv"
fl = [fo] + ([pd.read_csv(fn)] if fn.exists() else [])
fa = pd.concat(fl, ignore_index=True)
fa.to_csv(tmp / "failures_all.csv", index=False)
say(f"failures carried: {len(fa)} ({fa.rgi_id.unique().tolist()})")

# ---- reports, lookup cache --------------------------------------------------------------------
rep = (OLD / "compile_report.txt").read_text().splitlines()
new_rep = (NEW / "compile_report.txt").read_text().splitlines()
rep = [l for l in rep if not l.startswith(f"{FIX_F}:")]
rep.insert(0, f"MERGED: {FIX_F} x {FIX_R} from {NEW.name}; everything else from {OLD.name}")
rep += ["--- run_05 compile report ---"] + new_rep
(tmp / "compile_report.txt").write_text("\n".join(rep) + "\n")
(tmp / "analysis").mkdir()
shutil.copy2(OLD / "analysis" / "glacier_lookup.csv", tmp / "analysis" / "glacier_lookup.csv")
say(f"merge finished {datetime.now():%Y-%m-%d %H:%M}")
(tmp / "merge_report.txt").write_text("\n".join(log) + "\n")
tmp.rename(OUT)
print(f"-> {OUT}")
