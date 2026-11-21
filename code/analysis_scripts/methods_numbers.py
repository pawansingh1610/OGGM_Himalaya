#!/usr/bin/env python
"""Counts quoted in the Data and Methods sections of Draft_2, written to tables/methods_numbers.txt.
Reads per_forcing/<F>/metadata.csv and failures_all.csv (External2, read-only) and analysis/*.csv."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import COMPILED as C  # noqa: E402
FORC = ["ERA5", "ERA5-Land", "HARv2", "CRU_TS4", "CHELSA"]
out = []
say = out.append

lk = pd.read_csv(ROOT / "analysis/glacier_lookup.csv")
say(f"RGI6 glaciers in regions 13-15: {len(lk):,}; area {lk.area_rgi_km2.sum():,.1f} km2")
for r, d in lk.groupby("rgi_region"):
    say(f"  {r}: {len(d):,} glaciers, {d.area_rgi_km2.sum():,.1f} km2")
say(f"  TermType counts: {lk.TermType.value_counts().to_dict()}; SurgeType counts: {lk.SurgeType.value_counts().to_dict()}")

fail = pd.read_csv(C / "failures_all.csv")
say(f"failed glacier-runs: {len(fail)} ({fail.rgi_id.unique().tolist()}): {fail.error.iloc[0][:120]}")

cols = ["rgi_id", "dem_source", "flowline_type", "volume_source", "calib_period", "calib_status", "first_mb_year",
        "end_year", "n_filled_months", "A_ref_km2", "V_ref_km3", "melt_f", "lapse_rate", "region", "prior_sigma_t",
        "bounds_temp_bias", "bounds_prcp_fac", "temp_var", "prcp_var", "resolution"]
tot = 0
for f in FORC:
    m = pd.read_csv(C / "per_forcing" / f / "metadata.csv", usecols=cols)
    tot += len(m)
    say(f"{f}: n={len(m):,}  years {m.first_mb_year.min()}-{m.end_year.max()}  calib_period {m.calib_period.value_counts().to_dict()}")
    say(f"   DEM {m.dem_source.value_counts().to_dict()}  flowlines {m.flowline_type.value_counts().to_dict()}  "
        f"volume {m.volume_source.value_counts().to_dict()}  filled months>0: {(m.n_filled_months > 0).sum()}")
    say(f"   prior_sigma_t {m.prior_sigma_t.unique().tolist()}  bounds dT {m.bounds_temp_bias.unique().tolist()}  pf {m.bounds_prcp_fac.unique().tolist()}")
    if f == "ERA5":
        say(f"   V_ref (consensus/Bahr, all glaciers) = {m.V_ref_km3.sum():,.1f} km3")
        say(f"   melt_f by region {m.groupby('region').melt_f.unique().apply(list).to_dict()}  "
            f"lapse {m.groupby('region').lapse_rate.unique().apply(list).to_dict()}")
say(f"glacier-runs OK: {tot:,} of {len(lk) * len(FORC):,}")

fv = pd.read_csv(ROOT / "analysis/D_validation_pairs.csv")
say(f"field validation pairs file: {len(fv):,} glacier-years, {fv.rgi_id.nunique()} glaciers with RGI id, "
    f"years {fv.year.min()}-{fv.year.max()}")
dg = pd.read_csv(ROOT / "analysis/D_geodetic_vs_glaciological.csv")
say(f"glaciological vs geodetic: {len(dg)} glaciers with >= 3 years in 2000-2019")

txt = "\n".join(out)
print(txt)
(ROOT / "tables/methods_numbers.txt").write_text(txt + "\n")
