#!/usr/bin/env python
"""Area-consistency factors for run_06_areacons.

For every glacier x forcing of the reference run (run_04_full_fix):
    factor = A_RGI / mean(A_start over the forcing's calibration period)
where A_start is the V-A area at the start of each year (0 after extinction) and the calibration period is the
one stored in metadata (2000-2019; CHELSA 2000-2009). The pipeline (OGGM_V3_AREA_CONSISTENT=1) multiplies the
Hugonnet per-glacier target (specific MB on the fixed RGI area) by this factor, so that the modelled mass
change over the calibration period, sum_t B_t A_t, matches target * A_RGI.
The factor is capped to [1/FMAX, FMAX] (glaciers that vanish early in the period have a tiny mean area; their mass
loss is limited by the ice available anyway); capped rows are flagged.
Glaciers missing from the reference run get factor 1 (flag 'missing' in the pipeline).

Output: <calibration_output_v3>/area_factors_<ref run>.csv  (rgi_id, forcing, calib_period, A_RGI_km2,
A_mean_cal_km2, factor_raw, factor, capped) and a summary printed to stdout.
"""
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from config import OGGM_ROOT  # noqa: E402
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paths import PER_FORCING, RUN  # noqa: E402

FMAX = 1.5
OUT = OGGM_ROOT / "calibration_output_v3" / f"area_factors_{RUN}.csv"
FORC = ["ERA5", "ERA5-Land", "CRU_TS4", "CHELSA", "HARv2"]


def main():
    if OUT.exists():
        sys.exit(f"{OUT} exists - refusing to overwrite (the run may be using it).")
    parts = []
    for f in FORC:
        m = pd.read_csv(PER_FORCING / f / "metadata.csv", usecols=["rgi_id", "A_ref_km2", "calib_period"])
        cp = m.calib_period.iloc[0]
        assert (m.calib_period == cp).all()
        y0, y1 = int(cp[:4]), int(cp[11:15]) - 1          # '2000-01-01_2020-01-01' -> 2000..2019
        y = pd.read_parquet(PER_FORCING / f / "yearly.parquet", columns=["rgi_id", "year", "area_start_km2"])
        y = y[y.year.between(y0, y1)]
        n = y.groupby("rgi_id").year.nunique()
        a = y.groupby("rgi_id").area_start_km2.apply(lambda s: s.fillna(0).sum()) / (y1 - y0 + 1)
        d = m.set_index("rgi_id").join(a.rename("A_mean_cal_km2")).join(n.rename("n_years"))
        assert (d.n_years == y1 - y0 + 1).all(), f"{f}: incomplete calibration years"
        d["forcing"], d["calib_period"] = f, cp
        d["factor_raw"] = np.where(d.A_mean_cal_km2 > 0, d.A_ref_km2 / d.A_mean_cal_km2, np.inf)
        d["factor"] = d.factor_raw.clip(1 / FMAX, FMAX)
        d["capped"] = d.factor != d.factor_raw
        parts.append(d.reset_index()[["rgi_id", "forcing", "calib_period", "A_ref_km2", "A_mean_cal_km2",
                                      "factor_raw", "factor", "capped"]].rename(columns={"A_ref_km2": "A_RGI_km2"}))
        q = d.factor_raw.replace(np.inf, np.nan)
        print(f"{f:10s} {cp}  n={len(d):,}  factor median {q.median():.4f}  p1 {q.quantile(.01):.4f}  "
              f"p99 {q.quantile(.99):.4f}  max {q.max():.2f}  capped {int(d.capped.sum())} "
              f"(area {d.loc[d.capped, 'A_ref_km2'].sum():.1f} km2)  area-weighted {np.average(d.factor, weights=d.A_ref_km2):.4f}",
              flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"-> {OUT}  ({len(out):,} rows)")


if __name__ == "__main__":
    main()
