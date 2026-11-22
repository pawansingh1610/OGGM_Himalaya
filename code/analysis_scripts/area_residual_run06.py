#!/usr/bin/env python
"""Exact decomposition of the run_06_areacons mass residual over the calibration period.

Per glacier i (calib ok, not backward-vanished), n years of the calibration period (2000-2019; CHELSA 2000-2009):
  M_i  = sum_t dm_gt(i,t) / n                       modelled mass rate (= sum_t B_t A_t / n)
  T_i  = B_hug,i * A_RGI,i / 1000                   Hugonnet mass rate (Gt/yr)
  Bbar = mean_t B(i,t) over years with area > 0;  A4 = mean_t A(i,t) in run_04_full_fix;  A6 = same in run_06
Residual  sum_i (M_i - T_i) = capped + area_drift + covariance + other, with
  capped      = sum over glaciers with a capped factor of (M_i - T_i)         (factor stopped at 1/1.5 or 1.5)
  and over uncapped glaciers:
  area_drift  = sum Bbar (A6 - A4)        area in run_06 is smaller than the run_04 area the factor was built on
  covariance  = sum (M - Bbar A6)         B more negative when A is smaller (sum_t B_t A_t vs n Bbar Abar)
  other       = sum (Bbar A4 - T)         calibration residual + extinction within the period (should be ~0)
All terms in Gt/yr and in % of the target mass. Output: tables/area_consistency_residual_run06.csv
"""
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from config import OGGM_ROOT  # noqa: E402
from pathlib import Path

import numpy as np
import pandas as pd

OGGM = OGGM_ROOT
R6 = OGGM / "compiled_v3_run_06_areacons" / "per_forcing"
R4 = OGGM / "compiled_v3_run_04_full_fix" / "per_forcing"
TAB = Path(__file__).resolve().parents[1] / "tables"
FORC = ["ERA5", "ERA5-Land", "HARv2", "CRU_TS4", "CHELSA"]
CAL = {f: (2000, 2019) for f in FORC}
CAL["CHELSA"] = (2000, 2009)
TRUE = {"true", "1"}


def per_glacier(path, a0, a1, with_b):
    cols = ["rgi_id", "year", "area_start_km2", "dm_gt"] + (["mb_annual_mwe"] if with_b else [])
    y = pd.read_parquet(path, columns=cols, filters=[("year", ">=", a0), ("year", "<=", a1)])
    n = a1 - a0 + 1
    y["area_start_km2"] = y.area_start_km2.fillna(0.0)
    y["dm_gt"] = y.dm_gt.fillna(0.0)
    g = y.groupby("rgi_id")
    out = pd.DataFrame({"A": g.area_start_km2.sum() / n, "M": g.dm_gt.sum() / n})
    if with_b:
        out["Bbar"] = y[y.area_start_km2 > 0].groupby("rgi_id").mb_annual_mwe.mean()
    return out


def main():
    rows = []
    for f in FORC:
        a0, a1 = CAL[f]
        m = pd.read_csv(R6 / f / "metadata.csv", usecols=["rgi_id", "region", "calib_status", "backward_vanished",
                                                           "area_factor_flag", "geodetic_target_hugonnet",
                                                           "A_ref_km2"]).set_index("rgi_id")
        m = m[(m.calib_status == "ok") & ~m.backward_vanished.astype(str).str.lower().isin(TRUE)
              & m.geodetic_target_hugonnet.notna()]
        g6 = per_glacier(R6 / f / "yearly.parquet", a0, a1, True).reindex(m.index)
        g4 = per_glacier(R4 / f / "yearly.parquet", a0, a1, False).reindex(m.index)
        m["M"], m["A6"], m["Bbar"], m["A4"] = g6.M.fillna(0), g6.A.fillna(0), g6.Bbar.fillna(0), g4.A.fillna(0)
        m["T"] = m.geodetic_target_hugonnet * m.A_ref_km2 / 1000
        cap = m.area_factor_flag != "ok"
        m["capped"] = np.where(cap, m.M - m["T"], 0.0)
        m["area_drift"] = np.where(cap, 0.0, m.Bbar * (m.A6 - m.A4) / 1000)
        m["covariance"] = np.where(cap, 0.0, m.M - m.Bbar * m.A6 / 1000)
        m["other"] = np.where(cap, 0.0, m.Bbar * m.A4 / 1000 - m["T"])
        for r, g in list(m.groupby("region")) + [("ALL_HMA", m)]:
            T, M = g["T"].sum(), g.M.sum()
            d = dict(forcing=f, region=r, period=f"{a0}-{a1}", n=len(g), n_capped=int((g.area_factor_flag != "ok").sum()),
                     target_Gt_yr=T, model_Gt_yr=M, residual_Gt_yr=M - T, residual_pct=100 * (M / T - 1))
            for k in ["capped", "area_drift", "covariance", "other"]:
                d[f"{k}_Gt_yr"] = g[k].sum()
                d[f"{k}_pp"] = 100 * g[k].sum() / T       # same sign as residual_pct; terms sum to it
            rows.append(d)
        print(f, "done", flush=True)
    D = pd.DataFrame(rows)
    chk = D.residual_Gt_yr - D[[c for c in D if c.endswith("_Gt_yr") and c not in
                                ("target_Gt_yr", "model_Gt_yr", "residual_Gt_yr")]].sum(axis=1)
    assert chk.abs().max() < 1e-6, chk.abs().max()
    D.to_csv(TAB / "area_consistency_residual_run06.csv", index=False)
    pd.set_option("display.width", 250)
    print(D[["forcing", "region", "target_Gt_yr", "model_Gt_yr", "residual_pct", "capped_pp", "area_drift_pp",
             "covariance_pp", "other_pp"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
