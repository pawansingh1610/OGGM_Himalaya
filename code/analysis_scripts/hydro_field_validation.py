#!/usr/bin/env python
"""Table 5 with hydrological years (Oct-Sep) instead of calendar years, from run_h01_field_monthly.

Hydrological year y = Oct (y-1) .. Sep (y), matched to the field year y (WGMS labels by the end year). Calendar values are
the Jan-Dec sums of the same monthly output (= run_07 mb_annual_mwe, checked in the run). Metrics as 06_ANALYSIS_v3._metrics.
Sets: (a) common_all5 and all_pairs as in Table 5 (calendar); (b) the same with hydrological years; (c) like-for-like:
glacier-years with both calendar and hydrological values for all five forcings.
Output: context/reports/external2_round3/hydro_*.csv. Env oggm."""
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from config import OGGM_ROOT, CODE_DIR  # noqa: E402
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
RUN = OGGM_ROOT / "run_h01_field_monthly"
OUT = ROOT / "context/reports/external2_round3"; OUT.mkdir(parents=True, exist_ok=True)
FORC = [f for f in ["ERA5", "ERA5-Land", "HARv2", "CRU_TS4", "CHELSA"] if (RUN / f"monthly_{f}.csv").exists()]
P = pd.read_csv(ROOT / "analysis/D_validation_pairs.csv").dropna(subset=["rgi_id"])[["rgi_id", "year", "obs", "subregion"]]


def metrics(d, col):
    r = d[col] - d.obs
    ao = d.obs - d.groupby("rgi_id").obs.transform("mean"); am = d[col] - d.groupby("rgi_id")[col].transform("mean")
    return pd.Series(dict(N=len(d), n_glaciers=d.rgi_id.nunique(), bias=r.mean(), RMSE=np.sqrt((r ** 2).mean()),
                          r=np.corrcoef(d.obs, d[col])[0, 1], r_anomaly=np.corrcoef(ao, am)[0, 1],
                          std_mod=d[col].std(), std_obs=d.obs.std()))


for f in FORC:
    m = pd.read_csv(RUN / f"monthly_{f}.csv")
    cal = m.groupby(["rgi_id", "year"]).mb_mwe.agg(["sum", "count"])
    cal = cal.loc[cal["count"] == 12, "sum"].rename(f"{f}|cal")
    m["hy"] = m.year + (m.month >= 10)
    hyd = m.groupby(["rgi_id", "hy"]).mb_mwe.agg(["sum", "count"])
    hyd = hyd.loc[hyd["count"] == 12, "sum"].rename(f"{f}|hyd"); hyd.index.names = ["rgi_id", "year"]
    P = P.merge(cal.reset_index(), on=["rgi_id", "year"], how="left").merge(hyd.reset_index(), on=["rgi_id", "year"], how="left")
P.to_csv(OUT / "hydro_pairs.csv", index=False)

old = pd.read_csv(ROOT / "analysis/D_validation_pairs.csv").dropna(subset=["rgi_id"])
chk = P.merge(old, on=["rgi_id", "year"], suffixes=("", "_t5"))
print("check: calendar from monthly vs Table-5 pairs, max |diff| per forcing:",
      {f: float((chk[f"{f}|cal"] - chk[f]).abs().max()) for f in FORC})

rows = []
calc = [f"{f}|cal" for f in FORC]; hydc = [f"{f}|hyd" for f in FORC]
for kind in ["cal", "hyd"]:
    cols = [f"{f}|{kind}" for f in FORC]
    for sname, need in [("all_pairs", None), ("common_all5", cols), ("like_for_like", calc + hydc)]:
        for f in FORC:
            c = f"{f}|{kind}"
            d = P.dropna(subset=[c] if need is None else need)
            rows.append(metrics(d, c).rename((sname, kind, f)))
T = pd.DataFrame(rows); T.index = pd.MultiIndex.from_tuples(T.index, names=["set", "year_type", "forcing"])
T.to_csv(OUT / "hydro_table5.csv")
pd.set_option("display.width", 250)
print(T.round(3).to_string())
# difference hyd - cal on like-for-like
L = T.xs("like_for_like", level="set")
print("\nhyd - cal (like-for-like):\n", (L.xs("hyd", level="year_type") - L.xs("cal", level="year_type"))[["bias", "RMSE", "r", "r_anomaly", "std_mod"]].round(3).to_string())
