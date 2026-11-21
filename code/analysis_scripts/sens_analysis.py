#!/usr/bin/env python
"""Analysis of the 681-glacier sensitivity runs (scripts/sens_chain.sh). Env `oggm`.
For each run and forcing (glaciers: calib ok, not backward_vanished, calibration 2000-2019 or the run's own period):
  v8099   volume change 1980-1999, % of V_1999 of the same glaciers (sum dm_gt/0.9 over the window)
  b8099   specific mass balance 1980-1999 of the same glaciers (m w.e. yr-1): mean over the years of 1000 * sum(dm_gt) / sum(area_start_km2)
  err10s  area-weighted mean of (modelled - observed Hugonnet) 2010-2019 specific MB (m w.e. yr-1); rmse10s over glaciers
  acc     captured fraction of the observed 2010s-2000s acceleration (d_mod/d_obs, area-weighted)
Reference = run_s01_base. Writes tables/sens_runs.csv and prints delta tables."""
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from config import OGGM_ROOT  # noqa: E402
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OGGM = OGGM_ROOT
RUNS = {"run_s01_base": "baseline", "run_s02_hindcast": "calibrated 2000-2009 only", "run_s03_melt_p20": "melt factor +20 %",
        "run_s04_melt_m20": "melt factor -20 %", "run_s05_lapse_p001": "lapse rate +0.001 K/m", "run_s06_lapse_m001": "lapse rate -0.001 K/m", "run_s07_rev_hindcast": "calibrated 2010-2019 only (reverse hindcast)"}
FORC = ["ERA5", "ERA5-Land", "HARv2", "CRU_TS4", "CHELSA"]
rows = []
for run, lab in RUNS.items():
    base = OGGM / f"compiled_v3_{run}" / "per_forcing"
    if not base.exists():
        print("missing", run); continue
    for f in FORC:
        if not (base / f / "metadata.csv").exists(): continue        # e.g. CHELSA-W5E5 is not run in the reverse hindcast
        m = pd.read_csv(base / f / "metadata.csv", low_memory=False)
        y = pd.read_parquet(base / f / "yearly.parquet", columns=["rgi_id", "year", "dm_gt", "volume_km3", "area_start_km2"])
        ok = m[(m.calib_status == "ok") & ~m.backward_vanished.astype(bool)]
        ids = set(ok.rgi_id)
        yy = y[y.rgi_id.isin(ids)]
        dv = yy[yy.year.between(1980, 1999)].groupby("rgi_id").dm_gt.sum() / 0.9
        v99 = yy[yy.year == 1999].set_index("rgi_id").volume_km3
        gl = dv.index.intersection(v99.index)
        v8099 = 100 * dv[gl].sum() / v99[gl].sum()
        w8 = yy[yy.year.between(1980, 1999) & yy.rgi_id.isin(gl)].groupby("year")[["dm_gt", "area_start_km2"]].sum()
        b8099 = float((1000 * w8.dm_gt / w8.area_start_km2).mean())
        d = ok.dropna(subset=["hug_obs_2010_2020", "hug_mod_2010_2020", "hug_obs_2000_2010", "hug_mod_2000_2010"])
        if len(d) == 0:          # CHELSA-W5E5 ends in 2016: no 2010-2019 comparison
            rows.append(dict(run=run, label=lab, forcing=f, n=0, v8099_pct=v8099, b8099_mwe=b8099, calib_period=ok.calib_period.mode().iloc[0])); continue
        w = d.A_ref_km2
        big = d[d.A_ref_km2 >= 1.0]
        r_gl = float(np.corrcoef(big.hug_mod_2010_2020, big.hug_obs_2010_2020)[0, 1]) if len(big) > 10 else np.nan
        err = np.average(d.hug_mod_2010_2020 - d.hug_obs_2010_2020, weights=w)
        rmse = float(np.sqrt(np.average((d.hug_mod_2010_2020 - d.hug_obs_2010_2020) ** 2, weights=w)))
        acc = np.average(d.hug_mod_2010_2020 - d.hug_mod_2000_2010, weights=w) / np.average(d.hug_obs_2010_2020 - d.hug_obs_2000_2010, weights=w)
        rows.append(dict(run=run, label=lab, forcing=f, n=len(d), v8099_pct=v8099, b8099_mwe=b8099,
                         obs_2000s=np.average(d.hug_obs_2000_2010, weights=w), obs_2010s=np.average(d.hug_obs_2010_2020, weights=w),
                         mod_2000s=np.average(d.hug_mod_2000_2010, weights=w), mod_2010s=np.average(d.hug_mod_2010_2020, weights=w), r_glacier_2010s=r_gl, err10s=err, rmse10s=rmse, acc_captured=acc,
                         dT_median=m.loc[m.calib_status == "ok", "temp_bias"].median(), pf_median=m.loc[m.calib_status == "ok", "prcp_fac"].median(),
                         calib_period=ok.calib_period.mode().iloc[0]))
T = pd.DataFrame(rows); T.to_csv(ROOT / "tables" / "sens_runs.csv", index=False)
pd.set_option("display.width", 250)
for col in ("v8099_pct", "obs_2010s", "mod_2010s", "err10s", "rmse10s", "r_glacier_2010s", "acc_captured", "dT_median", "pf_median"):
    P = T.pivot_table(index="label", columns="forcing", values=col).reindex(columns=FORC)
    print(f"\n== {col}\n", P.round(3).to_string())
b = T[T.run == "run_s01_base"].set_index("forcing")
print("\n== delta of v8099 (pp) to baseline, and spread across forcings (max-min) per run")
for run, lab in RUNS.items():
    t = T[T.run == run].set_index("forcing")
    if t.empty or run == "run_s01_base": continue
    print(f"{lab:28s}", (t.v8099_pct - b.v8099_pct).round(2).to_dict(), "spread", round(t.v8099_pct.max() - t.v8099_pct.min(), 2), "vs base", round(b.v8099_pct.max() - b.v8099_pct.min(), 2))

# ---- LaTeX table for the Supplement (tables/S13_sensitivity.tex)
sg = lambda v, nd=2: ("$+$" if v > 0 else "$-$") + f"{abs(v):.{nd}f}" if abs(v) >= 0.5 * 10 ** -nd else f"{0:.{nd}f}"   # sign from v (was a fixed 0.005 threshold, wrong for nd=3)
ORDER = [("run_s01_base", "Baseline"), ("run_s03_melt_p20", "Melt factor $+20$\\,\\%"), ("run_s04_melt_m20", "Melt factor $-20$\\,\\%"),
         ("run_s05_lapse_p001", "Lapse rate $+0.001$\\,K\\,m$^{-1}$"), ("run_s06_lapse_m001", "Lapse rate $-0.001$\\,K\\,m$^{-1}$"),
         ("run_s02_hindcast", "Calibrated to 2000--2009"), ("run_s07_rev_hindcast", "Calibrated to 2010--2019")]
sgn = lambda v, nd=2: "--" if pd.isna(v) else sg(v, nd)
Lx = ["% generated by scripts/sens_analysis.py -- do not edit by hand", "\\begin{table}[t]",
      "\\caption{Sensitivity runs on the 681-glacier test subset. (a) Volume change 1980--1999 in per cent of the subset's $V_{1999}$ (change against the baseline in percentage points), and the spread between forcings; (a$'$) the 1980--1999 mass balance of the runs calibrated to the whole period or to one decade (change against the baseline). CHELSA-W5E5 is not run with the 2010--2019 calibration. (b) Validation run (calibrated to 2000--2009) and (c) hindcast (calibrated to 2010--2019), against the baseline (calibrated to 2000--2019): modelled mass balance of the decade left out (observed: $-0.263$ and $-0.150$\\,m\\,w.e.\\,yr$^{-1}$) and share of the observed change between the decades that the model reproduces. CHELSA-W5E5 ends in 2016 and is not tested in (b) and (c).}",
      "\\label{tab:sens}", "\\scriptsize", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lrrrrrr}", "\\tophline",
      "(a) Run & ERA5 & ERA5-Land & HARv2 & CRU~TS4 & CHELSA & Spread \\\\", "\\middlehline"]
V = T.pivot_table(index="run", columns="forcing", values="v8099_pct").reindex(columns=FORC)
for run, name in ORDER:
    if run not in V.index: continue
    r = V.loc[run]; d = r - V.loc["run_s01_base"]
    cells = [sgn(r[f]) + ("" if run == "run_s01_base" or pd.isna(r[f]) else f" ({sg(d[f])})") for f in FORC]
    Lx.append(f"{name} & " + " & ".join(cells) + f" & {r.max() - r.min():.2f} \\\\")
# 1980-1999 specific mass balance for the calibration-window runs (round-3 review: size of the calibration-window effect)
Bm = T.pivot_table(index="run", columns="forcing", values="b8099_mwe").reindex(columns=FORC)
Lx += ["\\middlehline", "(a$'$) 1980--1999 MB (m\\,w.e.\\,yr$^{-1}$) & ERA5 & ERA5-Land & HARv2 & CRU~TS4 & CHELSA & \\\\", "\\middlehline"]
for run, name in (("run_s01_base", "Baseline"), ("run_s02_hindcast", "Calibrated to 2000--2009"), ("run_s07_rev_hindcast", "Calibrated to 2010--2019")):
    r = Bm.loc[run]; d = r - Bm.loc["run_s01_base"]
    Lx.append(f"{name} & " + " & ".join(sgn(r[f], 3) + ("" if run == "run_s01_base" or pd.isna(r[f]) else f" ({sg(d[f], 3)})") for f in FORC) + " & \\\\")
for panel, run, dec, col in (("(b) Validation run", "run_s02_hindcast", "2010--2019", "mod_2010s"), ("(c) Hindcast", "run_s07_rev_hindcast", "2000--2009", "mod_2000s")):
    Lx += ["\\middlehline", f"{panel} & ERA5 & ERA5-Land & HARv2 & CRU~TS4 & & \\\\", "\\middlehline"]
    for r_, name in (("run_s01_base", "baseline"), (run, "this run")):
        t = T[(T.run == r_) & (T.n > 0)].set_index("forcing")
        Lx.append(f"{dec} MB, {name} & " + " & ".join(sg(t.loc[f, col], 3) for f in FORC[:4]) + " & & \\\\")
    for r_, name in (("run_s01_base", "baseline"), (run, "this run")):
        t = T[(T.run == r_) & (T.n > 0)].set_index("forcing")
        Lx.append(f"Share of observed change (\\%), {name} & " + " & ".join(f"{100 * t.loc[f, 'acc_captured']:.0f}" for f in FORC[:4]) + " & & \\\\")
Lx += ["\\bottomhline", "\\end{tabular}}", "\\end{table}"]
(ROOT / "tables" / "S13_sensitivity.tex").write_text("\n".join(Lx) + "\n")
print("wrote tables/S13_sensitivity.tex")
